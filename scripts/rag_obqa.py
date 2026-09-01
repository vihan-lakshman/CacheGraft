"""OpenBookQA-RAG: retrieve a science fact and answer the multiple-choice question.

C2C is on home turf here: short (single-letter) answers in the fuser's trained regime,
a knowledge bottleneck that retrieval fills, and the exact task C2C beat the receiver on
(39->53) in the paper reproduction. Uses the proven 4b_base fuser.

methods x fact:
  receiver  none|retrieved|oracle : Qwen3-0.6B
  text_rag  retrieved|oracle      : Qwen3-0.6B, fact prepended as TEXT
  c2c_rag   retrieved|oracle      : RosettaModel, 4B-Base KV fused in.
              --fuse all       : [fact+Q+choices] fused (C2C's original; sharer sees the question)
              --fuse fact_only : ONLY [preamble+fact] fused -> query-independent, precomputable
  sharer    none|retrieved|oracle : Qwen3-4B-Base (upper bound)
"""
import os, json, argparse, time
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import common as C
from rosetta.utils.evaluate import build_prompt as repo_build_prompt, extract_answer_from_content

HERE = os.path.dirname(__file__)
SHARER_BASE = C.SHARER_BASE  # proven 4b_base fuser (reproduced OpenBookQA C2C = 53.4)
LETTERS = ["A", "B", "C", "D", "E"]

def fmt_choices(ch):
    texts, labels = ch["text"], ch["label"]
    return "".join(f"{l}. {t}\n" for l, t in zip(labels, texts)), labels

def build_prompt(stem, ch_str, fact=None):
    # Inject the retrieved fact into the question slot; keep the repo's proven template
    q = f"Relevant fact: {fact}\n\n{stem}" if fact else stem
    return repo_build_prompt("mmlu-redux", "", q, ch_str, use_cot=False, use_template=True)

def parse_letter(text, labels):
    return extract_answer_from_content(text)

def retrieve_facts(test_stems, corpus_facts, corpus_emb):
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    q = em.encode(["Represent this sentence for searching relevant passages: " + s
                   for s in test_stems], normalize_embeddings=True, batch_size=128)
    idx = (q @ corpus_emb.T).argmax(1)
    return [corpus_facts[i] for i in idx]

def run_hf(model_name, prompts, batch=64, max_new=16):
    tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                  add_generation_prompt=True, enable_thinking=False) for p in prompts[i:i+batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del model; torch.cuda.empty_cache()
    return outs

def run_c2c(prompts, max_new=16, fuse="all", stems=None):
    """fuse='all'   : fuse the whole prompt [fact+question+choices] (C2C's original setting;
                      the sharer sees the question, so the cache is NOT precomputable).
       fuse='fact_only': fuse only [preamble+fact] and leave [question+choices] un-fused, so the
                      fused cache depends on the retrieved fact alone and can be built offline
                      -- the OpenBookQA analogue of doc-only fusion in rag_trivia.py."""
    from rosetta.utils.evaluate import load_rosetta_model
    mc = {"model_name": "Rosetta", "rosetta_config": {
        "base_model": C.RECEIVER, "teacher_model": SHARER_BASE,
        "is_do_alignment": False, "alignment_strategy": "first",
        "checkpoints_dir": C.get_fuser_dir()}}
    model, tok = load_rosetta_model(mc, {}, device=torch.device("cuda"),
                                    generation_config={"do_sample": False, "max_new_tokens": max_new})

    def sec(flag, n):
        return torch.tensor([[flag, 0]] * n).unsqueeze(0).cuda()

    outs = []
    for i, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to("cuda") for k, v in enc.items()}
        L = enc["input_ids"].shape[1]
        if fuse == "fact_only":
            # Fusion is applied PER SECTION, so fact-only fusion needs three sections:
            # [preamble+fact = fuse, question+choices = no-fuse, generation].
            c_q = text.rfind(stems[i])
            L_doc = sum(1 for k in range(L) if 0 < offs[k][1] <= c_q) if c_q > 0 else L - 1
            L_doc = max(1, min(L_doc, L - 2))          # leave room for question + generation
            idx = [sec(1, L_doc), sec(-1, (L - 1) - L_doc), sec(-1, 1)]
        else:
            idx = [sec(1, L - 1), sec(-1, 1)]
        with torch.no_grad():
            o = model.generate(**enc, kv_cache_index=idx,
                               do_sample=False, max_new_tokens=max_new)
        outs.append(tok.decode(o[0, L:], skip_special_tokens=True))
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["receiver", "text_rag", "c2c_rag", "sharer"])
    ap.add_argument("--fact", default="none", choices=["none", "retrieved", "oracle"])
    ap.add_argument("--fuse", default="all", choices=["all", "fact_only"],
                    help="c2c_rag only. 'all' = fuse [fact+question+choices] (C2C's original, "
                         "NOT precomputable). 'fact_only' = fuse [preamble+fact] alone, leaving "
                         "the question un-fused, so the fused cache is query-independent and can "
                         "be built offline (the cacheable configuration this paper claims).")
    ap.add_argument("--n_test", type=int, default=500)
    args = ap.parse_args()

    add = load_dataset("allenai/openbookqa", "additional")
    test = add["test"].select(range(min(args.n_test, len(add["test"]))))

    # corpus = unique gold facts across all splits (approximates the 1326-fact open book)
    corpus = sorted({f for sp in add.values() for f in sp["fact1"]})
    print(f"fact corpus size: {len(corpus)}", flush=True)

    stems = [e["question_stem"] for e in test]
    facts = None
    retr_hit = None
    if args.fact == "oracle":
        facts = [e["fact1"] for e in test]
    elif args.fact == "retrieved":
        from sentence_transformers import SentenceTransformer
        em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        cemb = em.encode(corpus, normalize_embeddings=True, batch_size=256)
        facts = retrieve_facts(stems, corpus, cemb)
        retr_hit = float(np.mean([f == e["fact1"] for f, e in zip(facts, test)]))
        print(f"retrieval exact-fact hit rate: {retr_hit:.3f}", flush=True)

    prompts, all_labels, golds = [], [], []
    for i, e in enumerate(test):
        ch_str, labels = fmt_choices(e["choices"])
        fact = facts[i] if facts else None
        prompts.append(build_prompt(e["question_stem"], ch_str, fact))
        all_labels.append(labels); golds.append(e["answerKey"])

    t0 = time.time()
    if args.method == "c2c_rag":
        gens = run_c2c(prompts, fuse=args.fuse, stems=stems)
    elif args.method == "sharer":
        gens = run_hf(SHARER_BASE, prompts)
    else:  # receiver, text_rag
        gens = run_hf(C.RECEIVER, prompts)
    dt = time.time() - t0

    preds = [parse_letter(g, all_labels[i]) for i, g in enumerate(gens)]
    cors = [p == g for p, g in zip(preds, golds)]
    acc = sum(cors) / len(cors)
    tag = f"{args.method}_{args.fact}" + ("_fact_only" if args.method == "c2c_rag"
                                           and args.fuse == "fact_only" else "")
    out = {"tag": tag, "method": args.method, "fact": args.fact,
           "fuse": args.fuse if args.method == "c2c_rag" else None, "n_test": len(test),
           "accuracy": acc, "n_correct": int(sum(cors)), "retr_hit": retr_hit, "seconds": round(dt, 1)}
    os.makedirs(os.path.join(HERE, "obqa_results"), exist_ok=True)
    with open(os.path.join(HERE, "obqa_results", f"{tag}.json"), "w") as f:
        json.dump({**out, "samples": [{"prompt": prompts[k], "gen": gens[k], "pred": preds[k],
                   "gold": golds[k], "ok": cors[k]} for k in range(len(gens))]}, f, indent=2)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
