"""TriviaQA-RAG: open-domain short-answer QA with passage retrieval.

Realistic RAG: documents fused offline (doc-only), question is query-time text.
  receiver  none                  : Qwen3-0.6B, no passage (parametric baseline)
  sharer    none                  : Qwen3-4B-Base, no passage
  text_rag  retrieved|oracle      : Qwen3-0.6B, passage as TEXT
  c2c       retrieved|oracle  + --fuse doc_only : fuse ONLY doc tokens (cacheable, realistic)
  c2c       retrieved|oracle  + --fuse doc_q    : fuse [doc+question] (upper bound)
"""
import os, json, argparse, re, time
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import common as C

HERE = os.path.dirname(__file__)
SHARER_BASE = C.SHARER_BASE

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

def first_line(t):
    t = t.strip().split("\n")[0]
    return re.sub(r"^(answer|the answer is|a:)\s*:?\s*", "", t, flags=re.I).strip()

# Uniform answer extraction (applied to EVERY method before scoring). The receiver/fused models
# tend to restate the question as a copular sentence with the answer as the predicate nominal
# ("<long restatement> is <ANSWER>"); we keep the tail after the last copula when the
# restatement is long and the tail is short. This recovers F1/EM from high-`contains` verbose
# output without forcing the model terse.
COPULA = re.compile(r"^(.*\b(?:is|was|are|were|been|be|known as|called|named|refers to|"
                    r"stands for|meaning)\b)\s+(.+)$", re.I)

def extract_answer(t):
    t = first_line(t)
    m = COPULA.match(t)
    if m and len(m.group(1).split()) >= 3 and 0 < len(m.group(2).split()) <= 6:
        t = m.group(2).strip()
    return t

def score(gen, norm_aliases):
    p = normalize(extract_answer(gen))
    em = any(a == p for a in norm_aliases)
    contains = any(a and a in p for a in norm_aliases)
    pt = p.split()
    f1 = 0.0
    for a in norm_aliases:
        at = a.split()
        if not at or not pt: continue
        common = sum((min(pt.count(w), at.count(w)) for w in set(at)))
        if common:
            prec, rec = common/len(pt), common/len(at)
            f1 = max(f1, 2*prec*rec/(prec+rec))
    return em, contains, f1, p

# Generic few-shot exemplars (NO overlap with the TriviaQA test set) that demonstrate the
# desired terse, entity-only answer style. Applied identically to every method so the
# comparison stays fair; their only job is to stop the model restating the question as a full
# sentence (the verbosity that tanks F1/EM while `contains` already passes). For doc_only
# fusion they sit in the fixed, cacheable preamble (before the query), so cacheability holds.
FEWSHOT = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In which year did the Titanic sink?", "1912"),
    ("Which ocean is the largest on Earth?", "Pacific Ocean"),
    ("What is the largest planet in the solar system?", "Jupiter"),
]

def _shots_block(shots):
    if not shots:
        return ""
    return "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in FEWSHOT[:shots]) + "\n\n"

# Terse-format fix for the verbosity problem. The model's instinct is to restate the question
# as a full sentence with the answer buried inside ("The highest mountain in Africa is
# Kilimanjaro.") -- `contains` passes but the restating tokens tank F1/EM. A short-answer
# instruction + clause truncation collapses that to the bare entity. Applied identically to
# every method, so the comparison stays fair (it just helps the wordier C2C channel most).
INSTR_TERSE = (" Reply with the answer only -- a name, entity, or number in at most three "
               "words. Do not repeat the question and give no explanation.")

def make_content(q, passage, shots=0, terse=False):
    sb = _shots_block(shots)
    extra = INSTR_TERSE if terse else ""
    if passage is None:
        return (f"Answer the trivia question with a short factual answer (no explanation).{extra}\n"
                f"{sb}Question: {q}")
    return (f"Answer the question using the context. Give only a short factual answer "
            f"(no explanation).{extra}\n\nContext:\n{passage}\n\n{sb}Question: {q}")

def run_hf(model_name, contents, batch=48, max_new=24):
    tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    outs = []
    for i in range(0, len(contents), batch):
        chunk = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                  add_generation_prompt=True, enable_thinking=False) for c in contents[i:i+batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del model; torch.cuda.empty_cache(); return outs

def run_c2c(contents, fuse, max_new=24):
    from rosetta.utils.evaluate import load_rosetta_model
    mc = {"model_name": "Rosetta", "rosetta_config": {
        "base_model": C.RECEIVER, "teacher_model": SHARER_BASE,
        "is_do_alignment": False, "alignment_strategy": "first", "checkpoints_dir": C.get_fuser_dir()}}
    model, tok = load_rosetta_model(mc, {}, device=torch.device("cuda"),
                                    generation_config={"do_sample": False, "max_new_tokens": max_new})
    outs = []
    for c in contents:
        full = tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to("cuda") for k, v in enc.items()}
        P = enc["input_ids"].shape[1]
        # The wrapper applies fusion PER SECTION (keyed on each section's first token),
        # so doc-only fusion needs THREE sections: [doc=fuse, question=no-fuse, gen].
        def sec(mask, n):
            return torch.tensor([[mask, 0]] * n).unsqueeze(0).cuda()
        if fuse == "doc_q":
            idx = [sec(1, P - 1), sec(-1, 1)]
        else:  # doc_only
            c_q = full.rfind("Question:")
            L_doc = sum(1 for i in range(P) if 0 < offs[i][1] <= c_q)
            L_doc = max(1, min(L_doc, P - 2))   # leave room for question + gen
            L_q = (P - 1) - L_doc
            idx = [sec(1, L_doc), sec(-1, L_q), sec(-1, 1)]
        with torch.no_grad():
            o = model.generate(**enc, kv_cache_index=idx, do_sample=False, max_new_tokens=max_new)
        outs.append(tok.decode(o[0, P:], skip_special_tokens=True))
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["receiver", "text_rag", "c2c", "sharer"])
    ap.add_argument("--fact", default="none", choices=["none", "retrieved", "oracle"])
    ap.add_argument("--fuse", default="doc_only", choices=["doc_only", "doc_q"])
    ap.add_argument("--distill", default="none", choices=["none", "summary"],
                    help="text_rag only: feed the offline 4B summary of the passage (cost-matched "
                         "text-channel baseline for doc_only) instead of the raw passage")
    ap.add_argument("--distill_tag", default="",
                    help="read <data>_distilled<tag>.json (matches trivia_distill.py --tag); "
                         "also appended to the result tag so variants do not overwrite")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--data", default="trivia", help="dataset name (see common.paths): trivia, squad, ...")
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot Q/A exemplars to prepend (terse-answer format induction); "
                         "applied to every method for a fair comparison")
    ap.add_argument("--terse", action="store_true",
                    help="verbosity fix: short-answer instruction + clause truncation, so the "
                         "model emits the bare entity instead of restating the question. Applied "
                         "to every method for a fair comparison (helps the wordier C2C most)")
    args = ap.parse_args()
    C.set_determinism()
    if args.shots > len(FEWSHOT):
        ap.error(f"--shots max is {len(FEWSHOT)}")
    if args.distill != "none":
        if args.method != "text_rag":
            ap.error("--distill applies only to --method text_rag (it is a text-channel baseline)")
        if args.fact == "none":
            ap.error("--distill needs --fact retrieved or oracle (there is no passage to distill)")

    P = C.paths(args.data)
    data = json.load(open(P["data"]))
    corpus = data["corpus"]
    qs = data["questions"][:args.n_test]

    fact_ids, retr_hit = None, None
    if args.fact == "retrieved":
        from sentence_transformers import SentenceTransformer
        cemb = np.load(P["emb"])
        em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"]
                          for q in qs], normalize_embeddings=True, batch_size=128)
        fact_ids = C.top1(qemb, cemb, qs)
        retr_hit = float(np.mean([fact_ids[i] in set(qs[i]["gold_ids"]) for i in range(len(qs))]))
        print(f"retrieval top-1 gold-chunk hit rate: {retr_hit:.3f}", flush=True)
    elif args.fact == "oracle":
        fact_ids = [q["gold_ids"][0] for q in qs]

    # resolve the chosen chunk ids to passage text -- raw, or its offline 4B summary.
    if fact_ids is None:
        facts = None
    elif args.distill == "summary":
        dpath = os.path.join(HERE, f"{args.data}_distilled{args.distill_tag}.json")
        if not os.path.exists(dpath):
            raise SystemExit(f"run `trivia_distill.py --data {args.data}` first to build "
                             f"{os.path.basename(dpath)}")
        distilled = json.load(open(dpath))      # aligned 1:1 with corpus (null = not distilled)
        missing = [i for i in fact_ids if not distilled[i]]
        if missing:
            raise SystemExit(f"{len(missing)} needed passages were not distilled (e.g. id "
                             f"{missing[0]}); re-run trivia_distill.py with matching --n_test "
                             f"or --subset all")
        facts = [distilled[i] for i in fact_ids]
    else:
        facts = [corpus[i] for i in fact_ids]

    contents = [make_content(q["q"], facts[i] if facts else None, args.shots, args.terse)
                for i, q in enumerate(qs)]

    # NB: generations are saved RAW. Answer extraction (copula/clause) is a uniform scoring-time
    # post-process (see extract_answer, and report_table.py for scoring) applied to every
    # tuned offline and never advantages one channel over another.
    t0 = time.time()
    if args.method == "c2c":
        gens = run_c2c(contents, args.fuse)
    elif args.method == "sharer":
        gens = run_hf(SHARER_BASE, contents)
    else:
        gens = run_hf(C.RECEIVER, contents)
    dt = time.time() - t0

    res = [score(g, qs[i]["norm_aliases"]) for i, g in enumerate(gens)]
    em = np.mean([r[0] for r in res]); contains = np.mean([r[1] for r in res]); f1 = np.mean([r[2] for r in res])
    tag = (f"{args.method}_{args.fact}"
           + (f"_{args.fuse}" if args.method == "c2c" else "")
           + (f"_distill{args.distill_tag}" if args.distill == "summary" else "")
           + (f"_s{args.shots}" if args.shots else "")
           + ("_terse" if args.terse else ""))
    out = {"tag": tag, "method": args.method, "fact": args.fact,
           "fuse": args.fuse if args.method == "c2c" else None,
           "distill": args.distill, "shots": args.shots, "terse": args.terse, "n": len(qs),
           "contains": round(float(contains), 4), "em": round(float(em), 4),
           "f1": round(float(f1), 4), "retr_hit": retr_hit, "seconds": round(dt, 1)}
    os.makedirs(P["results"], exist_ok=True)
    json.dump({**out, "samples": [{"q": qs[k]["q"], "gen": gens[k], "pred": res[k][3],
               "aliases": qs[k]["aliases"][:6], "ok": bool(res[k][1])} for k in range(len(gens))]},
              open(os.path.join(P["results"], f"{tag}.json"), "w"), indent=2)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
