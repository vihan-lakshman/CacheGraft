"""Fast prompt/format sweep on the RECEIVER (text channel) to find a terse-but-accurate
answer format, so we can fix the C2C verbosity problem without paying the C2C decode cost on
every iteration. Loads the 0.6B once, retrieves top-1, and tries each variant batched.

  python tune_prompt.py --n 120
"""
import os, json, re, argparse
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import common as C

HERE = os.path.dirname(__file__)

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

def first_line(t):
    t = t.strip().split("\n")[0]
    return re.sub(r"^(answer|the answer is|a:)\s*:?\s*", "", t, flags=re.I).strip()

def cut_sentence(t):           # up to first . ; newline
    return re.split(r"[.;\n]", t.strip(), maxsplit=1)[0].strip()

def cut_clause(t):             # also cut at comma / dash (first short span)
    return re.split(r"[.,;\n–—]| - ", t.strip(), maxsplit=1)[0].strip()

def score(pred, norm_aliases):
    p = normalize(pred)
    em = any(a == p for a in norm_aliases)
    contains = any(a and a in p for a in norm_aliases)
    pt = p.split(); f1 = 0.0
    for a in norm_aliases:
        at = a.split()
        if not at or not pt: continue
        common = sum(min(pt.count(w), at.count(w)) for w in set(at))
        if common:
            prec, rec = common/len(pt), common/len(at)
            f1 = max(f1, 2*prec*rec/(prec+rec))
    return em, contains, f1, len(p.split())

# ----- format variants: (name, instruction, assistant_primer, postprocess) -----
INSTR_BASE = ("Answer the question using the context. Give only a short factual answer "
              "(no explanation).")
INSTR_STRONG = ("Answer the question using the context. Respond with ONLY the answer itself -- "
                "a single name, entity, or number. Do not write a sentence, do not repeat the "
                "question, give no explanation.")
INSTR_NWORDS = ("Answer the question using the context. Reply with the answer in at most three "
                "words. Do not repeat the question and give no explanation.")

SHOT = ("Context:\nMount Everest, in the Himalayas on the China-Nepal border, is Earth's "
        "highest mountain above sea level at 8,849 metres.\n"
        "Question: What is the highest mountain on Earth?\nAnswer: Mount Everest\n\n")

def body(instr, passage, q, shot=""):
    return f"{instr}\n\n{shot}Context:\n{passage}\n\nQuestion: {q}"

VARIANTS = [
    ("base",                 lambda p,q: body(INSTR_BASE, p, q),   "",        first_line),
    ("strong",               lambda p,q: body(INSTR_STRONG, p, q), "",        first_line),
    ("strong+sent",          lambda p,q: body(INSTR_STRONG, p, q), "",        lambda t: cut_sentence(first_line(t))),
    ("strong+clause",        lambda p,q: body(INSTR_STRONG, p, q), "",        lambda t: cut_clause(first_line(t))),
    ("strong+A:prim+clause", lambda p,q: body(INSTR_STRONG, p, q), "Answer:", lambda t: cut_clause(first_line(t))),
    ("nwords+clause",        lambda p,q: body(INSTR_NWORDS, p, q), "",        lambda t: cut_clause(first_line(t))),
    ("1shot+clause",         lambda p,q: body(INSTR_STRONG, p, q, SHOT), "",  lambda t: cut_clause(first_line(t))),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max_new", type=int, default=24); args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, "trivia_data.json")))
    corpus = data["corpus"]; qs = data["questions"][:args.n]
    from sentence_transformers import SentenceTransformer
    cemb = np.load(os.path.join(HERE, "trivia_corpus_emb.npy"))
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"]
                      for q in qs], normalize_embeddings=True, batch_size=128)
    fact_ids = (qemb @ cemb.T).argmax(1)
    facts = [corpus[int(i)] for i in fact_ids]
    del em; torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(C.RECEIVER); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C.RECEIVER, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    def gen_all(build, primer, batch=48):
        outs = []
        prompts = [build(facts[i], qs[i]["q"]) for i in range(len(qs))]
        for i in range(0, len(prompts), batch):
            chunk = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                      add_generation_prompt=True, enable_thinking=False) + primer
                     for c in prompts[i:i+batch]]
            enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new,
                                   pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    print(f"{'variant':<22}{'contains':>9}{'EM':>7}{'F1':>8}{'words':>7}")
    rawcache = {}
    for name, build, primer, post in VARIANTS:
        key = (id(build), primer)
        raw = rawcache.get(key)
        if raw is None:
            raw = gen_all(build, primer); rawcache[key] = raw
        R = [score(post(raw[i]), qs[i]["norm_aliases"]) for i in range(len(qs))]
        c = np.mean([r[1] for r in R]); e = np.mean([r[0] for r in R])
        f = np.mean([r[2] for r in R]); w = np.mean([r[3] for r in R])
        print(f"{name:<22}{c*100:>8.1f}%{e*100:>6.1f}%{f*100:>7.1f}%{w:>7.1f}")

if __name__ == "__main__":
    main()
