"""Re-score saved TriviaQA generations with verbosity-robust metrics.
  em        : exact normalized match (original, strict)
  em_prefix : prediction (normalized) starts with a gold alias  -> verbosity-corrected
  contains  : gold alias appears anywhere in the (short) prediction
Also reports avg prediction word-count to quantify verbosity.
"""
import os, json, glob, re, math, argparse
import common as C
HERE = os.path.dirname(__file__)
_ap = argparse.ArgumentParser(); _ap.add_argument("--data", default="trivia")
ARGS = _ap.parse_args()
PATHS = C.paths(ARGS.data)

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

# Same uniform answer extraction as rag_trivia.score (kept in sync); re-extracts from the RAW
# saved generation so metrics reflect the copula post-process for every method equally.
COPULA = re.compile(r"^(.*\b(?:is|was|are|were|been|be|known as|called|named|refers to|"
                    r"stands for|meaning)\b)\s+(.+)$", re.I)

def extract_answer(t):
    t = t.strip().split("\n")[0]
    t = re.sub(r"^(answer|the answer is|a:)\s*:?\s*", "", t, flags=re.I).strip()
    m = COPULA.match(t)
    if m and len(m.group(1).split()) >= 3 and 0 < len(m.group(2).split()) <= 6:
        t = m.group(2).strip()
    return t

DATA = json.load(open(PATHS["data"]))
Q2AL = {q["q"]: q["norm_aliases"] for q in DATA["questions"]}  # full alias sets

def f1_of(p, aliases):
    pt = p.split(); best = 0.0
    for a in aliases:
        at = a.split()
        if not at or not pt: continue
        c = sum(min(pt.count(w), at.count(w)) for w in set(at))
        if c:
            prec, rec = c/len(pt), c/len(at); best = max(best, 2*prec*rec/(prec+rec))
    return best

def rescore(d):
    em = emp = cont = f1 = n = 0; words = 0
    for s in d["samples"]:
        p = normalize(extract_answer(s["gen"]))
        aliases = sorted([a for a in Q2AL.get(s["q"], []) if a], key=len, reverse=True)
        n += 1; words += len(p.split())
        em += any(p == a for a in aliases)
        emp += any(a and (p == a or p.startswith(a + " ")) for a in aliases)
        cont += any(a and a in p for a in aliases)
        f1 += f1_of(p, aliases)
    return em/n, emp/n, f1/n, cont/n, words/n

def cell_ok(d, metric="contains"):
    """Per-question correctness keyed by question -> for paired McNemar tests."""
    ok = {}
    for s in d["samples"]:
        aliases = [a for a in Q2AL.get(s["q"], []) if a]
        p = normalize(extract_answer(s["gen"]))
        ok[s["q"]] = (any(a == p for a in aliases) if metric == "em"
                      else any(a in p for a in aliases))
    return ok

def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar on the questions both cells answered (paired)."""
    keys = set(a_ok) & set(b_ok)
    only_a = sum(1 for k in keys if a_ok[k] and not b_ok[k])   # first-cell-only correct
    only_b = sum(1 for k in keys if b_ok[k] and not a_ok[k])   # second-cell-only correct
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    lo = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n))
    return only_a, only_b, p

order = ["receiver_none", "sharer_none",
         "text_rag_retrieved", "text_rag_retrieved_terse",
         "text_rag_retrieved_distill", "text_rag_retrieved_distill_terse",
         "c2c_retrieved_doc_only", "c2c_retrieved_doc_only_terse",
         "c2c_retrieved_doc_q", "c2c_retrieved_doc_q_terse",
         "text_rag_oracle", "text_rag_oracle_terse",
         "text_rag_oracle_distill", "text_rag_oracle_distill_terse",
         "c2c_oracle_doc_only", "c2c_oracle_doc_only_terse",
         "c2c_oracle_doc_q", "c2c_oracle_doc_q_terse"]
have = {json.load(open(f))["tag"]: json.load(open(f))
        for f in glob.glob(os.path.join(PATHS["results"], "*.json")) if not f.endswith(".judge.json")}
extra = [t for t in have if t not in order]
print(f"{'cell':<34}{'EM':>7}{'F1':>8}{'EM_pref':>9}{'contains':>10}{'words':>7}")
for t in order + sorted(extra):
    if t in have:
        em, emp, f1, cont, w = rescore(have[t])
        print(f"{t:<34}{em*100:>6.1f}%{f1*100:>7.1f}%{emp*100:>8.1f}%{cont*100:>9.1f}%{w:>7.1f}")
    elif t in order:
        print(f"{t:<34}{'(pending)':>7}")

# Paired significance. doc_only vs raw text isolates "fusion vs retrieval"; doc_only vs
# distilled isolates "KV channel vs text channel" (the information-bottleneck claim). We test
# EM (the verbosity-fix target) and contains (the robust knowledge win) on matched protocols.
PAIRS = [("c2c_retrieved_doc_only_terse", "text_rag_retrieved_terse"),
         ("c2c_retrieved_doc_only_terse", "text_rag_retrieved_distill_terse"),
         ("c2c_oracle_doc_only_terse",    "text_rag_oracle_terse"),
         ("c2c_oracle_doc_only_terse",    "text_rag_oracle_distill_terse"),
         ("c2c_retrieved_doc_only", "text_rag_retrieved")]
for metric in ("em", "contains"):
    shown = [(a, b) for a, b in PAIRS if a in have and b in have]
    if shown:
        print(f"\nMcNemar on `{metric}` (paired):  first-only / second-only correct, two-sided p")
        for a, b in shown:
            oa, ob, p = mcnemar(cell_ok(have[a], metric), cell_ok(have[b], metric))
            sig = "*" if p < 0.05 else " "
            print(f"  {a:<32} vs {b:<32}  {oa:>3} / {ob:<3}  p={p:.4f} {sig}")
