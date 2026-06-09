"""Re-score saved TriviaQA generations with verbosity-robust metrics.
  em        : exact normalized match (original, strict)
  em_prefix : prediction (normalized) starts with a gold alias  -> verbosity-corrected
  contains  : gold alias appears anywhere in the (short) prediction
Also reports avg prediction word-count to quantify verbosity.
"""
import os, json, glob, re, math
HERE = os.path.dirname(__file__)

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

DATA = json.load(open(os.path.join(HERE, "trivia_data.json")))
Q2AL = {q["q"]: q["norm_aliases"] for q in DATA["questions"]}  # full alias sets

def rescore(d):
    em = emp = cont = n = 0; words = 0
    for s in d["samples"]:
        p = normalize(s["pred"])
        aliases = sorted([a for a in Q2AL.get(s["q"], []) if a], key=len, reverse=True)
        n += 1; words += len(p.split())
        em += any(p == a for a in aliases)
        emp += any(a and (p == a or p.startswith(a + " ")) for a in aliases)
        cont += any(a and a in p for a in aliases)
    return em/n, emp/n, cont/n, words/n

def cell_ok(d):
    """Per-question `contains` outcome, keyed by question -> for paired tests."""
    ok = {}
    for s in d["samples"]:
        aliases = [a for a in Q2AL.get(s["q"], []) if a]
        p = normalize(s["pred"])
        ok[s["q"]] = any(a in p for a in aliases)
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
         "text_rag_retrieved", "text_rag_retrieved_distill", "c2c_retrieved_doc_only",
         "c2c_retrieved_doc_q",
         "text_rag_oracle", "text_rag_oracle_distill", "c2c_oracle_doc_only", "c2c_oracle_doc_q"]
have = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(HERE,"trivia_results/*.json"))}
print(f"{'cell':<28}{'EM':>7}{'EM_pref':>9}{'contains':>10}{'avg_words':>10}")
for t in order:
    if t in have:
        em, emp, cont, w = rescore(have[t])
        print(f"{t:<28}{em*100:>6.1f}%{emp*100:>8.1f}%{cont*100:>9.1f}%{w:>10.1f}")
    else:
        print(f"{t:<28}{'(pending)':>7}")

# Paired significance on `contains`: the distilled-text baseline is the one C2C must beat
# at equal query-time cost. doc_only vs raw text isolates "fusion vs retrieval"; doc_only vs
# distilled isolates "KV channel vs text channel" (the information-bottleneck claim).
PAIRS = [("c2c_retrieved_doc_only", "text_rag_retrieved"),
         ("c2c_retrieved_doc_only", "text_rag_retrieved_distill"),
         ("c2c_oracle_doc_only",    "text_rag_oracle_distill")]
shown = [(a, b) for a, b in PAIRS if a in have and b in have]
if shown:
    print("\nMcNemar on `contains` (paired):  first-only / second-only correct, two-sided p")
    for a, b in shown:
        oa, ob, p = mcnemar(cell_ok(have[a]), cell_ok(have[b]))
        sig = "*" if p < 0.05 else " "
        print(f"  {a:<26} vs {b:<26}  {oa:>3} / {ob:<3}  p={p:.4f} {sig}")
