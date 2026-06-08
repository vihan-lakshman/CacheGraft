"""Re-score saved TriviaQA generations with verbosity-robust metrics.
  em        : exact normalized match (original, strict)
  em_prefix : prediction (normalized) starts with a gold alias  -> verbosity-corrected
  contains  : gold alias appears anywhere in the (short) prediction
Also reports avg prediction word-count to quantify verbosity.
"""
import os, json, glob, re
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

order = ["receiver_none", "sharer_none", "text_rag_retrieved", "c2c_retrieved_doc_only",
         "c2c_retrieved_doc_q", "text_rag_oracle", "c2c_oracle_doc_only", "c2c_oracle_doc_q"]
have = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(HERE,"trivia_results/*.json"))}
print(f"{'cell':<28}{'EM':>7}{'EM_pref':>9}{'contains':>10}{'avg_words':>10}")
for t in order:
    if t in have:
        em, emp, cont, w = rescore(have[t])
        print(f"{t:<28}{em*100:>6.1f}%{emp*100:>8.1f}%{cont*100:>9.1f}%{w:>10.1f}")
    else:
        print(f"{t:<28}{'(pending)':>7}")
