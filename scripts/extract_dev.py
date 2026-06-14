"""Offline tuning of ANSWER EXTRACTION over already-saved RAW generations (no GPU).

The verbosity fix the data points to: don't force the model terse (that loses `contains`),
keep the high-`contains` verbose output and EXTRACT the entity at scoring time. C2C restates as
"[question] is/was [ANSWER]" with the answer as the predicate nominal after the last copula --
so clause truncation + copula-tail extraction recovers it. Applied uniformly to every method.

  python extract_dev.py c2c_retrieved_doc_only text_rag_retrieved
"""
import os, json, re, sys
HERE = os.path.dirname(__file__)

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

DATA = json.load(open(os.path.join(HERE, "trivia_data.json")))
Q2AL = {q["q"]: q["norm_aliases"] for q in DATA["questions"]}

# ---------------- extractor candidates (operate on the RAW generation) ----------------
def first_line(t):
    t = t.strip().split("\n")[0]
    return re.sub(r"^(answer|the answer is|a:)\s*:?\s*", "", t, flags=re.I).strip()

def cut_clause(t):
    return re.split(r"[.,;\n–—]| - ", t.strip(), maxsplit=1)[0].strip()

COPULA = re.compile(r"^(.*\b(?:is|was|are|were|been|be|known as|called|named|refers to|"
                    r"stands for|meaning)\b)\s+(.+)$", re.I)
def strip_copula(t):
    """If t is a '<restatement> is <ANSWER>' clause, keep the predicate nominal after the
    LAST copula -- but only when the restatement prefix is long (>=3 words), so we don't
    chop a short answer that merely starts with/contains a copula."""
    m = COPULA.match(t)
    if not m:
        return t
    prefix, tail = m.group(1), m.group(2)
    if len(prefix.split()) >= 3 and 0 < len(tail.split()) <= 6:
        return tail
    return t

EXTRACTORS = {
    "first_line (orig)":        lambda g: first_line(g),
    "clause":                   lambda g: cut_clause(first_line(g)),
    "copula":                   lambda g: strip_copula(first_line(g)),
    "clause+copula":            lambda g: cut_clause(strip_copula(first_line(g))),
    "copula+clause":            lambda g: strip_copula(cut_clause(first_line(g))),
}

def metrics(samples, ex):
    em = cont = f1 = w = n = 0
    for s in samples:
        pred = normalize(ex(s["gen"]))
        al = sorted([a for a in Q2AL.get(s["q"], []) if a], key=len, reverse=True)
        pt = pred.split(); n += 1; w += len(pt)
        em += any(pred == a for a in al)
        cont += any(a in pred for a in al)
        best = 0.0
        for a in al:
            at = a.split()
            if not at or not pt: continue
            c = sum(min(pt.count(x), at.count(x)) for x in set(at))
            if c:
                p, r = c/len(pt), c/len(at); best = max(best, 2*p*r/(p+r))
        f1 += best
    return cont/n*100, em/n*100, f1/n*100, w/n

for tag in sys.argv[1:]:
    d = json.load(open(os.path.join(HERE, f"trivia_results/{tag}.json")))
    print(f"\n=== {tag}  (n={len(d['samples'])}) ===")
    print(f"{'extractor':<22}{'contains':>9}{'EM':>7}{'F1':>8}{'words':>7}")
    for name, ex in EXTRACTORS.items():
        c, e, f, w = metrics(d["samples"], ex)
        print(f"{name:<22}{c:>8.1f}%{e:>6.1f}%{f:>7.1f}%{w:>7.1f}")
