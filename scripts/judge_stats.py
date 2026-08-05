"""Paired significance on LLM-judge verdicts.

`judge_trivia.py` writes `<tag>.judge.json` ({question: correct_bool}). This computes accuracy
per cell and an exact two-sided paired McNemar test between chosen pairs, on the questions both
cells answered. Companion to `rescore_trivia.py`, which does the same for lexical metrics.

  python judge_stats.py --data squad                       # accuracy table + default pairs
  python judge_stats.py --data squad --vs c2c_oracle_doc_only text_rag_oracle

Note on the unit of analysis: McNemar treats questions as independent. When a corpus has heavy
passage reuse (SQuAD's 500 questions read only 54 distinct paragraphs), errors cluster by
passage, so p-values are anti-conservative. `--cluster` reports a passage-level bootstrap
alongside, which is the honest interval for that design.
"""
import os, json, glob, math, argparse, itertools
import numpy as np
import common as C


def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar over the questions present in both cells."""
    keys = sorted(set(a_ok) & set(b_ok))
    only_a = sum(1 for k in keys if a_ok[k] and not b_ok[k])
    only_b = sum(1 for k in keys if b_ok[k] and not a_ok[k])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0, len(keys)
    lo = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n))
    return only_a, only_b, p, len(keys)


def cluster_bootstrap(a_ok, b_ok, q2doc, n_boot=10000, seed=0):
    """Bootstrap the accuracy difference by RESAMPLING PASSAGES, not questions. With heavy
    passage reuse the questions sharing a passage are not independent; resampling at the
    clustering level gives an interval that respects that."""
    keys = sorted(set(a_ok) & set(b_ok) & set(q2doc))
    if not keys:
        return None
    by_doc = {}
    for k in keys:
        by_doc.setdefault(q2doc[k], []).append(k)
    docs = sorted(by_doc)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        ks = [k for i in pick for k in by_doc[docs[i]]]
        diffs.append(np.mean([a_ok[k] for k in ks]) - np.mean([b_ok[k] for k in ks]))
    diffs = np.array(diffs)
    obs = np.mean([a_ok[k] for k in keys]) - np.mean([b_ok[k] for k in keys])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs * 100, np.percentile(diffs, 2.5) * 100, np.percentile(diffs, 97.5) * 100, \
        min(1.0, p), len(docs)


def question_to_doc(data_name, n_test=500):
    """Map each question string -> the corpus id it retrieves at top-1 (the clustering unit)."""
    P = C.paths(data_name)
    data = json.load(open(P["data"]))
    qs = data["questions"][:n_test]
    from sentence_transformers import SentenceTransformer
    cemb = np.load(P["emb"])
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"]
                      for q in qs], normalize_embeddings=True, batch_size=128)
    top = (qemb @ cemb.T).argmax(1)
    return {q["q"]: int(top[i]) for i, q in enumerate(qs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="trivia")
    ap.add_argument("--vs", nargs=2, action="append", metavar=("A", "B"),
                    help="compare cell A vs cell B (repeatable); default = every c2c* vs every "
                         "text_rag* within the same fact setting")
    ap.add_argument("--cluster", action="store_true",
                    help="also report a passage-level cluster bootstrap (needs GPU for retrieval)")
    args = ap.parse_args()
    RES = C.paths(args.data)["results"]

    cells = {}
    for f in sorted(glob.glob(os.path.join(RES, "*.judge.json"))):
        tag = os.path.basename(f)[:-len(".judge.json")]
        cells[tag] = json.load(open(f))
    if not cells:
        raise SystemExit(f"no *.judge.json in {RES} -- run judge_trivia.py first")

    print(f"{'cell':<42}{'n':>6}{'judge':>9}")
    for tag in sorted(cells):
        v = cells[tag]
        print(f"{tag:<42}{len(v):>6}{100*sum(v.values())/len(v):>8.1f}%")

    pairs = args.vs
    if not pairs:
        pairs = []
        for setting in ("retrieved", "oracle"):
            c2c = [t for t in cells if t.startswith("c2c") and f"_{setting}_" in t]
            txt = [t for t in cells if t.startswith("text_rag") and f"_{setting}" in t]
            pairs += [[a, b] for a, b in itertools.product(sorted(c2c), sorted(txt))]

    q2doc = question_to_doc(args.data) if args.cluster else None

    print(f"\nPaired exact McNemar on judge verdicts (A-only / B-only correct):")
    hdr = f"{'A':<34}{'B':<38}{'dA':>7}{'win':>5}{'lose':>6}{'p':>10}"
    print(hdr + (f"{'cluster 95% CI':>26}{'p_clu':>8}" if args.cluster else ""))
    for a, b in pairs:
        if a not in cells or b not in cells:
            print(f"  (skip {a} vs {b}: missing)"); continue
        A, B = cells[a], cells[b]
        keys = sorted(set(A) & set(B))
        d = 100 * (sum(A[k] for k in keys) - sum(B[k] for k in keys)) / len(keys)
        oa, ob, p, n = mcnemar(A, B)
        line = f"{a:<34}{b:<38}{d:>+7.1f}{oa:>5}{ob:>6}{p:>10.4f}"
        if args.cluster:
            r = cluster_bootstrap(A, B, q2doc)
            line += f"   [{r[1]:+.1f}, {r[2]:+.1f}] ({r[4]} docs){r[3]:>8.3f}" if r else " " * 26
        print(line + ("  *" if p < 0.05 else ""))


if __name__ == "__main__":
    main()
