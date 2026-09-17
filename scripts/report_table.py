"""Consolidated results table across cells, metrics and judge families.

Reads every `<data>_results/<tag>.json` plus any judge verdicts written beside it
(`<tag>.judge.json` from judge_trivia.py, `<tag>.judge_<short>.json` from judge_open.py) and
prints (a) one row per cell with every metric and (b) paired significance against a chosen
baseline, for each metric family.

Metric definitions, applied identically to every cell:
  phi-4     primary judge: microsoft/phi-4, run locally and greedily (deterministic)
  gemma2    second judge, a different model family (google/gemma-2-9b-it), same prompt
  contains  a gold alias appears in the raw generation (verbosity-tolerant, no model in the loop)
  EM / F1   against the uniform answer extraction saved as `pred` by the runner

Two metrics were tested and rejected, and the scripts are kept so the checks are reproducible:
an API judge (judge_trivia.py) flips 8-9%% of verdicts when re-run on identical generations,
enough to move a paired delta by 2.5 points; and BERTScore (score_bertscore.py), the CAG
protocol, ranks the 4B sharer BELOW the 0.6B receiver in every variant and correlates more
strongly with output length than with correctness, so it scores brevity on short-answer QA.
All of them use the dataset's FULL alias list from `<data>_data.json` -- the per-sample
`aliases` field is truncated for judge-prompt size and must not be used for scoring.

Unit of analysis: the question. Eval sets built by `prep.py` keep one question per source
passage, so questions are independent and question-level McNemar and bootstrap are the honest
tests. On a clustered set they would be anti-conservative.

  python report_table.py --data squad --baseline text_rag_retrieved \
      --cells text_rag_retrieved c2c_retrieved_doc_only recvkv_retrieved_doc_only
"""
import os, json, re, glob, math, argparse
import numpy as np
import common as C


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()


def f1(pred, aliases):
    pt = pred.split()
    best = 0.0
    for a in aliases:
        at = a.split()
        if not at or not pt:
            continue
        common = sum(min(pt.count(w), at.count(w)) for w in set(at))
        if common:
            prec, rec = common / len(pt), common / len(at)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def mcnemar(a, b, keys):
    """Exact two-sided paired McNemar."""
    na = sum(1 for k in keys if a[k] and not b[k])
    nb = sum(1 for k in keys if b[k] and not a[k])
    n = na + nb
    if n == 0:
        return na, nb, 1.0
    lo = min(na, nb)
    return na, nb, min(1.0, 2 * sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n))


def boot_ci(a, b, keys, n_boot=10000, seed=0):
    """Percentile bootstrap CI on the paired difference, resampling questions."""
    d = np.array([float(a[k]) - float(b[k]) for k in keys])
    rng = np.random.default_rng(seed)
    diffs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)


def kappa(a, b, keys):
    """Cohen's kappa between two binary raters."""
    n = len(keys)
    po = sum(1 for k in keys if a[k] == b[k]) / n
    pa, pb = sum(a[k] for k in keys) / n, sum(b[k] for k in keys) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def load_cells(data, results, tags):
    data_qs = json.load(open(C.paths(data)["data"]))["questions"]
    cells = {}
    for tag in tags:
        f = os.path.join(results, f"{tag}.json")
        if not os.path.exists(f):
            print(f"  (missing: {tag})")
            continue
        S = json.load(open(f))["samples"]
        rec = {"n": len(S), "words": float(np.mean([len(s["gen"].split()) for s in S])),
               "judges": {}}
        gold = C.gold_aliases(data_qs, S)
        keys = [C.item_key(i, s["q"]) for i, s in enumerate(S)]
        rec["contains"] = {k: any(a in normalize(s["gen"]) for a in al)
                           for k, s, al in zip(keys, S, gold)}
        rec["em"] = {k: s["pred"] in al for k, s, al in zip(keys, S, gold)}
        rec["f1"] = float(np.mean([f1(s["pred"], al) for s, al in zip(S, gold)]))
        bs = os.path.join(results, f"{tag}.bertscore.json")
        if os.path.exists(bs):
            rec["bertscore"] = {k: float(v) for k, v in C.load_scores(bs, S).items()}
        for jf in glob.glob(os.path.join(results, f"{tag}.judge*.json")):
            name = os.path.basename(jf)[len(tag) + 1:-5]        # judge | judge_<short>
            name = "haiku" if name == "judge" else name.replace("judge_", "")
            rec["judges"][name] = {k: bool(v) for k, v in C.load_scores(jf, S).items()}
        cells[tag] = rec
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="squad")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--baseline", default=None, help="cell to test every other cell against")
    ap.add_argument("--judges", nargs="+", default=["phi-4", "gemma2"],
                    help="primary first. Both defaults are local open-weights models run greedily, "
                         "so every reported number is reproducible from this repo with no API in "
                         "the loop. Pass 'haiku' explicitly to include the API judge, whose "
                         "verdicts flip 8-9%% when re-run on identical generations.")
    args = ap.parse_args()
    results = C.paths(args.data)["results"]
    cells = load_cells(args.data, results, args.cells)
    judges = [j for j in args.judges if any(j in c["judges"] for c in cells.values())]

    def pct(m):
        return 100 * sum(m.values()) / len(m)

    has_bs = any("bertscore" in c for c in cells.values())
    head = f"{'cell':<40}{'n':>5}" + "".join(f"{j:>10}" for j in judges) + \
           (f"{'BERTSc':>9}" if has_bs else "") + f"{'contains':>10}{'EM':>8}{'F1':>8}{'words':>7}"
    print(head); print("-" * len(head))
    for tag in args.cells:
        if tag not in cells: continue
        c = cells[tag]
        row = f"{tag:<40}{c['n']:>5}"
        for j in judges:
            v = c["judges"].get(j)
            row += f"{pct(v):>9.1f}%" if v else f"{'--':>10}"
        if has_bs:
            bsv = c.get("bertscore")
            row += f"{np.mean(list(bsv.values())):>9.4f}" if bsv else f"{'--':>9}"
        row += f"{pct(c['contains']):>9.1f}%{pct(c['em']):>8.1f}{100*c['f1']:>8.1f}{c['words']:>7.1f}"
        print(row)

    if args.baseline and args.baseline in cells:
        base = cells[args.baseline]
        print(f"\nPaired vs {args.baseline} "
              f"(exact two-sided McNemar; bootstrap CI resamples questions)")
        head = f"{'cell':<40}{'metric':<10}{'delta':>8}{'win/lose':>12}{'p':>11}{'95% CI':>18}"
        print(head); print("-" * len(head))
        for tag in args.cells:
            if tag == args.baseline or tag not in cells: continue
            for metric in judges + ["contains", "em", "bertscore"]:
                if metric in judges:
                    a, b = cells[tag]["judges"].get(metric), base["judges"].get(metric)
                else:
                    a, b = cells[tag].get(metric), base.get(metric)
                if not a or not b: continue
                keys = sorted(set(a) & set(b))
                if metric == "bertscore":
                    # continuous: paired bootstrap on the mean difference + Wilcoxon signed-rank
                    d = np.array([a[k] - b[k] for k in keys])
                    rng = np.random.default_rng(0)
                    di = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    try:
                        from scipy.stats import wilcoxon
                        p = float(wilcoxon(d, zero_method="wilcox").pvalue)
                    except Exception:
                        p = float(2 * min((di <= 0).mean(), (di >= 0).mean()))
                    win = int((d > 0).sum()); lose = int((d < 0).sum())
                    ci = f"[{100*np.percentile(di,2.5):+.2f}, {100*np.percentile(di,97.5):+.2f}]"
                    wl = f"{win}/{lose}"
                    print(f"{tag:<40}{metric:<10}{100*d.mean():>+8.2f}{wl:>12}{p:>11.2}{ci:>18}")
                    continue
                na, nb, p = mcnemar(a, b, keys)
                d, lo, hi = boot_ci(a, b, keys)
                print(f"{tag:<40}{metric:<10}{d:>+8.1f}{f'{na}/{nb}':>12}{p:>11.2}"
                      f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}")

    if len(judges) > 1:
        print(f"\nInter-judge agreement ({judges[0]} vs {judges[1]})")
        print(f"{'cell':<40}{'agree':>8}{'kappa':>8}")
        for tag in args.cells:
            if tag not in cells: continue
            a, b = cells[tag]["judges"].get(judges[0]), cells[tag]["judges"].get(judges[1])
            if not a or not b: continue
            keys = sorted(set(a) & set(b))
            k, po = kappa(a, b, keys)
            print(f"{tag:<40}{100*po:>7.1f}%{k:>8.3f}")


if __name__ == "__main__":
    main()
