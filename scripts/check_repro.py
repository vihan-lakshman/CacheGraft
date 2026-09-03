"""Compare two runs of the same cells, to verify reproducibility.

Every number in the paper should survive a clean re-run of the documented commands. This
compares a reference results directory against the current one, cell by cell, at three levels:

  generations  fraction of questions whose generated text is byte-identical
  metrics      contains / EM / F1 computed from those generations
  judges       verdict agreement per judge (phi-4 and gemma-2 are greedy, so an identical
               generation must produce an identical verdict; any drift here is a bug, unlike
               with an API judge, which flips 8-9% of verdicts on identical input)

Greedy decoding is deterministic in principle, but CUDA kernels are not bitwise-stable across
processes when batch composition differs, so a small generation-level divergence is expected and
correctness-preserving; the metric columns are what must hold.

  python check_repro.py --data squad --ref squad_results.v1
"""
import os, json, re, glob, argparse
import numpy as np
import common as C


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()


def load(d, tag):
    f = os.path.join(d, f"{tag}.json")
    if not os.path.exists(f):
        return None
    S = json.load(open(f))["samples"]
    return {s["q"]: s for s in S}


def judges(d, tag):
    out = {}
    for jf in glob.glob(os.path.join(d, f"{tag}.judge*.json")):
        name = os.path.basename(jf)[len(tag) + 1:-5]
        name = "haiku" if name == "judge" else name.replace("judge_", "")
        out[name] = {k: bool(v) for k, v in json.load(open(jf)).items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", required=True, help="reference results dir (e.g. squad_results.v1)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="metric difference in points that counts as a match")
    args = ap.parse_args()
    new_dir = C.paths(args.data)["results"]
    ref_dir = args.ref if os.path.isdir(args.ref) else os.path.join(os.path.dirname(new_dir), args.ref)
    q2al = {q["q"]: [a for a in q["norm_aliases"] if a]
            for q in json.load(open(C.paths(args.data)["data"]))["questions"]}

    tags = sorted({os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ref_dir, "*.json"))
                   if ".judge" not in f and ".bertscore" not in f})
    print(f"{args.data}: comparing {new_dir}  vs  {ref_dir}")
    print(f"{'cell':<40}{'gen==':>8}{'contains':>20}{'EM':>16}{'judges':>26}")
    worst = 0.0
    for tag in tags:
        a, b = load(new_dir, tag), load(ref_dir, tag)
        if a is None:
            print(f"{tag:<40}{'MISSING in new run':>8}"); continue
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"{tag:<40}{'no overlap':>8}"); continue
        same_gen = sum(1 for k in keys if a[k]["gen"] == b[k]["gen"]) / len(keys)

        def met(rec, fn):
            return 100 * np.mean([fn(rec[k]) for k in keys])
        cont = lambda s: any(x in normalize(s["gen"]) for x in q2al.get(s["q"], []))
        em = lambda s: s["pred"] in q2al.get(s["q"], [])
        ca, cb = met(a, cont), met(b, cont)
        ea, eb = met(a, em), met(b, em)
        ja, jb = judges(new_dir, tag), judges(ref_dir, tag)
        jtxt = []
        for name in sorted(set(ja) & set(jb)):
            ks = [k for k in keys if k in ja[name] and k in jb[name]]
            if not ks: continue
            agree = 100 * np.mean([ja[name][k] == jb[name][k] for k in ks])
            d = 100 * (np.mean([ja[name][k] for k in ks]) - np.mean([jb[name][k] for k in ks]))
            jtxt.append(f"{name} {agree:.0f}%/{d:+.1f}")
            worst = max(worst, abs(d))
        worst = max(worst, abs(ca - cb), abs(ea - eb))
        print(f"{tag:<40}{100*same_gen:>7.1f}%{f'{ca:.1f} vs {cb:.1f} ({ca-cb:+.1f})':>20}"
              f"{f'{ea:.1f} ({ea-eb:+.1f})':>16}{'  '.join(jtxt):>26}")
    verdict = "PASS" if worst <= args.tol else "CHECK"
    print(f"\n{verdict}: largest metric/judge difference = {worst:.2f} points (tolerance {args.tol})")


if __name__ == "__main__":
    main()
