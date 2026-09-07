"""Compare a fresh run against the numbers PUBLISHED IN THE README.

`check_repro.py` compares a re-run against a previous local results directory, which tests
determinism on one machine. This is the check a READER needs: it compares the run in
`<data>_results/` against the numbers actually published in README.md (`readme_reference.json`),
cell by cell, on phi-4 / Gemma / contains / EM.

    python check_readme.py            # after steps 1-4; TOL=1.0 points by default
    TOL=0.5 python check_readme.py    # stricter

Cells the run has not produced are reported as MISSING rather than silently skipped, so a
partial reproduction cannot look like a passing one.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_TF", "0")
import numpy as np
import common as C
from report_table import load_cells

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "readme_reference.json")))
METRICS = ["phi-4", "gemma2", "contains", "em"]
TOL = float(os.environ.get("TOL", "1.0"))

def pct(m): return 100 * sum(m.values()) / len(m)

worst, nchecked, nfail, missing = 0.0, 0, 0, []
for data, cells_ref in REF.items():
    results = C.paths(data)["results"]
    if not os.path.isdir(results):
        missing.append(f"{data}: no results dir"); continue
    tags = sorted({os.path.basename(f)[:-5] for f in glob.glob(os.path.join(results, "*.json"))
                   if ".judge" not in f and ".bertscore" not in f})
    cells = load_cells(data, results, tags)
    print(f"\n===== {data} =====")
    print(f"{'cell':<38}" + "".join(f"{m:>22}" for m in METRICS))
    for tag, ref in cells_ref.items():
        if tag not in cells:
            missing.append(f"{data}/{tag}"); print(f"{tag:<38}{'MISSING':>22}"); continue
        c = cells[tag]
        row = f"{tag:<38}"
        for m, r in zip(METRICS, ref):
            if r is None:
                row += f"{'--':>22}"; continue
            got = c["judges"].get(m) if m in ("phi-4", "gemma2") else c.get(m)
            if not got:
                row += f"{'no judge':>22}"; continue
            v = pct(got); d = v - r; nchecked += 1
            worst = max(worst, abs(d))
            if abs(d) > TOL: nfail += 1
            flag = "" if abs(d) <= TOL else "  <-- OFF"
            row += f"{f'{v:.1f} vs {r:.1f} ({d:+.1f})':>22}{flag}"
        print(row)

print(f"\n{'='*70}")
print(f"checked {nchecked} numbers against the README; {nfail} outside +/-{TOL} points")
print(f"largest absolute difference: {worst:.2f} points")
if missing: print(f"missing cells ({len(missing)}): {missing[:12]}")
print("VERDICT:", "REPRODUCES" if nfail == 0 and not missing else "DISCREPANCIES - see above")
