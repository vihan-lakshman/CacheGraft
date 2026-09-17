"""BERTScore scoring -- the second, fully deterministic instrument.

Why this and not a second LLM judge: re-judging byte-identical generations with the Claude API
flipped 8-9% of verdicts and moved a paired delta by 2.5 points, which is enough to change a
significance verdict. A local judge (judge_open.py) and BERTScore are both reproducible from
the committed code, so the headline never rests on a grader we cannot re-run.

Protocol follows the CAG paper (arXiv:2412.15605), which reports answer/ground-truth similarity
with BERTScore [Zhang et al., ICLR 2020]: library defaults, un-rescaled F1, default English
model. With multiple gold aliases we take the MAX over aliases, which is the standard
multi-reference treatment and matches how EM/contains handle alias sets here.

Scored on the RAW generation, exactly like the two judges -- not on the extracted answer. This
makes BERTScore a deliberately CONSERVATIVE instrument for this paper: its precision term
penalises length, and fused generations run ~1.5-2x longer than text-RAG ones, so the metric is
biased AGAINST the system we are testing. An effect that survives it is not a verbosity artifact.

  python score_bertscore.py --data squad text_rag_retrieved c2c_retrieved_doc_only

Writes `<tag>.bertscore.json` ({question: f1}), which report_table.py picks up automatically.
"""
import os, json, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--data", default="squad")
    ap.add_argument("--model", default=None,
                    help="override the BERTScore model (default: the library's English default)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--variant", default="f1", choices=["f1", "recall", "precision"],
                    help="F1 (the CAG protocol) is dominated by its precision term when the gold "
                         "answer is a 1-5 token span, so it scores brevity rather than "
                         "correctness; recall is the length-tolerant variant")
    ap.add_argument("--on", default="gen", choices=["gen", "pred"],
                    help="score the raw generation (like the judges) or the uniformly extracted "
                         "answer, which length-matches the channels before scoring")
    ap.add_argument("--tag_suffix", default="")
    args = ap.parse_args()

    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", model_type=args.model, batch_size=args.batch,
                        rescale_with_baseline=False)
    print(f"BERTScore model: {scorer._model_type}  (un-rescaled F1, max over gold aliases)")

    P = C.paths(args.data)
    data_qs = json.load(open(P["data"]))["questions"]
    RESULTS = P["results"]

    print(f"{'cell':<40}{'n':>6}{'BERTScore F1':>14}{'words':>8}")
    for tag in args.tags:
        f = os.path.join(RESULTS, f"{tag}.json")
        if not os.path.exists(f):
            print(f"  (missing: {tag})"); continue
        S = json.load(open(f))["samples"]
        # one (candidate, reference) pair per gold alias; max over aliases per question
        cands, refs, owner = [], [], []
        for i, (s, al) in enumerate(zip(S, C.gold_aliases(data_qs, S, field="aliases"))):
            for a in al or [""]:
                cands.append(s["gen"] if args.on == "gen" else s["pred"])
                refs.append(a); owner.append(i)
        Pr, Rc, F = scorer.score(cands, refs, verbose=False)
        f1 = {"f1": F, "recall": Rc, "precision": Pr}[args.variant].numpy()
        best = np.full(len(S), -1.0)
        for j, i in enumerate(owner):
            best[i] = max(best[i], float(f1[j]))
        C.write_scores(os.path.join(RESULTS, f"{tag}.bertscore{args.tag_suffix}.json"), S,
                       [float(b) for b in best])
        words = np.mean([len(s["gen"].split()) for s in S])
        print(f"{tag:<40}{len(S):>6}{best.mean():>14.4f}{words:>8.1f}")


if __name__ == "__main__":
    main()
