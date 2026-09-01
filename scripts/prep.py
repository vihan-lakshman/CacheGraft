"""Build a passage-retrieval eval set from a reading-comprehension dataset.

Emits the harness's shared schema, so every runner (`rag_trivia.py`, `rag_trivia_cached.py`,
`trivia_distill.py`, the judges, `report_table.py`) works via `--data <name>`:

    <name>_data.json       {questions: [{q, aliases, norm_aliases, gold_ids}], corpus: [str]}
    <name>_corpus_emb.npy  bge-small embeddings of the corpus, for top-k retrieval

Two design choices matter for the statistics and are enforced here:

* **One question per source passage** (`--max_per_para 1`, the default). SQuAD-style corpora ask
  many questions about the same paragraph; scoring all of them treats correlated items as
  independent, which makes question-level p-values anti-conservative and lets a handful of
  passages dominate an estimate. With one question per passage, the question IS the independent
  unit and no cluster correction is needed. `--pick random` additionally avoids the annotation
  order bias (SQuAD annotators tend to write the most direct question first).
* **Chunking of long contexts** (`--chunk_words`). CovidQA and QASPER ship whole papers as the
  context; left whole they would be an order of magnitude longer than a SQuAD paragraph, which
  changes both the encoding burden and the KV footprint. Contexts longer than the budget are
  split into ~`chunk_words`-word passages and the chunk containing the answer becomes gold; the
  other chunks stay in the corpus as distractors. Short contexts pass through untouched.

    python prep.py --data squad          --target 800
    python prep.py --data covidqa        --target 800
    python prep.py --data adversarialqa  --target 800
    python prep.py --data qasper         --target 800
"""
import os, json, re, random, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
from datasets import load_dataset
import common as C


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------------------------
# Loaders. Each yields (question, answers: list[str], context: str). Everything downstream --
# de-clustering, chunking, gold marking, embedding -- is shared.
# ---------------------------------------------------------------------------------------------

def _squad_like(repo, split, config=None):
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    for ex in ds:
        yield ex["question"], list(ex["answers"]["text"]), ex["context"]


def load_squad():
    """SQuAD v1.1 dev: single-hop extractive QA over Wikipedia paragraphs."""
    return _squad_like("rajpurkar/squad", "validation")


def load_adversarialqa():
    """AdversarialQA: SQuAD-format, but questions were written adversarially against a reader,
    so the encoding burden per passage is higher by construction."""
    return _squad_like("UCLNLP/adversarial_qa", "validation", config="adversarialQA")


def load_covidqa():
    """CovidQA (deepset): SQuAD-format extractive QA over CORD-19 papers. Contexts are full
    papers, so chunking matters here; answers are outside a small model's parametric memory."""
    return _squad_like("deepset/covid_qa_deepset", "train")


def load_qasper():
    """QASPER: QA over NLP papers. We keep only questions with extractive answer spans, and use
    the paper's full text as the context so retrieval has to find the right passage."""
    # allenai/qasper is a script-based dataset, which `datasets` no longer executes; read the
    # Hub's parquet conversion of the same split instead.
    ds = load_dataset("allenai/qasper", split="validation", revision="refs/convert/parquet")
    for paper in ds:
        paras = [p for sec in paper["full_text"]["paragraphs"] for p in sec if p.strip()]
        if not paras:
            continue
        context = "\n".join(paras)
        qas = paper["qas"]
        for q, answers in zip(qas["question"], qas["answers"]):
            spans = []
            for a in answers["answer"]:
                if not a.get("unanswerable"):
                    spans += [s for s in (a.get("extractive_spans") or []) if s.strip()]
            if spans:
                yield q, sorted(set(spans)), context


LOADERS = {"squad": load_squad, "adversarialqa": load_adversarialqa,
           "covidqa": load_covidqa, "qasper": load_qasper}


def chunk(text, size):
    """Split into ~`size`-word passages. size=0, or a short text, passes through unchanged."""
    words = text.split()
    if not size or len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=sorted(LOADERS))
    ap.add_argument("--target", type=int, default=800, help="questions to keep")
    ap.add_argument("--chunk_words", type=int, default=0,
                    help="split contexts longer than this into ~this many words per passage "
                         "(0 = never chunk; needed for CovidQA/QASPER, whose contexts are whole papers)")
    ap.add_argument("--max_per_para", type=int, default=1,
                    help="questions kept per source passage. 1 = independent units (default); "
                         "a larger value reproduces clustered SQuAD-style sampling")
    ap.add_argument("--pick", default="random", choices=["first", "random"],
                    help="which question to keep per passage; 'random' avoids annotation-order bias")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan", type=int, default=100000, help="max source examples to read")
    args = ap.parse_args()

    # group source examples by their context, so we can keep at most N questions per passage
    by_ctx, order = {}, []
    for i, (q, answers, ctx) in enumerate(LOADERS[args.data]()):
        if i >= args.scan:
            break
        answers = [a for a in answers if a and a.strip()]
        if not answers:
            continue
        if ctx not in by_ctx:
            by_ctx[ctx] = []; order.append(ctx)
        by_ctx[ctx].append((q, answers))

    # The corpus is EVERY passage in the source split, not only the ones we ask questions about.
    # Questions are capped at --target, but the retrieval pool must stay full-size or top-k
    # retrieval is made artificially easy.
    corpus, chunk_id, ctx_chunks = [], {}, {}
    for ctx in order:
        ids = []
        for ch in chunk(ctx, args.chunk_words):
            if ch not in chunk_id:
                chunk_id[ch] = len(corpus); corpus.append(ch)
            ids.append(chunk_id[ch])
        ctx_chunks[ctx] = ids

    rng = random.Random(args.seed)
    questions = []
    n_ctx_used = 0
    for ctx in order:
        if len(questions) >= args.target:
            break
        ids = ctx_chunks[ctx]
        cands = by_ctx[ctx][:]
        if args.pick == "random":
            rng.shuffle(cands)
        kept = 0
        for q, answers in cands:
            if kept >= args.max_per_para or len(questions) >= args.target:
                break
            norm_aliases = sorted({normalize(a) for a in answers if normalize(a)})
            if not norm_aliases:
                continue
            # gold = the chunk that actually contains an answer alias
            gold = [i for i in ids if any(a in normalize(corpus[i]) for a in norm_aliases)]
            if not gold:
                continue
            questions.append({"q": q, "aliases": sorted(set(answers)),
                              "norm_aliases": norm_aliases, "gold_ids": gold[:1]})
            kept += 1
        if kept:
            n_ctx_used += 1

    if not questions:
        raise SystemExit(f"no usable questions for {args.data}")
    P = C.paths(args.data)
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    words = np.mean([len(corpus[q["gold_ids"][0]].split()) for q in questions])
    print(f"{args.data}: questions={len(questions)}  source_passages={n_ctx_used}  "
          f"corpus_chunks={len(corpus)}  mean_gold_words={words:.0f}")

    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    emb = em.encode(corpus, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    np.save(P["emb"], emb)
    print(f"saved corpus embeddings {emb.shape} -> {os.path.basename(P['emb'])}")


if __name__ == "__main__":
    main()
