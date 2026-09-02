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

def _squad_like(repo, split, config=None, header=None):
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    for ex in ds:
        h = header(ex["context"]) if header else ""
        yield ex["question"], list(ex["answers"]["text"]), ex["context"], h


def load_squad():
    """SQuAD v1.1 dev: single-hop extractive QA over Wikipedia paragraphs."""
    return _squad_like("rajpurkar/squad", "validation")


def load_adversarialqa():
    """AdversarialQA: SQuAD-format, but questions were written adversarially against a reader,
    so the encoding burden per passage is higher by construction."""
    return _squad_like("UCLNLP/adversarial_qa", "validation", config="adversarialQA")


def load_covidqa():
    """CovidQA (deepset): SQuAD-format extractive QA over CORD-19 papers. Contexts are full
    papers, so chunking matters here; answers are outside a small model's parametric memory.
    CovidQA has no title field, but every context begins with the paper title, so the first
    line serves as the header."""
    return _squad_like("deepset/covid_qa_deepset", "train",
                       header=lambda c: c.strip().split("\n")[0].strip())


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
                yield q, sorted(set(spans)), context, f"{paper['title']}. {paper['abstract']}"


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
    ap.add_argument("--header", action="store_true",
                    help="prepend the document's title (CovidQA) or title+abstract (QASPER) to "
                         "every chunk of it, so a paragraph-level chunk is self-describing to "
                         "both the retriever and the reader")
    ap.add_argument("--header_words", type=int, default=40,
                    help="truncate the header to this many words; a full QASPER abstract would "
                         "otherwise dwarf a 140-word chunk")
    ap.add_argument("--scope", default="corpus", choices=["corpus", "document"],
                    help="'corpus' = retrieve over every chunk in the split. 'document' = "
                         "restrict retrieval to the question's own source document (records "
                         "pool_ids per question); the realistic setting for QA over a known "
                         "paper, and what makes paragraph-level retrieval tractable there")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--scan", type=int, default=100000, help="max source examples to read")
    args = ap.parse_args()

    # group source examples by their context, so we can keep at most N questions per passage
    by_ctx, order, headers = {}, [], {}
    for i, (q, answers, ctx, header) in enumerate(LOADERS[args.data]()):
        if i >= args.scan:
            break
        answers = [a for a in answers if a and a.strip()]
        if not answers:
            continue
        if ctx not in by_ctx:
            by_ctx[ctx] = []; order.append(ctx); headers[ctx] = header
        by_ctx[ctx].append((q, answers))

    # The corpus is EVERY passage in the source split, not only the ones we ask questions about.
    # Questions are capped at --target, but the retrieval pool must stay full-size or top-k
    # retrieval is made artificially easy.
    corpus, chunk_id, ctx_chunks = [], {}, {}
    for ctx in order:
        prefix = ""
        if args.header and headers.get(ctx):
            prefix = " ".join(headers[ctx].split()[:args.header_words]).strip()
            prefix = prefix + ("" if prefix.endswith((".", "?", "!")) else ".") + "\n\n"
        ids = []
        for ch in chunk(ctx, args.chunk_words):
            ch = prefix + ch
            if ch not in chunk_id:
                chunk_id[ch] = len(corpus); corpus.append(ch)
            ids.append(chunk_id[ch])
        ctx_chunks[ctx] = ids

    # De-cluster on the GOLD CHUNK, not the source context. For un-chunked corpora (SQuAD,
    # AdversarialQA) a context is one chunk, so this is exactly "one question per paragraph".
    # For chunked long-document corpora (CovidQA, QASPER) a single paper supplies many
    # independent passages, and capping per paper would throw most of the dataset away --
    # CovidQA has only ~147 documents but thousands of distinct answer-bearing passages.
    rng = random.Random(args.seed)
    questions, per_gold = [], {}
    for ctx in order:
        if len(questions) >= args.target:
            break
        ids = ctx_chunks[ctx]
        cands = by_ctx[ctx][:]
        if args.pick == "random":
            rng.shuffle(cands)
        for q, answers in cands:
            if len(questions) >= args.target:
                break
            norm_aliases = sorted({normalize(a) for a in answers if normalize(a)})
            if not norm_aliases:
                continue
            # gold = the chunk that actually contains an answer alias
            gold = [i for i in ids if any(a in normalize(corpus[i]) for a in norm_aliases)]
            if not gold or per_gold.get(gold[0], 0) >= args.max_per_para:
                continue
            # ALL answer-bearing chunks are gold (as in trivia_prep.py): retrieval that
            # returns a different chunk containing the answer is a hit, not a miss. The oracle
            # condition and the de-clustering key both use gold_ids[0] as the canonical passage.
            rec = {"q": q, "aliases": sorted(set(answers)),
                   "norm_aliases": norm_aliases, "gold_ids": gold}
            if args.scope == "document":
                rec["pool_ids"] = ids
            questions.append(rec)
            per_gold[gold[0]] = per_gold.get(gold[0], 0) + 1
    n_ctx_used = len(per_gold)

    if not questions:
        raise SystemExit(f"no usable questions for {args.data}")
    P = C.paths(args.data)
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    words = np.mean([len(corpus[q["gold_ids"][0]].split()) for q in questions])
    pool = np.mean([len(q.get("pool_ids", corpus)) for q in questions])
    print(f"{args.data}: questions={len(questions)}  distinct_gold_passages={n_ctx_used}  "
          f"corpus_chunks={len(corpus)}  mean_gold_words={words:.0f}  "
          f"mean_retrieval_pool={pool:.0f}")

    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    emb = em.encode(corpus, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    np.save(P["emb"], emb)
    print(f"saved corpus embeddings {emb.shape} -> {os.path.basename(P['emb'])}")


if __name__ == "__main__":
    main()
