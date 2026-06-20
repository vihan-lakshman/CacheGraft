"""Prepare NarrativeQA (summaries setting) in the shared RAG schema.

Abstractive QA over a narrative SUMMARY (~500 words). Answers are paraphrased, not literal spans,
and require understanding the story -> the ideal "encoding-hard + non-extractive" fusion probe.
Context = the story summary; gold = the question's own summary; aliases = the reference answers.

  python narrativeqa_prep.py --target 500
"""
import os, json, re, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
from datasets import load_dataset
import common as C

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--no_emb", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("deepmind/narrativeqa", split="validation")
    ctx_id, corpus, questions = {}, [], []
    for ex in ds:
        if len(questions) >= args.target:
            break
        ctx = ex["document"]["summary"]["text"].strip()
        if not ctx:
            continue
        if ctx not in ctx_id:
            ctx_id[ctx] = len(corpus); corpus.append(ctx)
        aliases = sorted({a["text"].strip() for a in ex["answers"] if a["text"].strip()})
        norm = sorted({normalize(a) for a in aliases if normalize(a)})
        if not norm:
            continue
        questions.append({"q": ex["question"]["text"], "aliases": aliases,
                          "norm_aliases": norm, "gold_ids": [ctx_id[ctx]]})

    print(f"kept_questions={len(questions)}  corpus_contexts={len(corpus)}")
    P = C.paths("narrativeqa")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    if not args.no_emb:
        from sentence_transformers import SentenceTransformer
        em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        emb = em.encode(corpus, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        np.save(P["emb"], emb); print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
