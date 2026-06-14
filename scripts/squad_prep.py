"""Prepare a self-contained SQuAD v1.1 retrieval set (same schema as trivia_prep).

SQuAD bundles a context paragraph with every question, so the corpus is just the set of unique
dev contexts (single-hop; gold = the question's own paragraph). Writes squad_data.json
({questions:[{q,aliases,norm_aliases,gold_ids}], corpus:[...]}) + squad_corpus_emb.npy, so the
whole RAG harness runs via `--data squad`.

  python squad_prep.py --target 500
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
    ap.add_argument("--target", type=int, default=500, help="questions to keep")
    args = ap.parse_args()

    ds = load_dataset("rajpurkar/squad", split="validation")
    # corpus = unique dev contexts (the full retrieval pool, incl. distractors)
    ctx_id = {}
    corpus = []
    for ex in ds:
        c = ex["context"]
        if c not in ctx_id:
            ctx_id[c] = len(corpus); corpus.append(c)

    questions = []
    for ex in ds:
        if len(questions) >= args.target:
            break
        aliases = sorted(set(ex["answers"]["text"]))
        norm_aliases = sorted({normalize(a) for a in aliases if a.strip()})
        if not norm_aliases:
            continue
        gid = ctx_id[ex["context"]]
        # sanity: the answer must actually be in the gold context (true for SQuAD v1.1)
        if not any(a in normalize(corpus[gid]) for a in norm_aliases):
            continue
        questions.append({"q": ex["question"], "aliases": aliases,
                          "norm_aliases": norm_aliases, "gold_ids": [gid]})

    print(f"kept_questions={len(questions)}  corpus_contexts={len(corpus)}")
    P = C.paths("squad")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))

    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    emb = em.encode(corpus, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    np.save(P["emb"], emb)
    print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
