"""Prepare BoolQ in the shared RAG schema (same as squad_prep).

Yes/no questions over a Wikipedia passage. The answer is rarely a literal span -- it requires
COMBINING facts in the passage (non-extractive inference), so it probes whether fusion's digested
representation helps where span-copying can't. Bundled context -> corpus is unique passages;
gold = the question's own passage.

  python boolq_prep.py --target 500
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

    ds = load_dataset("google/boolq", split="validation")
    ctx_id, corpus, questions = {}, [], []
    for ex in ds:
        if len(questions) >= args.target:
            break
        ctx = ex["passage"].strip()
        if ctx not in ctx_id:
            ctx_id[ctx] = len(corpus); corpus.append(ctx)
        ans = "yes" if ex["answer"] else "no"
        questions.append({"q": ex["question"], "aliases": [ans],
                          "norm_aliases": [ans], "gold_ids": [ctx_id[ctx]]})

    print(f"kept_questions={len(questions)}  corpus_contexts={len(corpus)}")
    P = C.paths("boolq")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    if not args.no_emb:
        from sentence_transformers import SentenceTransformer
        em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        emb = em.encode(corpus, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        np.save(P["emb"], emb); print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
