"""Prepare PubMedQA (pqa_labeled) in the shared RAG schema (same as squad_prep).

Dense biomedical reading comprehension: each question ships an abstract context and a
conclusion-style answer (yes/no/maybe). Tests whether a small receiver UNDER-ENCODES dense
technical context vs the sharer (the fusion-headroom screen). Bundled context -> corpus is the
set of unique abstracts; gold = the question's own abstract.

  python pubmedqa_prep.py --target 500
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
    ap.add_argument("--no_emb", action="store_true", help="skip corpus embeddings (oracle-only screen)")
    args = ap.parse_args()

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ctx_id, corpus, questions = {}, [], []
    for ex in ds:
        if len(questions) >= args.target:
            break
        ctx = " ".join(ex["context"]["contexts"]).strip()
        if not ctx:
            continue
        if ctx not in ctx_id:
            ctx_id[ctx] = len(corpus); corpus.append(ctx)
        dec = ex["final_decision"].strip()          # yes / no / maybe
        questions.append({"q": ex["question"], "aliases": [dec],
                          "norm_aliases": [normalize(dec)], "gold_ids": [ctx_id[ctx]]})

    print(f"kept_questions={len(questions)}  corpus_contexts={len(corpus)}")
    P = C.paths("pubmedqa")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    if not args.no_emb:
        from sentence_transformers import SentenceTransformer
        em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        emb = em.encode(corpus, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        np.save(P["emb"], emb); print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
