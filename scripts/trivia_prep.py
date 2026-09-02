"""Prepare a self-contained TriviaQA passage-retrieval set.

Streams rc.wikipedia, chunks each question's wiki article into ~100-word passages,
pools all chunks into one corpus, marks gold chunks (contain an answer alias),
keeps answerable questions, and saves data + corpus embeddings.
"""
import os, json, re, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
from datasets import load_dataset
import common as C

HERE = os.path.dirname(__file__)

def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def chunk_words(text, size=100, max_chunks=60):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)][:max_chunks]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="answerable questions to keep")
    ap.add_argument("--scan", type=int, default=1500, help="max stream examples to scan")
    ap.add_argument("--out", default="trivia",
                    help="dataset name to write (see common.paths). Use a distinct name to build "
                         "a second TriviaQA setting side by side, e.g. --out trivia500")
    args = ap.parse_args()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True)
    corpus, questions = [], []
    seen = 0
    for ex in ds:
        seen += 1
        if seen > args.scan or len(questions) >= args.target:
            break
        wc = ex["entity_pages"].get("wiki_context", [])
        if not wc:
            continue
        aliases = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        norm_aliases = sorted({normalize(a) for a in aliases if a.strip()})
        chunks = chunk_words(" ".join(wc))
        gold_local = [c for c in chunks if any(a and a in normalize(c) for a in norm_aliases)]
        if not gold_local:
            continue  # unanswerable from context -> skip
        base = len(corpus)
        corpus.extend(chunks)
        gold_ids = [base + i for i, c in enumerate(chunks)
                    if any(a and a in normalize(c) for a in norm_aliases)]
        questions.append({"q": ex["question"], "aliases": aliases,
                          "norm_aliases": norm_aliases, "gold_ids": gold_ids})

    P = C.paths(args.out)
    print(f"scanned={seen}  kept_questions={len(questions)}  corpus_chunks={len(corpus)}")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))

    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    emb = em.encode(corpus, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    np.save(P["emb"], emb)
    print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
