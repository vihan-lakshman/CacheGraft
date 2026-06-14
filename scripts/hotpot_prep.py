"""Prepare a self-contained HotpotQA (distractor) retrieval set (same schema as trivia/squad_prep).

HotpotQA bundles 10 paragraphs per question (2 gold supporting + 8 distractors) and is MULTI-HOP
(usually needs 2 paragraphs) -> the decisive test for whether cross-document attention matters.
Corpus = unique paragraphs pooled across the sampled questions; gold_ids = the question's gold
(supporting-fact) paragraphs. Writes hotpot_data.json + hotpot_corpus_emb.npy -> run via --data hotpot.

  python hotpot_prep.py --target 500
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
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=500)
    args = ap.parse_args()
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    pid = {}; corpus = []; questions = []
    for ex in ds:
        if len(questions) >= args.target:
            break
        titles = ex["context"]["title"]; sents = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        gold_ids = []
        for t, ss in zip(titles, sents):
            para = (t + ". " + " ".join(ss)).strip()
            if para not in pid:
                pid[para] = len(corpus); corpus.append(para)
            if t in gold_titles:
                gold_ids.append(pid[para])
        ans = ex["answer"]
        norm_aliases = sorted({normalize(ans)} - {""})
        if not norm_aliases or not gold_ids:
            continue
        questions.append({"q": ex["question"], "aliases": [ans], "norm_aliases": norm_aliases,
                          "gold_ids": sorted(set(gold_ids)), "type": ex["type"]})

    ng = np.mean([len(q["gold_ids"]) for q in questions])
    print(f"kept_questions={len(questions)}  corpus_paras={len(corpus)}  avg_gold_paras={ng:.2f}")
    P = C.paths("hotpot")
    json.dump({"questions": questions, "corpus": corpus}, open(P["data"], "w"))
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    emb = em.encode(corpus, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    np.save(P["emb"], emb); print("saved corpus embeddings", emb.shape)

if __name__ == "__main__":
    main()
