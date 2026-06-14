"""The fundamental text-RAG baselines on HotpotQA: the RECEIVER (0.6B) alone and the SHARER (4B)
alone, reading the top-k retrieved passages as TEXT (full attention). C2C must beat the receiver
baseline; the sharer is the knowledge ceiling. Batched/fast.

  python hotpot_baselines.py --kk 4 --n 150
"""
import os, json, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
import common as C
from rag_trivia import run_hf, make_content, score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hotpot"); ap.add_argument("--kk", type=int, default=4)
    ap.add_argument("--n", type=int, default=150)
    A = ap.parse_args(); P = C.paths(A.data)
    data = json.load(open(P["data"])); corpus, qs = data["corpus"], data["questions"][:A.n]

    import torch
    from sentence_transformers import SentenceTransformer
    cemb = np.load(P["emb"]); em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"] for q in qs],
                     normalize_embeddings=True, batch_size=128)
    ranks = np.argsort(-(qemb @ cemb.T), axis=1)[:, :A.kk]
    del em; torch.cuda.empty_cache()

    os.makedirs(P["results"], exist_ok=True)
    for who, model in [("recv", C.RECEIVER), ("share", C.SHARER_BASE)]:
        for kk in range(1, A.kk + 1):
            ctx = ["\n\n".join(corpus[int(ranks[i][j])] for j in range(kk)) for i in range(len(qs))]
            contents = [make_content(qs[i]["q"], ctx[i]) for i in range(len(qs))]
            gens = run_hf(model, contents)
            res = [score(g, qs[i]["norm_aliases"]) for i, g in enumerate(gens)]
            tag = f"{who}rag_topk{kk}"
            cont = float(np.mean([r[1] for r in res]))
            json.dump({"tag": tag, "n": len(qs), "contains": round(cont, 4),
                       "samples": [{"q": qs[i]["q"], "gen": gens[i], "pred": res[i][3],
                                    "aliases": qs[i]["aliases"][:6], "ok": bool(res[i][1])} for i in range(len(gens))]},
                      open(os.path.join(P["results"], f"{tag}.json"), "w"))
            print(f"{tag:<16} contains={cont*100:.1f}", flush=True)

if __name__ == "__main__":
    main()
