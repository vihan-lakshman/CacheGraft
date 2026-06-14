"""C2C + CacheBlend (store-both design) correctness + recovery check.

naive = block-local fused (re-rotate base+sharer, fuse)         [lossy, our top-k result]
blend = CacheBlend the base cache, then fuse                    [recovers cross-chunk, 4B offline]
full  = full-attention C2C (fuse whole doc jointly)             [ceiling]

Checks: k=1 ~= monolithic C2C; blend agreement-with-full >> naive at k>1.
"""
import os, json
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
import common as C, kvcache as KV, cacheblend as CB
from rag_trivia import score

INSTR = "Answer the question using the context. Give only a short factual answer (no explanation)."
RATIO, CHECK = 0.15, 1

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="squad"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kk", type=int, default=3); ap.add_argument("--fact", default="retrieved",
                    choices=["retrieved", "oracle"])
    A = ap.parse_args(); n, Kk = A.n, A.kk
    PATHS = C.paths(A.data)
    model, base, tok = KV.load_c2c(); model.eval()
    for p in model.projector_list: p.eval()
    rinv = base.model.rotary_emb.inv_freq.cuda()
    sinv = model.model_list[1].model.rotary_emb.inv_freq.cuda()
    data = json.load(open(PATHS["data"])); corpus, qs = data["corpus"], data["questions"][:n]
    if A.fact == "oracle":
        ranks = np.array([(q["gold_ids"] + q["gold_ids"] * Kk)[:Kk] for q in qs])
    else:
        from sentence_transformers import SentenceTransformer
        cemb = np.load(PATHS["emb"]); em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"] for q in qs],
                         normalize_embeddings=True, batch_size=128)
        ranks = np.argsort(-(qemb @ cemb.T), axis=1)[:, :Kk]; del em; torch.cuda.empty_cache()
    TPL = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    head, tail = TPL.split("\x00")
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    pre = enc(head + INSTR + "\n\nContext:\n"); P = pre.shape[1]; eos = tok.eos_token_id

    def dec(cache, qids, qpos):
        from transformers.cache_utils import DynamicCache
        c = DynamicCache()
        c.key_cache = [k.clone() for k in cache.key_cache]; c.value_cache = [v.clone() for v in cache.value_cache]
        past, logits, clen = KV.prefill_plain(base, qids, qpos, past=c)
        return tok.decode(KV.decode_greedy(base, past, logits, clen, qpos + qids.shape[1], eos), skip_special_tokens=True)

    R = {m: {k: [] for k in range(1, Kk+1)} for m in ["naive","blend","full"]}
    SAMP = {m: {k: [] for k in range(1, Kk+1)} for m in ["naive","blend","full"]}
    AG = {m: {k: [] for k in range(1, Kk+1)} for m in ["naive","blend"]}
    pb, ps = KV.build_prefusion(model, pre)                          # preamble base, sharer
    for i in range(n):
        chunks = [enc(corpus[int(ranks[i][j])] + "\n\n")[:, :320] for j in range(Kk)]
        cbs = [KV.build_prefusion_prefixed(model, pre, c) for c in chunks]  # (base,sharer) per chunk @P
        qids = enc("Question: " + qs[i]["q"] + tail)
        for kk in range(1, Kk+1):
            doc = torch.cat([pre] + chunks[:kk], 1); N = doc.shape[1]
            full = KV.build_block_cache(model, doc, 0)               # full-attention C2C ceiling
            bbl = [pb]; sbl = [ps]; off = P
            for j in range(kk):
                d = off - P
                bbl.append(KV.re_rotate_cache(cbs[j][0], d, rinv)); sbl.append(KV.re_rotate_cache(cbs[j][1], d, sinv))
                off += chunks[j].shape[1]
            naive_base = KV.cat_caches(bbl); sharer = KV.cat_caches(sbl)
            naive = KV.fuse_caches(model, naive_base, sharer)
            blend = CB.blend_forward_c2c(model, base, doc, naive_base, sharer, RATIO, CHECK)
            g = {}
            for name, c in [("naive", naive), ("blend", blend), ("full", full)]:
                gen = dec(c, qids, N); g[name] = gen.strip()
                sc = score(gen, qs[i]["norm_aliases"]); R[name][kk].append(sc[1])
                SAMP[name][kk].append({"q": qs[i]["q"], "gen": gen, "pred": sc[3],
                                       "aliases": qs[i]["aliases"][:6], "ok": bool(sc[1])})
            AG["naive"][kk].append(g["naive"] == g["full"]); AG["blend"][kk].append(g["blend"] == g["full"])

    import json as _j; os.makedirs(PATHS["results"], exist_ok=True)
    for name in ["naive", "blend", "full"]:
        for kk in range(1, Kk + 1):
            tag = f"c2c{name}_{A.fact}_cbtopk{kk}"
            _j.dump({"tag": tag, "n": n, "samples": SAMP[name][kk]},
                    open(os.path.join(PATHS["results"], f"{tag}.json"), "w"))

    print(f"\nC2C+CacheBlend {A.data} {A.fact} (n={n}, recomp={int(RATIO*100)}%):  contains | agreement-with-full")
    print(f"{'k':>3}{'naive':>8}{'blend':>7}{'full':>7}  |{'naive=full':>11}{'blend=full':>12}")
    for kk in range(1, Kk+1):
        print(f"{kk:>3}{np.mean(R['naive'][kk])*100:>7.1f}{np.mean(R['blend'][kk])*100:>7.1f}{np.mean(R['full'][kk])*100:>7.1f}"
              f"  |{np.mean(AG['naive'][kk])*100:>10.0f}%{np.mean(AG['blend'][kk])*100:>11.0f}%")

if __name__ == "__main__":
    main()
