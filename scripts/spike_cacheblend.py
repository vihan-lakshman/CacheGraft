"""SPIKE: CacheBlend principle (receiver-only), oracle version.

Cache each chunk's KV independently (block-local = our recvblock), then RESTORE cross-chunk
attention by replacing only the top-RATIO% high-KV-deviation (HKVD) tokens' KV with their
full-attention values. If blend ~= full (and >> naive), selective recompute recovers quality
without fine-tuning -> build the efficient (partial-forward) version next.

  naive = block-local concat (recvblock)   full = contiguous full attention (text)   blend = CacheBlend
"""
import os, json
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers.cache_utils import DynamicCache
import common as C, kvcache as KV, cacheblend as CB
from rag_trivia import score

INSTR = "Answer the question using the context. Give only a short factual answer (no explanation)."
RATIO = 0.15           # fraction of tokens recomputed (HKVD)
CHECK_LAYER = 1        # layer whose K-deviation selects HKVD (as in CacheBlend)

def clone_cache(c):
    o = DynamicCache()
    for k, v in zip(c.key_cache, c.value_cache):
        o.key_cache.append(k.clone()); o.value_cache.append(v.clone())
    return o

@torch.no_grad()
def full_cache(base, ids):
    out = base(input_ids=ids, attention_mask=torch.ones_like(ids),
               position_ids=torch.arange(ids.shape[1], device="cuda").unsqueeze(0), use_cache=True)
    return out.past_key_values

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="squad"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--kk", type=int, default=3); ap.add_argument("--fact", default="retrieved",
                    choices=["retrieved", "oracle"]); ap.add_argument("--skip_blend", action="store_true")
    A = ap.parse_args()
    n, Kk = A.n, A.kk
    PATHS = C.paths(A.data)
    model, base, tok = KV.load_c2c(); model.eval()
    inv = base.model.rotary_emb.inv_freq.to("cuda")
    data = json.load(open(PATHS["data"]))
    corpus, qs = data["corpus"], data["questions"][:n]
    if A.fact == "oracle":      # use the question's gold paragraphs (pad with -1 -> handled below)
        ranks = np.array([(q["gold_ids"] + q["gold_ids"] * Kk)[:Kk] for q in qs])
    else:
        from sentence_transformers import SentenceTransformer
        cemb = np.load(PATHS["emb"]); em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"] for q in qs],
                         normalize_embeddings=True, batch_size=128)
        ranks = np.argsort(-(qemb @ cemb.T), axis=1)[:, :Kk]; del em; torch.cuda.empty_cache()
        hit = np.mean([len(set(ranks[i]) & set(qs[i]["gold_ids"])) >= len(qs[i]["gold_ids"])
                       for i in range(len(qs))])
        print(f"retrieval all-gold-in-top{Kk}: {hit:.2f}", flush=True)

    TPL = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    head, tail = TPL.split("\x00")
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    pre_ids = enc(head + INSTR + "\n\nContext:\n"); P = pre_ids.shape[1]
    eos = tok.eos_token_id

    def decode_from(cache, qids, qpos):
        past, logits, clen = KV.prefill_plain(base, qids, qpos, past=clone_cache(cache))
        return tok.decode(KV.decode_greedy(base, past, logits, clen, qpos + qids.shape[1], eos),
                          skip_special_tokens=True)

    R = {m: {k: [] for k in range(1, Kk + 1)} for m in ["naive", "blend", "full"]}
    SAMP = {m: {k: [] for k in range(1, Kk + 1)} for m in ["naive", "blend", "full"]}
    AG = {m: {k: [] for k in range(1, Kk + 1)} for m in ["naive", "blend"]}  # token-agreement vs full
    for i in range(n):
        chunks = [enc(corpus[int(ranks[i][j])] + "\n\n")[:, :320] for j in range(Kk)]
        qids = enc("Question: " + qs[i]["q"] + tail)
        for kk in range(1, Kk + 1):
            doc = torch.cat([pre_ids] + chunks[:kk], 1); N = doc.shape[1]
            full = full_cache(base, doc)
            blocks = [full_cache(base, pre_ids)]; off = P                  # naive: block-local
            for j in range(kk):
                blocks.append(KV.re_rotate_cache(full_cache(base, chunks[j]), off, inv))
                off += chunks[j].shape[1]
            naive = KV.cat_caches(blocks)
            methods = [("naive", naive), ("full", full)]
            if not A.skip_blend:    # efficient CacheBlend: recompute top-RATIO% HKVD (self-consistent)
                methods.insert(1, ("blend", CB.blend_forward(base, doc, naive, ratio=RATIO, check_layer=CHECK_LAYER)))
            gens = {}
            for name, c in methods:
                g = decode_from(c, qids, N)
                gens[name] = g.strip()
                sc = score(g, qs[i]["norm_aliases"])
                R[name][kk].append(sc[1])
                SAMP[name][kk].append({"q": qs[i]["q"], "gen": g, "pred": sc[3],
                                       "aliases": qs[i]["aliases"][:6], "ok": bool(sc[1])})
            AG["naive"][kk].append(gens["naive"] == gens["full"])
            if not A.skip_blend:
                AG["blend"][kk].append(gens["blend"] == gens["full"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}", flush=True)

    names = ["naive", "full"] + ([] if A.skip_blend else ["blend"])
    os.makedirs(PATHS["results"], exist_ok=True)
    import json as _j
    for name in names:
        for kk in range(1, Kk + 1):
            tag = f"{name}_{A.fact}_cbtopk{kk}"
            _j.dump({"tag": tag, "n": n, "samples": SAMP[name][kk]},
                    open(os.path.join(PATHS["results"], f"{tag}.json"), "w"))

    bl = lambda d, kk: ("" if A.skip_blend else f"{np.mean(d[kk])*100:>7.1f}")
    blag = lambda kk: ("" if A.skip_blend else f"{np.mean(AG['blend'][kk])*100:>11.0f}%")
    print(f"\n{A.data} {A.fact} (n={n}, recomp={int(RATIO*100)}%):  contains | token-agreement-with-full")
    print(f"{'k':>3}{'naive':>8}{'blend':>7}{'full':>7}  | {'naive=full':>11}{'blend=full':>12}")
    for kk in range(1, Kk + 1):
        print(f"{kk:>3}{np.mean(R['naive'][kk])*100:>7.1f}{bl(R['blend'], kk):>7}"
              f"{np.mean(R['full'][kk])*100:>7.1f}  | {np.mean(AG['naive'][kk])*100:>10.0f}%{blag(kk)}")

if __name__ == "__main__":
    main()
