"""Top-k C2C-RAG, precompute + prefill (4B always offline). Two cacheable composition modes:

  --mode rerotate : cache pre-fusion KVs (receiver + 4B) ONCE per chunk (O(1) storage); at query
                    re-rotate to the slot position + run the fuser. Approximate (bf16 RoPE).
  --mode slot     : cache the FUSED chunk KV per (chunk, slot) (O(k) storage); exact, no fuser
                    at query. The exact reference.

Both: compose [preamble | block@slot1 | ... | block@slotk | question] and decode. Writes
c2c_retrieved_topk{k}_{mode}; score with judge_trivia.py.

  python rag_topk.py --data squad --k 3 --mode rerotate --n_test 500
"""
import os, json, time, argparse
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers.cache_utils import DynamicCache
import common as C
import kvcache as KV
from rag_trivia import make_content, score

INSTR = ("Answer the question using the context. Give only a short factual answer "
         "(no explanation).")

def to_dev(cache, dev):
    out = DynamicCache()
    for k, v in zip(cache.key_cache, cache.value_cache):
        out.key_cache.append(k.to(dev)); out.value_cache.append(v.to(dev))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="squad")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", default="rerotate", choices=["rerotate", "slot", "text", "recvblock"],
                    help="text = contiguous full-attention text-RAG (ideal). recvblock = receiver "
                         "KV block-composed (same block-local mechanism as C2C but NO fuser) -- "
                         "isolates the block-composition cost from the fusion. rerotate/slot = C2C.")
    ap.add_argument("--lmax", type=int, default=320)
    ap.add_argument("--n_test", type=int, default=500)
    args = ap.parse_args()
    PATHS = C.paths(args.data)
    data = json.load(open(PATHS["data"])); corpus = data["corpus"]; qs = data["questions"][:args.n_test]

    from sentence_transformers import SentenceTransformer
    cemb = np.load(PATHS["emb"]); em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"] for q in qs],
                     normalize_embeddings=True, batch_size=128)
    ranks = np.argsort(-(qemb @ cemb.T), axis=1)[:, :args.k]
    del em; torch.cuda.empty_cache()

    model, base, tok = KV.load_c2c(); model.eval()
    for p in model.projector_list: p.eval()
    eos = tok.eos_token_id
    recv_inv = base.model.rotary_emb.inv_freq.to("cuda")
    shar_inv = model.model_list[1].model.rotary_emb.inv_freq.to("cuda")

    TPL = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    head, tail = TPL.split("\x00")
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids
    pre_ids = enc(head + INSTR + "\n\nContext:\n"); P = pre_ids.shape[1]
    slot_start = lambda j: P + (j - 1) * args.lmax
    print(f"mode={args.mode} P={P} lmax={args.lmax} k={args.k} n={len(qs)}", flush=True)

    block_ids_of = lambda cid: enc(corpus[cid] + "\n\n")[:, :args.lmax]

    # ---- Phase 1 (offline): build caches (skipped for the text control) ----
    canon, fused_slot, canon_recv = {}, {}, {}
    t_build = 0.0
    for i in range(len(qs)) if args.mode != "text" else []:
        for j in range(1, args.k + 1):
            cid = int(ranks[i][j - 1])
            t0 = time.time()
            if args.mode == "rerotate":
                if cid not in canon:                                  # O(1): once per chunk
                    b, s = KV.build_prefusion_prefixed(model, pre_ids, block_ids_of(cid))
                    canon[cid] = (to_dev(b, "cpu"), to_dev(s, "cpu"))
            elif args.mode == "recvblock":
                if cid not in canon_recv:                             # receiver-only, no fuser
                    rc = KV.receiver_kv_prefixed(base, pre_ids, block_ids_of(cid))
                    canon_recv[cid] = to_dev(rc, "cpu")
            else:
                if (cid, j) not in fused_slot:                        # O(k): per (chunk, slot)
                    fc = KV.build_block_prefixed(model, pre_ids, block_ids_of(cid), slot_start(j))
                    fused_slot[(cid, j)] = to_dev(fc, "cpu")
            t_build += time.time() - t0
    print(f"phase1 offline: {len(canon) or len(fused_slot) or len(canon_recv)} caches built "
          f"in {t_build:.1f}s", flush=True)

    def block_at(cid, j):
        delta = slot_start(j) - P                         # chunk built at canonical pos P (j=1 -> 0)
        if args.mode == "rerotate":
            b, s = canon[cid]
            return KV.fuse_caches(model, KV.re_rotate_cache(to_dev(b, "cuda"), delta, recv_inv),
                                  KV.re_rotate_cache(to_dev(s, "cuda"), delta, shar_inv))
        if args.mode == "recvblock":                      # receiver KV, re-rotated, NO fuser
            return KV.re_rotate_cache(to_dev(canon_recv[cid], "cuda"), delta, recv_inv)
        return to_dev(fused_slot[(cid, j)], "cuda")

    # preamble block: FUSED for C2C modes; plain receiver KV for the recvblock control.
    if args.mode == "text":
        pre_kv = None
    elif args.mode == "recvblock":
        pre_kv = KV.prefill_plain(base, pre_ids, 0, past=None)[0]
    else:
        pre_kv = to_dev(KV.build_block_cache(model, pre_ids, 0), "cuda")

    # ---- Phase 2 (online): for each k, compose + decode ----
    for kk in range(1, args.k + 1):
        gens, t_q = [], 0.0
        for i in range(len(qs)):
            qids = enc("Question: " + qs[i]["q"] + tail)
            t0 = time.time()
            if args.mode == "text":   # contiguous text-RAG control: concat passages, full attention
                full = torch.cat([pre_ids] + [block_ids_of(int(ranks[i][j - 1])) for j in range(1, kk + 1)]
                                 + [qids], dim=1)
                past, logits, clen = KV.prefill_plain(base, full, 0, past=None)
                g = KV.decode_greedy(base, past, logits, clen, full.shape[1], eos)
            else:
                blocks = [block_at(int(ranks[i][j - 1]), j) for j in range(1, kk + 1)]
                composed = KV.cat_caches([pre_kv] + blocks)
                L_last = block_ids_of(int(ranks[i][kk - 1])).shape[1]
                q_start = slot_start(kk) + L_last         # right after last chunk (k=1 -> P+L = monolithic)
                past, logits, clen = KV.prefill_plain(base, qids, q_start, past=composed)
                g = KV.decode_greedy(base, past, logits, clen, q_start + qids.shape[1], eos)
            t_q += time.time() - t0
            gens.append(tok.decode(g, skip_special_tokens=True))
        res = [score(g, qs[i]["norm_aliases"]) for i, g in enumerate(gens)]
        tagm = ({"text": f"text_retrieved_topk{kk}", "recvblock": f"recvblock_retrieved_topk{kk}"}
                .get(args.mode, f"c2c_retrieved_topk{kk}_{args.mode}"))
        out = {"tag": tagm, "mode": args.mode, "k": kk,
               "lmax": args.lmax, "n": len(qs),
               "contains": round(float(np.mean([r[1] for r in res])), 4),
               "em": round(float(np.mean([r[0] for r in res])), 4),
               "f1": round(float(np.mean([r[2] for r in res])), 4), "query_seconds": round(t_q, 1)}
        os.makedirs(PATHS["results"], exist_ok=True)
        json.dump({**out, "samples": [{"q": qs[k]["q"], "gen": gens[k], "pred": res[k][3],
                   "aliases": qs[k]["aliases"][:6], "ok": bool(res[k][1])} for k in range(len(gens))]},
                  open(os.path.join(PATHS["results"], f"{out['tag']}.json"), "w"), indent=2)
        print(json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
