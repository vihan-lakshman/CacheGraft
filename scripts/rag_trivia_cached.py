"""TriviaQA-RAG via the OFFLINE-CACHEABLE doc-only path (the hardened artifact).

Phase 1 (offline): build a fused doc KV for every retrieved/oracle passage and persist it to
disk -- the 4B sharer + fuser run ONLY here, once per passage, query-agnostic.
Phase 2 (online): for each question, load its doc cache from disk and decode with the base
0.6B receiver alone (no 4B, no fuser). Generations are saved in the same schema as run_c2c, so
judge_trivia.py / judge_open.py / report_table.py work unchanged.

  python rag_trivia_cached.py --fact retrieved            # or --fact oracle [--terse]
  python judge_trivia.py c2c_retrieved_doc_only_cached    # then score

This reproduces the inline `c2c ... --fuse doc_only` path up to greedy fp nondeterminism.
"""
import os, json, time, argparse
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
import common as C
import kvcache as KV
from rag_trivia import make_content, score   # shared prompt + scoring (copula extraction)

HERE = os.path.dirname(__file__)
DUMMY_Q = "?"   # the doc section excludes the question, so any placeholder works for the build


def resolve_facts(qs, fact, emb_path):
    if fact == "oracle":
        return [q["gold_ids"][0] for q in qs], None
    from sentence_transformers import SentenceTransformer
    cemb = np.load(emb_path)
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"]
                      for q in qs], normalize_embeddings=True, batch_size=128)
    top = (qemb @ cemb.T).argmax(1)
    fact_ids = [int(i) for i in top]
    hit = float(np.mean([fact_ids[i] in set(qs[i]["gold_ids"]) for i in range(len(qs))]))
    del em; torch.cuda.empty_cache()
    return fact_ids, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", default="retrieved", choices=["retrieved", "oracle"])
    ap.add_argument("--terse", action="store_true")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--data", default="trivia")
    ap.add_argument("--recv_kv", action="store_true",
                    help="CONTROL: build the doc cache from the receiver's OWN KV (fusion off) "
                         "through the identical pipeline -- should match text_rag (isolates whether "
                         "the win comes from fusion vs the doc-only-cache machinery)")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    PATHS = C.paths(args.data)
    args.cache_dir = args.cache_dir or PATHS["kvcache"] + ("_recv" if args.recv_kv else "")
    os.makedirs(args.cache_dir, exist_ok=True)

    data = json.load(open(PATHS["data"]))
    corpus = data["corpus"]; qs = data["questions"][:args.n_test]
    fact_ids, hit = resolve_facts(qs, args.fact, PATHS["emb"])
    if hit is not None:
        print(f"retrieval top-1 gold-chunk hit rate: {hit:.3f}", flush=True)

    model, base, tok = KV.load_c2c()

    # ---- Phase 1 (offline): build + persist a fused doc cache per unique passage ----
    keys = []                                  # per-question (cache_path, L_doc, ids, P)
    built, t_build = set(), 0.0
    for i, q in enumerate(qs):
        passage = corpus[fact_ids[i]]
        full = tok.apply_chat_template([{"role": "user", "content": make_content(DUMMY_Q, passage, terse=args.terse)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids, L_doc, _ = KV.split_doc(tok, full)
        key = KV.doc_key(ids, L_doc); path = os.path.join(args.cache_dir, f"{key}.pt")
        if key not in built and not os.path.exists(path):
            t0 = time.time(); cache = KV.build_doc_cache(model, ids, L_doc, fuse=not args.recv_kv)
            KV.save_cache(cache, L_doc, path); t_build += time.time() - t0
        built.add(key); keys.append(path)
    print(f"phase1 offline: built {len(built)} unique doc caches in {t_build:.1f}s", flush=True)

    # ---- Phase 2 (online): load cache from disk, decode question with the base receiver ----
    gens, t_query = [], 0.0
    for i, q in enumerate(qs):
        passage = corpus[fact_ids[i]]
        full = tok.apply_chat_template([{"role": "user", "content": make_content(q["q"], passage, terse=args.terse)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids, L_doc, P = KV.split_doc(tok, full)
        cache, _ = KV.load_cache(keys[i])
        t0 = time.time()
        g = KV.decode_with_cache(base, ids, L_doc, P, cache, tok.eos_token_id)
        t_query += time.time() - t0
        gens.append(tok.decode(g, skip_special_tokens=True))
    print(f"phase2 online: decoded {len(qs)} (receiver-only) in {t_query:.1f}s "
          f"({1000*t_query/len(qs):.0f} ms/q)", flush=True)

    res = [score(g, qs[i]["norm_aliases"]) for i, g in enumerate(gens)]
    em = np.mean([r[0] for r in res]); cont = np.mean([r[1] for r in res]); f1 = np.mean([r[2] for r in res])
    tag = ((f"recvkv_{args.fact}_doc_only" if args.recv_kv else f"c2c_{args.fact}_doc_only_cached")
           + ("_terse" if args.terse else ""))
    out = {"tag": tag, "method": "recv_kv" if args.recv_kv else "c2c_cached",
           "fact": args.fact, "fuse": "receiver_own_kv" if args.recv_kv else "doc_only",
           "terse": args.terse, "n": len(qs), "contains": round(float(cont), 4),
           "em": round(float(em), 4), "f1": round(float(f1), 4), "retr_hit": hit,
           "build_seconds": round(t_build, 1), "query_seconds": round(t_query, 1)}
    os.makedirs(PATHS["results"], exist_ok=True)
    json.dump({**out, "samples": [{"q": qs[k]["q"], "gen": gens[k], "pred": res[k][3],
               "aliases": qs[k]["aliases"][:6], "ok": bool(res[k][1])} for k in range(len(gens))]},
              open(os.path.join(PATHS["results"], f"{tag}.json"), "w"), indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
