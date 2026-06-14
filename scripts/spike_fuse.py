"""SPIKE 1: validate that fuse_caches (fuser run on cached KVs, no model forward) reproduces the
RosettaModel's own fused KV. If this matches, we can cache pre-fusion KVs and fuse at query time.
"""
import os
os.environ.setdefault("USE_TF", "0")
import torch, json
import common as C, kvcache as KV

def main():
    model, base, tok = KV.load_c2c()
    model.eval()
    for p in model.projector_list:
        p.eval()
    print("projector[0] training mode:", model.projector_list[0].training)
    data = json.load(open(os.path.join(os.path.dirname(__file__), "squad_data.json")))
    chunk = data["corpus"][0]
    ids = tok(chunk + "\n\n", add_special_tokens=False, return_tensors="pt").input_ids
    L = ids.shape[1]
    inp = torch.cat([ids, ids[:, :1]], 1).cuda()
    # capture each projector's (inputs, output) during the reference forward
    cap = {}
    for idx, p in enumerate(model.projector_list):
        orig = p.forward
        def mk(orig, idx):
            def f(src, tgt, *a, **k):
                out = orig(src, tgt, *a, **k)
                cap[idx] = (src[0].detach().clone(), tgt[0].detach().clone(), out[0].detach().clone())
                return out
            return f
        p.forward = mk(orig, idx)
    # reference fused via the model (doc fused as non-final section)
    out = model.forward(kv_cache_index=[KV._sec(1, L), KV._sec(-1, 1)], input_ids=inp,
                        attention_mask=torch.ones_like(inp), use_cache=True)
    fused_ref = out.past_key_values                          # fused (doc+dummy)
    base_kv = model.kv_cache_dict[0][0]                      # pre-fusion receiver KV (doc)
    sharer_kv = model.kv_cache_dict[0][1]                    # 4B KV (doc)
    print(f"L={L}  base layers={len(base_kv.key_cache)} sharer layers={len(sharer_kv.key_cache)} "
          f"fused layers={len(fused_ref.key_cache)}")
    # diagnostics: layer mapping + per-layer error
    pd = model.projector_dict[0][1]
    print("projector_dict[0][1] sample (target->[(source,proj)]):",
          {k: pd[k] for k in list(pd)[:4]})
    print("multi_source_fusion_mode:", getattr(model, "multi_source_fusion_mode", "?"))
    # my standalone fusion
    my = KV.fuse_caches(model, base_kv, sharer_kv)
    print("per-layer key err (first 6):",
          [round((fused_ref.key_cache[l][:, :, :L, :].float() - my.key_cache[l][:, :, :L, :].float()).abs().max().item(), 4)
           for l in range(6)])
    # compare MY projector inputs to the captured wrapper inputs, for proj idx 1 (target layer 1)
    for idx in [1, 2]:
        src_cap, tgt_cap, out_cap = cap[idx]
        tl = [t for t in model.projector_dict[0][1] if model.projector_dict[0][1][t][0][1] == idx][0]
        sl = model.projector_dict[0][1][tl][0][0]
        my_out = model.projector_list[idx].forward(
            (sharer_kv.key_cache[sl], sharer_kv.value_cache[sl]),
            (base_kv.key_cache[tl], base_kv.value_cache[tl]))[0]
        print(f"proj {idx} (target {tl}, source {sl}): "
              f"src_in_err={ (src_cap-sharer_kv.key_cache[sl]).abs().max().item():.4f} "
              f"tgt_in_err={ (tgt_cap-base_kv.key_cache[tl]).abs().max().item():.4f} "
              f"out_err(wrapper vs my recompute)={ (out_cap-my_out).abs().max().item():.4f}")
    # compare on the doc span [0:L]
    maxerr = 0.0; nfused = 0
    for l in range(len(my.key_cache)):
        a = fused_ref.key_cache[l][:, :, :L, :]
        b = my.key_cache[l][:, :, :L, :]
        e = (a.float() - b.float()).abs().max().item()
        maxerr = max(maxerr, e)
        if not torch.allclose(a, base_kv.key_cache[l], atol=1e-3):
            nfused += 1
    print(f"fused layers (differ from base): {nfused}/{len(my.key_cache)}")
    print(f"max |fused_ref - my_fused| over keys: {maxerr:.6f}")
    # decisive: is the FINAL cache for layer 1 equal to the projector output captured at fusion?
    out1 = cap[1][2]
    print("fused_ref[1][:L] vs captured proj-out:",
          round((fused_ref.key_cache[1][:, :, :L, :] - out1).abs().max().item(), 4))
    print("fused_ref[1][:L] vs base[1]:",
          round((fused_ref.key_cache[1][:, :, :L, :] - base_kv.key_cache[1]).abs().max().item(), 4))
    print("captured proj-out vs base[1] (fusion delta):",
          round((out1 - base_kv.key_cache[1]).abs().max().item(), 4))
    print("MATCH" if maxerr < 1e-2 else "MISMATCH")

    # end-to-end: does decoding from my_fused match decoding from the model's fused cache?
    import copy
    q = json.load(open(os.path.join(os.path.dirname(__file__), "squad_data.json")))["questions"][0]["q"]
    qids = tok("\n\nQuestion: " + q + "\nAnswer:", add_special_tokens=False, return_tensors="pt").input_ids
    def dec(fused):
        from rosetta.model.wrapper import clone_kv_cache
        c = clone_kv_cache(fused)
        for l in range(len(c.key_cache)):                 # truncate dummy -> L
            c.key_cache[l] = c.key_cache[l][:, :, :L, :].contiguous()
            c.value_cache[l] = c.value_cache[l][:, :, :L, :].contiguous()
        g = KV.decode_with_cache(base, torch.cat([torch.zeros(1, L, dtype=torch.long), qids], 1),
                                 L, L + qids.shape[1], c, tok.eos_token_id, max_new=20)
        return tok.decode(g, skip_special_tokens=True)
    print("decode(model fused) :", repr(dec(fused_ref)))
    print("decode(my   fused)  :", repr(dec(my)))

if __name__ == "__main__":
    main()
