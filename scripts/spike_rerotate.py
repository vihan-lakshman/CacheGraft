"""SPIKE 2: validate that (cache pre-fusion KVs at canonical pos 0) -> re-rotate by slot_start
-> fuse  ==  building the fused block directly at slot_start (the reference). If exact, one
canonical cached copy per chunk serves any slot (O(1) storage, not O(k))."""
import os
os.environ.setdefault("USE_TF", "0")
import torch, json
import common as C, kvcache as KV

def build_prefusion(model, ids):
    """Run the fused forward at canonical pos 0; return (base_kv, sharer_kv) pre-fusion caches."""
    L = ids.shape[1]
    inp = torch.cat([ids, ids[:, :1]], 1).cuda()
    model.forward(kv_cache_index=[KV._sec(1, L), KV._sec(-1, 1)], input_ids=inp,
                  attention_mask=torch.ones_like(inp), use_cache=True)
    from rosetta.model.wrapper import clone_kv_cache
    return clone_kv_cache(model.kv_cache_dict[0][0]), clone_kv_cache(model.kv_cache_dict[0][1])

def main():
    model, base, tok = KV.load_c2c(); model.eval()
    for p in model.projector_list: p.eval()
    recv_inv = base.model.rotary_emb.inv_freq.to("cuda")
    shar_inv = model.model_list[1].model.rotary_emb.inv_freq.to("cuda")
    print("recv inv_freq dim:", recv_inv.shape, " shar:", shar_inv.shape)

    chunk = json.load(open(os.path.join(os.path.dirname(__file__), "squad_data.json")))["corpus"][0]
    ids = tok(chunk + "\n\n", add_special_tokens=False, return_tensors="pt").input_ids
    L = ids.shape[1]
    ref0 = KV.build_block_cache(model, ids, 0)
    for S in [320, 640]:
        refS = KV.build_block_cache(model, ids, S)
        kd = max((refS.key_cache[l][:, :, :L, :].float() - ref0.key_cache[l][:, :, :L, :].float()).abs().max().item() for l in range(len(ref0.key_cache)))
        vd = max((refS.value_cache[l][:, :, :L, :].float() - ref0.value_cache[l][:, :, :L, :].float()).abs().max().item() for l in range(len(ref0.value_cache)))
        print(f"MODEL build@{S} vs build@0:  key diff={kd:.4f}  val diff={vd:.4f}  (val should be ~0 if offset-invariant)")
    for S in [0, 320, 640]:
        base_kv, shar_kv = build_prefusion(model, ids)
        my = KV.fuse_caches(model, KV.re_rotate_cache(base_kv, S, recv_inv),
                            KV.re_rotate_cache(shar_kv, S, shar_inv))
        ref = KV.build_block_cache(model, ids, S)            # build fused directly at slot S
        err = max((ref.key_cache[l][:, :, :L, :].float() - my.key_cache[l][:, :, :L, :].float()).abs().max().item()
                  for l in range(len(my.key_cache)))
        verr = max((ref.value_cache[l][:, :, :L, :].float() - my.value_cache[l][:, :, :L, :].float()).abs().max().item()
                   for l in range(len(my.value_cache)))
        print(f"slot_start={S:4d}: max key err={err:.5f}  val err={verr:.5f}  {'OK' if max(err,verr)<1e-2 else 'MISMATCH'}")

if __name__ == "__main__":
    main()
