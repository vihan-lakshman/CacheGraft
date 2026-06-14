"""Efficient CacheBlend (receiver-only): a manual Qwen3 layer loop that recomputes only the
top-ratio% high-KV-deviation (HKVD) tokens to restore cross-chunk attention over a block-local
KV cache. Self-consistent (HKVD tokens attend to the mixed cache), training-free.

check_layer = L (>= num layers) + ratio=1 => recomputes ALL tokens at ALL layers == full
attention; used to validate the manual forward against the model. ratio<1 => CacheBlend.
"""
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


@torch.no_grad()
def blend_forward(base, doc_ids, naive, ratio=0.15, check_layer=1):
    cfg = base.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // nh)
    nrep, scaling, L = nh // nkv, hd ** -0.5, cfg.num_hidden_layers
    N = doc_ids.shape[1]
    pos_all = torch.arange(N, device="cuda")

    h = base.model.embed_tokens(doc_ids)                              # (1, N, hid)
    cos, sin = base.model.rotary_emb(h, pos_all.unsqueeze(0))         # (1, N, hd)
    ck = [naive.key_cache[i].clone() for i in range(L)]
    cv = [naive.value_cache[i].clone() for i in range(L)]
    active = pos_all.clone()

    for i in range(L):
        layer = base.model.layers[i]; attn = layer.self_attn
        hn = layer.input_layernorm(h[:, active, :]); M = active.shape[0]
        q = attn.q_norm(attn.q_proj(hn).view(1, M, nh, hd)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hn).view(1, M, nkv, hd)).transpose(1, 2)
        v = attn.v_proj(hn).view(1, M, nkv, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:, active, :], sin[:, active, :])

        if i == check_layer:                                         # select HKVD, restrict
            diff = ((k - ck[i][:, :, active, :]).float() ** 2).sum(dim=[0, 1, 3])
            sel = torch.topk(diff, max(1, int(M * ratio))).indices
            active = active[sel]; q, k, v = q[:, :, sel, :], k[:, :, sel, :], v[:, :, sel, :]

        ck[i][:, :, active, :] = k; cv[i][:, :, active, :] = v       # write recomputed into cache
        kk, vv = repeat_kv(ck[i], nrep), repeat_kv(cv[i], nrep)      # attend to full (mixed) cache
        scores = (q @ kk.transpose(2, 3)) * scaling                 # (1, nh, M, N)
        mask = pos_all[None, :] <= active[:, None]                   # causal by absolute position
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        out = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) @ vv
        out = attn.o_proj(out.transpose(1, 2).reshape(1, active.shape[0], nh * hd))
        ha = h[:, active, :] + out
        h[:, active, :] = ha + layer.mlp(layer.post_attention_layernorm(ha))

    c = DynamicCache(); c.key_cache, c.value_cache = ck, cv
    return c


@torch.no_grad()
def blend_forward_c2c(model, base, doc_ids, naive_base, sharer_cache, ratio=0.15, check_layer=1):
    """C2C + CacheBlend (store-both design): recover cross-chunk attention in the RECEIVER base
    cache via CacheBlend, then fuse with the block-local sharer cache. The 4B never runs at query
    (its KV is precomputed); only the receiver partial-forward + the cheap fuser run online."""
    import kvcache as KV
    blended_base = blend_forward(base, doc_ids, naive_base, ratio, check_layer)
    return KV.fuse_caches(model, blended_base, sharer_cache)
