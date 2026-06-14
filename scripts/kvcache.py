"""Offline-cacheable doc-only C2C fusion.

The doc-only fused KV of [preamble+passage] is query-independent, so it can be built ONCE
offline (the 4B sharer + fuser run here) and reused at query time by the base receiver alone
(no 4B, no fuser). This module is the hardened artifact behind that claim:

  build_doc_cache : prefill [preamble+passage] with fusion -> fused receiver KV (DynamicCache)
  save_cache/load_cache : persist/restore a fused doc cache to/from disk
  decode_with_cache : inject a cached doc KV into the base receiver and decode the question

Validated to reproduce the inline doc_only path up to greedy floating-point nondeterminism
(see spike_cache.py / rag_trivia_cached.py).
"""
import os, hashlib
os.environ.setdefault("USE_TF", "0")
import torch
from transformers.cache_utils import DynamicCache
import common as C


def load_c2c(receiver=C.RECEIVER, sharer=C.SHARER_BASE, fuser_dir=None, max_new=24):
    """Load the RosettaModel; return (rosetta_model, base_receiver_model, tokenizer)."""
    from rosetta.utils.evaluate import load_rosetta_model
    mc = {"model_name": "Rosetta", "rosetta_config": {
        "base_model": receiver, "teacher_model": sharer,
        "is_do_alignment": False, "alignment_strategy": "first",
        "checkpoints_dir": fuser_dir or C.get_fuser_dir()}}
    model, tok = load_rosetta_model(mc, {}, device=torch.device("cuda"),
                                    generation_config={"do_sample": False, "max_new_tokens": max_new})
    return model, model.model_list[0], tok


def split_doc(tok, full):
    """Tokenize a templated prompt; split at the last 'Question:' (same rule as run_c2c).
    Returns (ids[1,P], L_doc, P). Tokens [0:L_doc] are the query-independent doc section."""
    enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"]
    P = ids.shape[1]
    c_q = full.rfind("Question:")
    L_doc = max(1, min(sum(1 for i in range(P) if 0 < offs[i][1] <= c_q), P - 2))
    return ids, L_doc, P


def doc_key(ids, L_doc):
    """Stable content hash of the doc section (keys the on-disk cache; preamble-text aware)."""
    toks = ids[0, :L_doc].tolist()
    return hashlib.sha1((",".join(map(str, toks))).encode()).hexdigest()[:16]


def _sec(mask, n):
    return torch.tensor([[mask, 0]] * n).unsqueeze(0).cuda()


@torch.no_grad()
def build_doc_cache(model, ids, L_doc, fuse=True):
    """Receiver KV of tokens [0:L_doc]. Built with a 1-token dummy so the doc is a NON-final
    section (fusion only applies to non-final sections), then truncated.
    fuse=True : doc section fused with the sharer (mask 1) -> the C2C cache.
    fuse=False: doc section NOT fused (mask -1) -> the receiver's OWN KV, through the identical
                pipeline. A control: predicts equivalence to text_rag (gain must come from fusion)."""
    inp = ids[:, :L_doc + 1].cuda()
    doc_sec = _sec(1, L_doc) if fuse else _sec(-1, L_doc)
    out = model.forward(kv_cache_index=[doc_sec, _sec(-1, 1)], input_ids=inp,
                        attention_mask=torch.ones_like(inp), use_cache=True)
    pkv = out.past_key_values
    cache = DynamicCache()
    for l in range(len(pkv.key_cache)):
        cache.key_cache.append(pkv.key_cache[l][:, :, :L_doc, :].contiguous())
        cache.value_cache.append(pkv.value_cache[l][:, :, :L_doc, :].contiguous())
    return cache


def save_cache(cache, L_doc, path):
    torch.save({"L_doc": L_doc,
                "key": [k.to(torch.float16).cpu() for k in cache.key_cache],
                "value": [v.to(torch.float16).cpu() for v in cache.value_cache]}, path)


def load_cache(path, device="cuda", dtype=torch.bfloat16):
    d = torch.load(path, map_location=device)
    cache = DynamicCache()
    cache.key_cache = [k.to(dtype) for k in d["key"]]
    cache.value_cache = [v.to(dtype) for v in d["value"]]
    return cache, d["L_doc"]


# ---------------------------------------------------------------------------------------------
# Slot-based top-k composition (precompute + prefill only; NO fusion at query time).
# Each chunk's fused KV is built at the FIXED slot position it will occupy (fused keys are not
# RoPE-re-rotatable, so we bake the position in at build time -> exact, k x storage). At query we
# concatenate [preamble | block@slot1 | ... | block@slotk] and prefill the question. Block-local:
# chunks attend only to themselves (not to each other), which is inherent to any cacheable scheme.
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def build_block_cache(model, block_ids, slot_start):
    """Fused receiver KV of a standalone chunk placed at absolute positions [slot_start:+L].
    Keys are RoPE-rotated for the slot position (baked in). Built per (chunk, slot)."""
    L = block_ids.shape[1]
    inp = torch.cat([block_ids, block_ids[:, :1]], dim=1).cuda()          # +1 dummy (non-final sec)
    pos = torch.arange(slot_start, slot_start + L + 1, device="cuda").unsqueeze(0)
    out = model.forward(kv_cache_index=[_sec(1, L), _sec(-1, 1)], input_ids=inp,
                        attention_mask=torch.ones_like(inp), position_ids=pos, use_cache=True)
    pkv = out.past_key_values
    cache = DynamicCache()
    for l in range(len(pkv.key_cache)):
        cache.key_cache.append(pkv.key_cache[l][:, :, :L, :].contiguous())
        cache.value_cache.append(pkv.value_cache[l][:, :, :L, :].contiguous())
    return cache

def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

def re_rotate_cache(cache, delta, inv_freq):
    """Shift a cached block's absolute positions by `delta` via RoPE composition (R(delta)*key).
    Valid for VANILLA (pre-fusion) keys -> one canonical copy serves any slot. Values untouched."""
    if delta == 0:
        return cache
    freqs = float(delta) * inv_freq                       # (D/2,)
    emb = torch.cat([freqs, freqs])                       # (D,)
    cos, sin = emb.cos(), emb.sin()
    out = DynamicCache()
    for l in range(len(cache.key_cache)):
        k = cache.key_cache[l]
        c = cos.to(k.dtype); s = sin.to(k.dtype)
        out.key_cache.append(k * c + _rotate_half(k) * s)
        out.value_cache.append(cache.value_cache[l])
    return out

@torch.no_grad()
def fuse_caches(model, base_kv, sharer_kv, base_idx=0, source_idx=1):
    """Run ONLY the fuser/projector on cached pre-fusion KVs (no model forward), mirroring
    RosettaModel.forward's parallel single-source fusion. Lets the fuser run at query time while
    the 4B stays offline. base_kv = receiver KV, sharer_kv = 4B KV (both at the same positions)."""
    from rosetta.model.wrapper import clone_kv_cache
    fused = clone_kv_cache(base_kv)
    for target_layer, pair_list in model.projector_dict[base_idx][source_idx].items():
        base_pair = (base_kv.key_cache[target_layer], base_kv.value_cache[target_layer])
        src_layer, proj_idx = pair_list[0]
        src_pair = (sharer_kv.key_cache[src_layer], sharer_kv.value_cache[src_layer])
        pk, pv = model.projector_list[proj_idx].forward(src_pair, base_pair)
        dt = base_kv.key_cache[target_layer].dtype                # match cache dtype (bf16)
        fused.key_cache[target_layer] = pk.to(dt)
        fused.value_cache[target_layer] = pv.to(dt)
    return fused

@torch.no_grad()
def receiver_kv_prefixed(base, pre_ids, block_ids):
    """Plain RECEIVER KV of a chunk, conditioned on the preamble (no 4B, no fuser). Returns the
    chunk slice [P:P+L]. For the receiver-only block-composition control (block-attention text-RAG)."""
    P, L = pre_ids.shape[1], block_ids.shape[1]
    full = torch.cat([pre_ids, block_ids], 1).cuda()
    out = base(input_ids=full, attention_mask=torch.ones_like(full),
               position_ids=torch.arange(P + L, device="cuda").unsqueeze(0), use_cache=True)
    pkv = out.past_key_values
    c = DynamicCache()
    for l in range(len(pkv.key_cache)):
        c.key_cache.append(pkv.key_cache[l][:, :, P:P + L, :].contiguous())
        c.value_cache.append(pkv.value_cache[l][:, :, P:P + L, :].contiguous())
    return c

@torch.no_grad()
def build_prefusion_prefixed(model, pre_ids, block_ids):
    """Like build_prefusion but the 4B sees [preamble + chunk] (chunk conditioned on the
    instruction, matching the validated monolithic encoding). Returns the PRE-fusion base/sharer
    KV for the CHUNK slice only (canonical chunk position = right after the preamble, pos P)."""
    P, L = pre_ids.shape[1], block_ids.shape[1]
    inp = torch.cat([pre_ids, block_ids, block_ids[:, :1]], 1).cuda()
    model.forward(kv_cache_index=[_sec(1, P + L), _sec(-1, 1)], input_ids=inp,
                  attention_mask=torch.ones_like(inp), use_cache=True)
    def slc(c):
        out = DynamicCache()
        for l in range(len(c.key_cache)):
            out.key_cache.append(c.key_cache[l][:, :, P:P + L, :].contiguous())
            out.value_cache.append(c.value_cache[l][:, :, P:P + L, :].contiguous())
        return out
    return slc(model.kv_cache_dict[0][0]), slc(model.kv_cache_dict[0][1])

@torch.no_grad()
def build_block_prefixed(model, pre_ids, block_ids, slot_start):
    """Fused chunk KV at [slot_start:+L], built with the preamble (at 0:P) as context across the
    position gap. The exact slot-based counterpart of build_prefusion_prefixed (+re-rotate)."""
    P, L = pre_ids.shape[1], block_ids.shape[1]
    inp = torch.cat([pre_ids, block_ids, block_ids[:, :1]], 1).cuda()
    pos = torch.cat([torch.arange(P), torch.arange(slot_start, slot_start + L + 1)]).unsqueeze(0).cuda()
    out = model.forward(kv_cache_index=[_sec(1, P + L), _sec(-1, 1)], input_ids=inp,
                        attention_mask=torch.ones_like(inp), position_ids=pos, use_cache=True)
    pkv = out.past_key_values
    cache = DynamicCache()
    for l in range(len(pkv.key_cache)):
        cache.key_cache.append(pkv.key_cache[l][:, :, P:P + L, :].contiguous())
        cache.value_cache.append(pkv.value_cache[l][:, :, P:P + L, :].contiguous())
    return cache

@torch.no_grad()
def build_prefusion(model, block_ids):
    """Run the fused forward over a chunk at canonical pos 0 but return the PRE-fusion caches
    (receiver base KV, 4B sharer KV) instead of the fused result. For the re-rotation top-k path:
    cache these once per chunk (O(1)), then at query re-rotate + fuse_caches at any slot."""
    L = block_ids.shape[1]
    inp = torch.cat([block_ids, block_ids[:, :1]], 1).cuda()
    model.forward(kv_cache_index=[_sec(1, L), _sec(-1, 1)], input_ids=inp,
                  attention_mask=torch.ones_like(inp), use_cache=True)
    from rosetta.model.wrapper import clone_kv_cache
    return clone_kv_cache(model.kv_cache_dict[0][0]), clone_kv_cache(model.kv_cache_dict[0][1])

def cat_caches(caches):
    """Concatenate DynamicCaches along the sequence dim (positions are already baked into keys)."""
    out = DynamicCache()
    for l in range(len(caches[0].key_cache)):
        out.key_cache.append(torch.cat([c.key_cache[l] for c in caches], dim=2))
        out.value_cache.append(torch.cat([c.value_cache[l] for c in caches], dim=2))
    return out

@torch.no_grad()
def prefill_plain(base, ids, start_pos, past=None):
    """Plain (un-fused) receiver prefill of `ids` at positions [start_pos:], over optional past.
    Returns (past, last_logits, cache_len). Used for the fixed preamble and the query question."""
    L = ids.shape[1]
    past_len = 0 if past is None else past.get_seq_length()
    pos = torch.arange(start_pos, start_pos + L, device="cuda").unsqueeze(0)
    attn = torch.ones((1, past_len + L), dtype=torch.long, device="cuda")
    out = base(input_ids=ids.cuda(), attention_mask=attn, position_ids=pos,
               past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :], past_len + L

@torch.no_grad()
def decode_greedy(base, past, last_logits, cache_len, next_pos, eos_id, max_new=24):
    """Greedy decode continuing from a prefilled (gapped-position) composed cache."""
    gen = []
    for _ in range(max_new):
        nxt = int(last_logits.argmax(-1))
        if nxt == eos_id:
            break
        gen.append(nxt)
        attn = torch.ones((1, cache_len + 1), dtype=torch.long, device="cuda")
        out = base(input_ids=torch.tensor([[nxt]], device="cuda"), attention_mask=attn,
                   position_ids=torch.tensor([[next_pos]], device="cuda"),
                   past_key_values=past, use_cache=True)
        past, last_logits, cache_len, next_pos = out.past_key_values, out.logits[:, -1, :], cache_len + 1, next_pos + 1
    return gen

@torch.no_grad()
def decode_with_cache(base, ids, L_doc, P, cache, eos_id, max_new=24):
    """Inject a cached doc KV into the base receiver, prefill the question, greedy-decode.
    No 4B / no fuser runs here -- this is the online half of the cacheable design."""
    cont = ids[:, L_doc:P].cuda()
    out = base(input_ids=cont, attention_mask=torch.ones((1, P), dtype=torch.long, device="cuda"),
               position_ids=torch.arange(L_doc, P, device="cuda").unsqueeze(0),
               past_key_values=cache, use_cache=True)
    past, logits, cur, gen = out.past_key_values, out.logits[:, -1, :], P, []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        if nxt == eos_id:
            break
        gen.append(nxt)
        out = base(input_ids=torch.tensor([[nxt]], device="cuda"),
                   attention_mask=torch.ones((1, cur + 1), dtype=torch.long, device="cuda"),
                   position_ids=torch.tensor([[cur]], device="cuda"),
                   past_key_values=past, use_cache=True)
        past, logits, cur = out.past_key_values, out.logits[:, -1, :], cur + 1
    return gen
