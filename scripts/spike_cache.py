"""SPIKE: prove the offline-cacheable doc-only path == the inline doc_only path.

Build: prefill [preamble+doc] with fusion (doc as a NON-final section so fusion applies),
capture the receiver's fused past_key_values, truncate the 1 dummy token -> the fused doc cache.
Query: inject that cache into the BASE receiver (model_list[0]) and decode the question with NO
fuser / NO 4B. Compare to the inline run_c2c generation token-for-token.
"""
import os, sys
os.environ.setdefault("USE_TF", "0")
import torch
from transformers.cache_utils import DynamicCache
import common as C

SHARER_BASE = C.SHARER_BASE

def make_content(q, passage):
    return ("Answer the question using the context. Give only a short factual answer "
            f"(no explanation).\n\nContext:\n{passage}\n\nQuestion: {q}")

def doc_split(tok, full):
    """Return (full_ids[1,P], L_doc) using the same rule as run_c2c (offset of last 'Question:')."""
    enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"]
    P = ids.shape[1]
    c_q = full.rfind("Question:")
    L_doc = sum(1 for i in range(P) if 0 < offs[i][1] <= c_q)
    L_doc = max(1, min(L_doc, P - 2))
    return ids, L_doc, P

def sec(mask, n):
    return torch.tensor([[mask, 0]] * n).unsqueeze(0).cuda()

def build_doc_cache(model, ids, L_doc):
    """Fused KV of [0:L_doc], built with a dummy non-final token then truncated."""
    inp = ids[:, :L_doc + 1].cuda()                       # doc + 1 dummy
    idx = [sec(1, L_doc), sec(-1, 1)]                      # doc fused, dummy no-fuse (final)
    out = model.forward(kv_cache_index=idx, input_ids=inp,
                        attention_mask=torch.ones_like(inp), use_cache=True)
    pkv = out.past_key_values
    cache = DynamicCache()
    for l in range(len(pkv.key_cache)):
        cache.key_cache.append(pkv.key_cache[l][:, :, :L_doc, :].contiguous())
        cache.value_cache.append(pkv.value_cache[l][:, :, :L_doc, :].contiguous())
    return cache

def query_with_cache(base, ids, L_doc, P, cache, eos_id, max_new=24):
    """Manual prefill + greedy decode against the cached doc KV (base receiver only)."""
    cont = ids[:, L_doc:P].cuda()                          # "Question: ..." + gen prompt
    pos = torch.arange(L_doc, P, device="cuda").unsqueeze(0)
    attn = torch.ones((1, P), dtype=torch.long, device="cuda")
    out = base(input_ids=cont, attention_mask=attn, position_ids=pos,
               past_key_values=cache, use_cache=True)
    past, logits, cur = out.past_key_values, out.logits[:, -1, :], P
    gen = []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        if nxt == eos_id:
            break
        gen.append(nxt)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.long, device="cuda")], 1)
        out = base(input_ids=torch.tensor([[nxt]], device="cuda"), attention_mask=attn,
                   position_ids=torch.tensor([[cur]], device="cuda"),
                   past_key_values=past, use_cache=True)
        past, logits, cur = out.past_key_values, out.logits[:, -1, :], cur + 1
    return torch.tensor(gen)

def main():
    from rosetta.utils.evaluate import load_rosetta_model
    import json
    mc = {"model_name": "Rosetta", "rosetta_config": {
        "base_model": C.RECEIVER, "teacher_model": SHARER_BASE,
        "is_do_alignment": False, "alignment_strategy": "first", "checkpoints_dir": C.get_fuser_dir()}}
    model, tok = load_rosetta_model(mc, {}, device=torch.device("cuda"),
                                    generation_config={"do_sample": False, "max_new_tokens": 24})
    base = model.model_list[0]
    data = json.load(open(os.path.join(os.path.dirname(__file__), "trivia_data.json")))
    qs = data["questions"][:6]; corpus = data["corpus"]
    n_match = 0
    for q in qs:
        passage = corpus[q["gold_ids"][0]]                # oracle doc for a clean test
        content = make_content(q["q"], passage)
        full = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        ids, L_doc, P = doc_split(tok, full)
        # inline doc_only (reference)
        idx = [sec(1, L_doc), sec(-1, (P - 1) - L_doc), sec(-1, 1)]
        o_inline = model.generate(input_ids=ids.cuda(), kv_cache_index=idx,
                                  do_sample=False, max_new_tokens=24)[0, P:]
        # cached path
        cache = build_doc_cache(model, ids, L_doc)
        o_cached = query_with_cache(base, ids, L_doc, P, cache, tok.eos_token_id)
        s_inline = tok.decode(o_inline, skip_special_tokens=True).strip()
        s_cached = tok.decode(o_cached, skip_special_tokens=True).strip()
        match = s_inline == s_cached
        n_match += match
        print(f"[{'OK ' if match else 'DIFF'}] inline={s_inline[:60]!r}\n       cached={s_cached[:60]!r}")
    print(f"\n{n_match}/{len(qs)} exact-match (cached vs inline)")

if __name__ == "__main__":
    main()
