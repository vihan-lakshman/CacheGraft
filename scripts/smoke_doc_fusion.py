"""Smoke-test doc-only fusion: fuse ONLY the document tokens, question stays un-fused.
Confirms the wrapper honors a per-position (mixed) kv_cache_index.
"""
import os, torch
os.environ.setdefault("USE_TF", "0")
from rosetta.utils.evaluate import load_rosetta_model
import common as C

mc = {"model_name": "Rosetta", "rosetta_config": {
    "base_model": C.RECEIVER, "teacher_model": C.SHARER_BASE,
    "is_do_alignment": False, "alignment_strategy": "first", "checkpoints_dir": C.get_fuser_dir()}}
model, tok = load_rosetta_model(mc, {}, device=torch.device("cuda"),
                                generation_config={"do_sample": False, "max_new_tokens": 24})

doc = ("Sunset Boulevard is a musical with music by Andrew Lloyd Webber. "
       "It premiered in the United States in Los Angeles on 9 December 1993, "
       "and opened on Broadway in November 1994.")
question = "Which Lloyd Webber musical premiered in the US in December 1993?"
content = (f"Answer the question using the context. Give only a short answer.\n\n"
           f"Context:\n{doc}\n\nQuestion: {question}")
full = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
offsets = enc.pop("offset_mapping")[0].tolist()
enc = {k: v.to("cuda") for k, v in enc.items()}
P = enc["input_ids"].shape[1]
# NOTE: RosettaModel applies fusion PER SECTION (keyed on each section's first token),
# NOT per token. So doc-only fusion needs THREE sections: [doc=fuse, question=no-fuse, gen].
c_q = full.rfind("Question:")
L_doc = max(1, min(sum(1 for i in range(P) if 0 < offsets[i][1] <= c_q), P - 2))
print(f"prompt tokens={P}  fused(doc) tokens={L_doc}  unfused(question+gen)={P-L_doc}")

def sec(mask, n):
    return torch.tensor([[mask, 0]] * n).unsqueeze(0).cuda()

def gen(mode):
    if mode == "none":
        idx = [sec(-1, P - 1), sec(-1, 1)]
    elif mode == "doc_q":          # fuse the whole [doc+question] prompt
        idx = [sec(1, P - 1), sec(-1, 1)]
    else:                          # doc_only: fuse ONLY the document section
        idx = [sec(1, L_doc), sec(-1, (P - 1) - L_doc), sec(-1, 1)]
    with torch.no_grad():
        o = model.generate(**enc, kv_cache_index=idx, do_sample=False, max_new_tokens=24)
    return tok.decode(o[0, P:], skip_special_tokens=True).strip()

for m in ["none", "doc_only", "doc_q"]:
    print(f"[{m:8}] -> {gen(m)!r}")
