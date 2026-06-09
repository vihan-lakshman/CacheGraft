"""Offline 4B distillation of every corpus passage (question-agnostic).

The cost-matched TEXT-channel counterpart to doc_only fusion. Instead of fusing the 4B's
KV-cache of a passage, we run the SAME 4B (Qwen3-4B-Base) over each passage ONCE, offline,
and store a short textual summary. At query time the receiver reads that summary in place of
the raw passage -- so, exactly like doc_only fusion, the 4B never runs at inference and never
sees the question. The ONLY thing that differs from C2C is the channel: text here vs KV there.

We deliberately use the SAME 4B-Base as the C2C sharer (not an instruct model) so the
comparison isolates the channel, not the model. Base models follow a 1-shot completion format
more reliably than a chat instruction, so we prompt by example and cut at the next "Passage:".
(Pass --model Qwen/Qwen3-4B-Instruct for an upper-bound text baseline -- but note that is no
longer the same model the fuser was built on, so it stops being a clean channel comparison.)

  python trivia_distill.py        # distill all corpus chunks -> trivia_distilled.json
"""
import os, json, argparse, time
os.environ.setdefault("USE_TF", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import common as C

HERE = os.path.dirname(__file__)

# 1-shot exemplar: teaches the base model the format and gives it a place to stop.
SHOT = ("Passage: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was "
        "completed in 1889 and stands 330 metres tall. It was designed by the engineering "
        "company of Gustave Eiffel.\n"
        "Key facts: The Eiffel Tower is a wrought-iron tower in Paris, France. It was completed "
        "in 1889 and is 330 metres tall. It was designed by Gustave Eiffel's company.\n\n")

def build(doc):
    return SHOT + f"Passage: {doc}\nKey facts:"

def clean(cont):
    # keep only the generated "Key facts" continuation; stop at the next exemplar boundary
    for stop in ("\nPassage:", "\n\n", "\nKey facts:"):
        j = cont.find(stop)
        if j != -1:
            cont = cont[:j]
    return " ".join(cont.split()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.SHARER_BASE, help="distiller (default = the C2C sharer)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=96)
    args = ap.parse_args()

    corpus = json.load(open(os.path.join(HERE, "trivia_data.json")))["corpus"]
    print(f"distilling {len(corpus)} passages with {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    summaries, t0 = [], time.time()
    for i in range(0, len(corpus), args.batch):
        prompts = [build(d) for d in corpus[i:i + args.batch]]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new,
                               pad_token_id=tok.pad_token_id)
        cont = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # fall back to the raw passage if the model emitted nothing usable, so the baseline
        # never gets an empty context (which would unfairly tank it).
        summaries += [clean(c) or corpus[i + j] for j, c in enumerate(cont)]
        if (i // args.batch) % 10 == 0:
            print(f"  {len(summaries)}/{len(corpus)}  ({time.time() - t0:.0f}s)", flush=True)

    json.dump(summaries, open(os.path.join(HERE, "trivia_distilled.json"), "w"))
    print(f"saved trivia_distilled.json  ({len(summaries)} summaries, {time.time() - t0:.0f}s)")
    for k in (0, len(summaries) // 2):  # eyeball a couple
        print(f"\n[passage] {corpus[k][:160]}...\n[summary] {summaries[k]}")

if __name__ == "__main__":
    main()
