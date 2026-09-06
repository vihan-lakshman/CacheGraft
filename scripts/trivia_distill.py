"""Offline sharer-distillation of corpus passages (question-agnostic).

The cost-matched TEXT-channel counterpart to doc_only fusion. Instead of fusing the sharer's
KV-cache of a passage, we run the SAME sharer (Qwen3-4B-Base by default) over each passage
ONCE, offline, and store a short textual summary. At query time the receiver reads that summary
in place of the raw passage -- so, exactly like doc_only fusion, the sharer never runs at
inference and never sees the question. The ONLY thing that differs from C2C is the channel:
text here vs KV there. It is the strongest offline-precomputable *text* competitor, and it
costs ~1e-5 the storage of a fused KV cache, so it is the baseline the systems claim must beat.

We deliberately use the SAME 4B-Base as the C2C sharer (not an instruct model) so the
comparison isolates the channel, not the model. Base models follow a 1-shot completion format
more reliably than a chat instruction, so we prompt by example and cut at the next "Passage:".
(Pass --model Qwen/Qwen3-4B-Instruct for an upper-bound text baseline -- but note that is no
longer the same model the fuser was built on, so it stops being a clean channel comparison.)

  python trivia_distill.py --data squad            # distill the passages squad actually uses
  python trivia_distill.py --data trivia --subset all    # distill the whole corpus

Writes `<data>_distilled.json`: a list aligned 1:1 with the corpus, with `null` for passages
that were not distilled (only the retrieved/oracle passages are needed, and distilling 20k
unused chunks costs hours).
"""
import os, json, argparse, time
os.environ.setdefault("USE_TF", "0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import common as C

HERE = os.path.dirname(__file__)

# 1-shot exemplar: teaches the base model the format and gives it a place to stop.
SHOT = ("Passage: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was "
        "completed in 1889 and stands 330 metres tall. It was designed by the engineering "
        "company of Gustave Eiffel.\n"
        "Key facts: The Eiffel Tower is a wrought-iron tower in Paris, France. It was completed "
        "in 1889 and is 330 metres tall. It was designed by Gustave Eiffel's company.\n\n")

# A deliberately STRONGER text-channel prompt. A query-agnostic summary must serve every future
# question about the passage, so a summarizer that drops named entities, dates and numbers is
# throwing away exactly the spans extractive QA asks for. This variant instructs the distiller to
# preserve them, and exists so the distilled baseline is beaten at its best, not at its default.
SHOT_DETAILED = (
    "Passage: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was "
    "completed in 1889 and stands 330 metres tall. It was designed by the engineering "
    "company of Gustave Eiffel.\n"
    "All facts (keep every name, date, number and title): The Eiffel Tower is a wrought-iron "
    "lattice tower located in Paris, France. It was completed in 1889. It stands 330 metres "
    "tall. It was designed by the engineering company of Gustave Eiffel.\n\n")

def build(doc, style="facts"):
    if style == "detailed":
        return SHOT_DETAILED + (f"Passage: {doc}\nAll facts (keep every name, date, number "
                                f"and title):")
    return SHOT + f"Passage: {doc}\nKey facts:"

def clean(cont):
    # keep only the generated "Key facts" continuation; stop at the next exemplar boundary
    for stop in ("\nPassage:", "\n\n", "\nKey facts:", "\nAll facts"):
        j = cont.find(stop)
        if j != -1:
            cont = cont[:j]
    cont = " ".join(cont.split()).strip()
    # Trim a max_new-truncated tail back to the last complete sentence. Without this ~1/3 of
    # summaries end mid-clause, which would handicap the baseline for a decoding-budget reason
    # rather than a channel reason. Only applied when it leaves most of the summary intact.
    if cont and cont[-1] not in ".!?\"'":
        cut = max(cont.rfind(p) for p in (". ", "! ", "? "))
        if cut > 0.6 * len(cont):
            cont = cont[:cut + 1]
    return cont.strip()


def used_ids(data, paths, n_test):
    """Corpus ids the eval actually touches: top-1 retrieved + oracle, for the first n_test
    questions. Retrieval is recomputed with the SAME embedder/query prefix as rag_trivia.py, so
    the ids match exactly."""
    qs = data["questions"][:n_test]
    ids = {q["gold_ids"][0] for q in qs}
    from sentence_transformers import SentenceTransformer
    cemb = np.load(paths["emb"])
    em = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    qemb = em.encode(["Represent this sentence for searching relevant passages: " + q["q"]
                      for q in qs], normalize_embeddings=True, batch_size=128)
    ids |= set(C.top1(qemb, cemb, qs))
    del em; torch.cuda.empty_cache()
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="trivia", help="dataset name (see common.paths)")
    ap.add_argument("--model", default=C.SHARER_BASE, help="distiller (default = the C2C sharer)")
    ap.add_argument("--style", default="facts", choices=["facts", "detailed"],
                    help="'facts' = terse key-facts summary (the cost-matched default); "
                         "'detailed' = entity/date/number-preserving prompt (a deliberately "
                         "stronger text-channel competitor, for the robustness check)")
    ap.add_argument("--tag", default=None,
                    help="output suffix -> <data>_distilled<tag>.json (default: none)")
    ap.add_argument("--subset", default="used", choices=["used", "all"],
                    help="'used' = only the retrieved/oracle passages of the first --n_test "
                         "questions (what the eval reads); 'all' = the whole corpus")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=96)
    args = ap.parse_args()

    P = C.paths(args.data)
    data = json.load(open(P["data"]))
    corpus = data["corpus"]
    targets = list(range(len(corpus))) if args.subset == "all" else used_ids(data, P, args.n_test)
    print(f"distilling {len(targets)}/{len(corpus)} passages with {args.model} "
          f"(subset={args.subset})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    out = [None] * len(corpus)
    t0, done = time.time(), 0
    for i in range(0, len(targets), args.batch):
        ids = targets[i:i + args.batch]
        prompts = [build(corpus[j], args.style) for j in ids]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new,
                               pad_token_id=tok.pad_token_id)
        cont = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # fall back to the raw passage if the model emitted nothing usable, so the baseline
        # never gets an empty context (which would unfairly tank it).
        for j, c in zip(ids, cont):
            out[j] = clean(c) or corpus[j]
        done += len(ids)
        if (i // args.batch) % 5 == 0:
            print(f"  {done}/{len(targets)}  ({time.time() - t0:.0f}s)", flush=True)

    path = f"{P['distill_prefix']}{args.tag or ''}.json"
    json.dump(out, open(path, "w"))
    n_words = [len(s.split()) for s in out if s]
    print(f"saved {os.path.basename(path)}  ({done} summaries, {time.time() - t0:.0f}s, "
          f"avg {np.mean(n_words):.0f} words vs passage "
          f"{np.mean([len(corpus[j].split()) for j in targets]):.0f})")
    for k in targets[:1] + targets[len(targets) // 2:len(targets) // 2 + 1]:  # eyeball a couple
        print(f"\n[passage] {corpus[k][:200]}...\n[summary] {out[k]}")

if __name__ == "__main__":
    main()
