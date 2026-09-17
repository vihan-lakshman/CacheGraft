"""Side-by-side demo of what CacheGraft actually generates, on SQuAD (or any prepped dataset).

For each question it runs the SAME prompt tokens through:
  text     receiver (Qwen3-0.6B) reading the passage as text              -- the baseline
  cached   CacheGraft two-phase path: fused doc KV built offline, written to disk, loaded, and
           decoded by the base receiver alone (no 4B, no fuser on the query path)
  control  (--control) the identical cache pipeline with fusion off (receiver's own KV)
  sharer   (--sharer)  Qwen3-4B-Base reading the passage as text          -- the ceiling

`text` and `cached` see byte-identical token ids; the only difference is whose KV fills the
document section. Correctness is `contains` on the raw generation against the FULL alias list,
the same rule as report_table.py.

Two modes:

  # show specific questions (indices into <data>_data.json)
  python demo_squad.py --idx 3 17 42 --sharer --control

  # scan the first N questions, report the full win/loss breakdown, then show examples of each
  python demo_squad.py --scan 100 --show 3 --sharer --latex demo_example.tex

Scan mode prints BOTH directions (cached-only wins AND text-only wins) with counts, so an example
chosen for the paper can be described honestly as one of W wins vs L losses out of N.

Generations here are batch-1; the reported text-RAG cells were batched with left padding, so an
individual generation can differ from the one in <data>_results/ by decode noise.
"""
import os, json, re, time, argparse
os.environ.setdefault("USE_TF", "0")
import torch
import common as C
import kvcache as KV
from rag_trivia import make_content, normalize
from rag_trivia_cached import resolve_facts, DUMMY_Q

MAX_NEW = 24


def contains(gen, norm_aliases):
    g = normalize(gen)
    return any(a and a in g for a in norm_aliases)


def templated(tok, q, passage, terse):
    return tok.apply_chat_template([{"role": "user", "content": make_content(q, passage, terse=terse)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def gen_text(model, tok, full):
    enc = tok(full, return_tensors="pt").to("cuda")
    o = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def gen_cached(model, base, tok, passage, q, cache_dir, fuse, terse):
    """The two-phase path, exactly as rag_trivia_cached.py runs it. Returns (gen, info)."""
    # phase 1 (offline): the doc section is built from a placeholder question
    ids0, L0, _ = KV.split_doc(tok, templated(tok, DUMMY_Q, passage, terse))
    path = os.path.join(cache_dir, f"{KV.doc_key(ids0, L0)}.pt")
    build_s = None
    if not os.path.exists(path):
        t0 = time.time()
        KV.save_cache(KV.build_doc_cache(model, ids0, L0, fuse=fuse), L0, path)
        build_s = time.time() - t0
    # phase 2 (online): the real question, receiver only
    ids, L_doc, P = KV.split_doc(tok, templated(tok, q, passage, terse))
    cache, L_saved = KV.load_cache(path)
    assert L_saved == L_doc, f"cached doc section has {L_saved} tokens, query prompt has {L_doc}"
    t0 = time.time()
    g = KV.decode_with_cache(base, ids, L_doc, P, cache, tok.eos_token_id, max_new=MAX_NEW)
    torch.cuda.synchronize()
    info = {"doc_tokens": L_doc, "cache_mb": os.path.getsize(path) / 2**20,
            "build_s": build_s, "decode_ms": 1000 * (time.time() - t0)}
    return tok.decode(g, skip_special_tokens=True), info


def answer_snippet(passage, norm_aliases, max_words=45):
    """The passage sentence(s) holding the answer, for a compact paper example."""
    sents = re.split(r"(?<=[.!?])\s+", passage.strip())
    hits = [i for i, s in enumerate(sents) if any(a in normalize(s) for a in norm_aliases)]
    if not hits:
        words = passage.split()
        return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    s = sents[hits[0]]
    words = s.split()
    s = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    return ("... " if hits[0] > 0 else "") + s + (" ..." if hits[0] < len(sents) - 1 else "")


# ------------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------------
LABELS = {"text": "receiver text-RAG (0.6B)", "cached": "CacheGraft, persisted cache",
          "control": "receiver-own-KV control", "sharer": "sharer text-RAG (4B)"}


def show(ex, full_passage):
    bar = "=" * 96
    print(f"\n{bar}\n[{ex['idx']}] {ex['q']}")
    print(f"gold: {ex['aliases']}   passage: {'gold' if ex['hit'] else 'NOT gold'} (id {ex['fact_id']})")
    print("-" * 96)
    print(ex["passage"] if full_passage else ex["snippet"])
    print("-" * 96)
    for k, r in ex["systems"].items():
        mark = "OK " if r["ok"] else "-- "
        print(f"{mark}{LABELS[k]:<30} {r['gen'].strip()!r}")
    i = ex["systems"]["cached"]["info"]
    extra = f", built in {i['build_s']:.2f}s" if i["build_s"] is not None else ", reused from disk"
    print(f"   cache: {i['doc_tokens']} doc tokens, {i['cache_mb']:.1f} MB{extra}; "
          f"decode {i['decode_ms']:.0f} ms")


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in " ".join(s.split()))


def to_latex(exs, n_scanned=None, counts=None):
    """A single-column figure for the acmart paper (uses its \\sys macro)."""
    names = {"text": "Text-RAG", "cached": r"\sys{}", "control": "Own-KV", "sharer": "4B text"}
    mark = {True: r"\cmark", False: r"\xmark"}
    out = []
    for ex in exs:
        rows = "\n".join(
            f"{names[k]} & {tex_escape(r['gen'].strip())} & {mark[r['ok']]} \\\\"
            for k, r in ex["systems"].items())
        out.append(
            "\\begin{tabular}{@{}p{0.2\\linewidth}p{0.66\\linewidth}c@{}}\n\\toprule\n"
            f"\\multicolumn{{3}}{{@{{}}p{{0.95\\linewidth}}@{{}}}}{{\\emph{{Passage:}} "
            f"{tex_escape(ex['snippet'])}}} \\\\\n"
            f"\\multicolumn{{3}}{{@{{}}p{{0.95\\linewidth}}@{{}}}}{{\\emph{{Q:}} {tex_escape(ex['q'])} "
            f"\\quad \\emph{{Gold:}} {tex_escape(ex['aliases'][0])}}} \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}")
    note = ""
    if counts:
        note = (f" Over the first {n_scanned} questions, \\sys{{}} is correct where text-RAG is not "
                f"on {counts['cached_only']} and the reverse on {counts['text_only']}.")
    return ("% requires: \\usepackage{pifont} \\newcommand{\\cmark}{\\ding{51}} "
            "\\newcommand{\\xmark}{\\ding{55}}\n"
            "\\begin{figure}[t]\n\\centering\\scriptsize\n"
            + "\n\\par\\medskip\n".join(out)
            + "\n\\caption{The same receiver and the same prompt tokens; only the document-section "
              "KV differs. \\sys{} output is from the persisted cache, decoded with no sharer on "
              f"the query path. Correctness is gold-alias containment.{note}}}\n"
              "\\label{fig:example}\n\\end{figure}\n")


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="squad")
    ap.add_argument("--fact", default="retrieved", choices=["retrieved", "oracle"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int, nargs="+", help="question indices to show")
    g.add_argument("--scan", type=int, help="compare text vs cached over the first N questions")
    ap.add_argument("--show", type=int, default=3, help="scan mode: examples to show per bucket")
    ap.add_argument("--sharer", action="store_true", help="also run the 4B text-RAG ceiling")
    ap.add_argument("--control", action="store_true", help="also run the receiver-own-KV control")
    ap.add_argument("--terse", action="store_true", help="matched short-answer prompt")
    ap.add_argument("--full_passage", action="store_true", help="print the whole passage")
    ap.add_argument("--json", default=None, help="write all examples here")
    ap.add_argument("--latex", default=None, help="write a paper figure for the shown examples")
    args = ap.parse_args()
    C.set_determinism()

    P = C.paths(args.data)
    if not os.path.exists(P["data"]):
        raise SystemExit(f"missing {P['data']}: run `python prep.py --data {args.data} --target 800`")
    data = json.load(open(P["data"]))
    corpus, allq = data["corpus"], data["questions"]
    idx = args.idx if args.idx else list(range(min(args.scan, len(allq))))
    qs = [allq[i] for i in idx]
    fact_ids, _ = resolve_facts(qs, args.fact, P["emb"])

    model, base, tok = KV.load_c2c(max_new=MAX_NEW)
    fused_dir, own_dir = P["kvcache"], P["kvcache"] + "_recv"
    os.makedirs(fused_dir, exist_ok=True)
    if args.control:
        os.makedirs(own_dir, exist_ok=True)

    # pass 1: the two systems every example needs
    exs = []
    for n, (i, q, fid) in enumerate(zip(idx, qs, fact_ids)):
        passage = corpus[fid]
        full = templated(tok, q["q"], passage, args.terse)
        t = gen_text(base, tok, full)
        c, info = gen_cached(model, base, tok, passage, q["q"], fused_dir, True, args.terse)
        exs.append({"idx": i, "q": q["q"], "aliases": q["aliases"], "fact_id": fid,
                    "hit": fid in q["gold_ids"], "passage": passage,
                    "snippet": answer_snippet(passage, q["norm_aliases"]),
                    "systems": {"text": {"gen": t, "ok": contains(t, q["norm_aliases"])},
                                "cached": {"gen": c, "ok": contains(c, q["norm_aliases"]),
                                           "info": info}}})
        if args.scan and (n + 1) % 25 == 0:
            print(f"  scanned {n + 1}/{len(idx)}", flush=True)

    # choose what to display
    counts = None
    if args.scan:
        b = {"cached_only": [], "text_only": [], "both": [], "neither": []}
        for e in exs:
            t, c = e["systems"]["text"]["ok"], e["systems"]["cached"]["ok"]
            b["both" if t and c else "cached_only" if c else "text_only" if t else "neither"].append(e)
        counts = {k: len(v) for k, v in b.items()}
        shown = b["cached_only"][:args.show] + b["text_only"][:args.show]
    else:
        shown = exs

    # pass 2: the optional systems, only for the examples we display
    if args.control:
        for e in shown:
            g_, _ = gen_cached(model, base, tok, e["passage"], e["q"], own_dir, False, args.terse)
            e["systems"]["control"] = {"gen": g_, "ok": contains(g_, allq[e["idx"]]["norm_aliases"])}
    if args.sharer and shown:
        del model, base; torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        stok = AutoTokenizer.from_pretrained(C.SHARER_BASE)
        sm = AutoModelForCausalLM.from_pretrained(C.SHARER_BASE, torch_dtype=torch.bfloat16,
                                                  device_map="cuda").eval()
        for e in shown:
            g_ = gen_text(sm, stok, templated(stok, e["q"], e["passage"], args.terse))
            e["systems"]["sharer"] = {"gen": g_, "ok": contains(g_, allq[e["idx"]]["norm_aliases"])}

    for e in shown:
        show(e, args.full_passage)

    if counts:
        n = len(exs)
        print(f"\n{'=' * 96}\n{args.data}/{args.fact}, first {n} questions (contains):")
        print(f"  text-RAG {sum(e['systems']['text']['ok'] for e in exs) / n:.1%}   "
              f"CacheGraft {sum(e['systems']['cached']['ok'] for e in exs) / n:.1%}")
        print(f"  both right {counts['both']}  |  CacheGraft only {counts['cached_only']}  |  "
              f"text only {counts['text_only']}  |  neither {counts['neither']}")
        print("  shown above: the first examples from each of the two disagreement buckets")

    if args.json:
        json.dump({"data": args.data, "fact": args.fact, "terse": args.terse, "counts": counts,
                   "examples": exs}, open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(to_latex(shown, len(exs) if counts else None, counts))
        print(f"wrote {args.latex}")


if __name__ == "__main__":
    main()
