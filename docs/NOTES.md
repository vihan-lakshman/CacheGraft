# Project notes / session handoff

Context to resume this work on a fresh machine. (The README has the polished reproduction
instructions; this file is the "what we did, why, and what's next" log.)

## Goal

Take Cache-to-Cache (C2C) "beyond the paper" into **retrieval-augmented generation**: fuse a
large sharer model's KV-cache *representation* of retrieved context into a small receiver model,
instead of feeding the receiver the retrieved *text*. Receiver = Qwen3-0.6B, Sharer =
Qwen3-4B-Base (frozen), Fuser = released `nics-efc/C2C_Fuser :: qwen3_0.6b+qwen3_4b_base_Fuser`.

## Status: what we ran and found

| Experiment | Outcome | Key numbers |
|---|---|---|
| **Paper reproduction** (OpenBookQA, no retrieval) | ✅ matched | receiver 39.6, C2C 53.4 (paper 39.2 / 53.4) |
| **GSM8K-RAG** (NOT in repo) | ❌ negative | C2C-RAG ~30 vs 0.6B zero-shot 54.7, text-RAG 49.0, 4B 76.7 |
| **OpenBookQA-RAG** | ✅ strong win | receiver 39.6, text 46.2, **C2C 62.2**, oracle: text 56.4 / C2C 69.0 |
| **TriviaQA-RAG** (contains) | ✅ win on contains | receiver 17.0, text 40.8, **C2C 44.6 (doc-only) / 48.2 (doc+q)**; oracle 56.0 / 60.4 / 63.0 |

**Thesis:** C2C-RAG helps when the task is **knowledge-bottlenecked with short answers** (the
fuser's trained regime) and fusion injects a *digested* representation the small model uses better
than raw text (which distracts it). It fails when the task is **reasoning-bottlenecked with long
CoT** (GSM8K) — retrieval can't supply reasoning, and long generation is out of the fuser's regime.

## Mechanism notes (important, non-obvious)

- **Fusion is per-section, not per-token.** `RosettaModel` reads one fusion flag per
  `kv_cache_index` *section* (`kv_cache_index[i][0][0][0]`, the section's first token).
  - Full-prompt fusion (`doc_q`): 2 sections `[fuse all prompt, gen]`.
  - **Doc-only fusion** (realistic/cacheable): 3 sections `[doc=fuse, question=no-fuse, gen]`.
  - We initially shipped a per-token mask (silently became full fusion — `doc_only==doc_q`, all
    500 gens identical). Fixed to the 3-section form in `rag_trivia.py`/`smoke_doc_fusion.py`.
- **Why no TurboRAG position handling here:** we compute everything in one forward pass and only
  choose *where* fusion applies. By causality this equals an offline question-agnostic doc cache,
  and for **top-1 + fixed preamble** the doc sits at consistent absolute positions → no RoPE
  mismatch. Position rearrangement (TurboRAG) is only needed for a persisted offline cache with
  **top-k / multi-chunk** composition.
- **Cacheability cost** = doc_q − doc_only ≈ 2.6–3.6 pts (the price of the deployable setup where
  the 4B never sees the query).

## Environment gotchas (cost us time)

- Install C2C with `pip install -e C2C --no-deps` to **keep torch 2.7** (C2C pins torch 2.6 /
  transformers 4.52.4; the torch mismatch is harmless, transformers 4.52.4 matters).
- **`USE_TF=0`** — the system TensorFlow hijacks the transformers import path; scripts set it, but
  export it if importing elsewhere.
- **`jinja2>=3.1`** — older jinja silently makes `apply_chat_template` skip every sample.
- Hardware used: single A100-40GB (peak ~12 GB). Greedy decoding throughout.

## Metric caveat (TriviaQA)

C2C outputs are wordier (~10 words vs ~6), so on strict **EM** text-RAG leads even though C2C wins
on **contains**. `rescore_trivia.py` adds verbosity-robust **prefix-EM** (narrows but doesn't flip
EM). We agreed `contains` is the win criterion; converting it to an EM win is a prompt-engineering
/ constrained-decoding task.

## Open questions / next steps (ranked)

1. **Verbosity → EM:** constrained / short-answer prompt or post-processing to turn the `contains`
   win into an EM win on TriviaQA. Highest-value quick follow-up.
2. **Top-k retrieval + true offline KV cache** with TurboRAG-style position rearrangement (the
   efficiency story; needed once chunks are composed).
3. **Train a GSM8K-format fuser** to test whether the reasoning negative is fixable or fundamental.
4. Better retriever — OBQA exact-fact hit is only 48% but C2C-RAG already 62.2 (oracle 69.0), so
   retrieval is the bottleneck, not the method.

## Where things live (on the original machine)

- This repo: experiment scripts + README. Run from `scripts/`.
- C2C package (the `rosetta` library + RosettaModel): cloned separately, `pip install -e`.
- Fuser checkpoints / base models: Hugging Face cache (auto-downloaded; `common.get_fuser_dir()`).
- **GSM8K scripts are NOT in this repo** (deliberately). On the original box they were in
  `C2C/rag_exp/` as `gen_demo_bank.py`, `run_eval.py`, `smoke_test.py` (+ a GSM8K-flavored
  `common.py`). Re-create from the OBQA/Trivia harnesses if needed.
- Results JSON (every generation saved) land in `scripts/obqa_results/` and
  `scripts/trivia_results/` — gitignored; regenerate by re-running.
