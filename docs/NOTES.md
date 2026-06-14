# Project notes / session handoff

Context to resume this work on a fresh machine. (The README has the polished reproduction
instructions; this file is the "what we did, why, and what's next" log.)

## Goal

Take Cache-to-Cache (C2C) "beyond the paper" into **retrieval-augmented generation**: fuse a
large sharer model's KV-cache *representation* of retrieved context into a small receiver model,
instead of feeding the receiver the retrieved *text*. Receiver = Qwen3-0.6B, Sharer =
Qwen3-4B-Base (frozen), Fuser = released `nics-efc/C2C_Fuser :: qwen3_0.6b+qwen3_4b_base_Fuser`.

## Current status (2026-06-13) — the paper

**SOLID WIN (single-hop, knowledge-bottlenecked):** the deployable, cacheable C2C-RAG — precompute
the 4B's fused KV per passage offline (4B never at query), 0.6B decodes from the loaded cache —
**beats the 0.6B reading text itself.** SQuAD top-1, LLM-judge n=500: receiver text-RAG 51.2,
**C2C cached 58.4** (+7.1, McNemar p=0.0002), 4B ceiling 70.0 — closes 38% of the 0.6B->4B gap at
0.6B cost. Holds on TriviaQA. Scored with a verbosity-robust reference-based LLM judge (the
fused model is wordier; encode/decode decomposition — fusion lifts recall/contains >> EM).

**HONEST BOUNDARY (multi-hop, reasoning-bottlenecked):** on HotpotQA the cacheable C2C does NOT
beat the receiver-alone baseline (consistent with the GSM8K negative). Cacheable fusion supplies
per-passage *knowledge*, not cross-passage *reasoning*, which the small model does itself.

**MACHINERY built + validated:** offline fused-KV cache (reproduces inline); CacheBlend (selective
HKVD recompute) recovers cross-chunk attention training-free (56-111% receiver / 33-47% C2C of the
block-local->full gap) — a real mechanism, but not a win over baseline on multi-hop. Multi-doc
loss is mostly RoPE position GAPS, not block-locality (dense contiguous composition ~flat on
single-hop). See dated sections below for the full arc. **Always measure recv-RAG + share-RAG
(text-RAG) baselines on a new dataset before claiming a C2C win** (the HotpotQA lesson).

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

1. ~~**Verbosity → EM**~~ **DONE** (2026-06-11): terse instruction + uniform copula extraction;
   C2C now wins EM/F1/contains, all significant. See "Verbosity fix — SOLVED" above. Remaining:
   re-run the distilled-text baseline under the new protocol to complete the table.
2. **Top-k retrieval + true offline KV cache** with TurboRAG-style position rearrangement (the
   efficiency story; needed once chunks are composed).
3. **Train a GSM8K-format fuser** to test whether the reasoning negative is fixable or fundamental.
4. Better retriever — OBQA exact-fact hit is only 48% but C2C-RAG already 62.2 (oracle 69.0), so
   retrieval is the bottleneck, not the method.

## Available fuser checkpoints (nics-efc/C2C_Fuser)

All 7 released fusers share the `<subfolder>/final` layout, so any is a drop-in swap
(`common.FUSER_SUBFOLDER` + RECEIVER/SHARER). We use `qwen3_0.6b+qwen3_4b_base_Fuser`.
Receiver+sharer pairs: `qwen3_0.6b+qwen3_4b_base` (ours, base sharer),
`qwen3_0.6b+qwen3_4b` (instruct sharer, same pair), `qwen3_0.6b+qwen2.5_0.5b`,
`qwen3_0.6b+llam3.2_1b`, `qwen3_0.6b+qwen2.5_1.5b_math`, `qwen3_1.7b+qwen2.5_1.5b`,
`qwen3_8b+qwen2.5_7b`. Candidate ablations:
- **instruct-sharer fuser** (`qwen3_0.6b+qwen3_4b`): a cleaner "answer-shaped" sharer KV may
  reduce verbosity on its own — but the honest text baseline must then also switch to the
  instruct 4B (our distilled baseline uses base 4B for a clean channel comparison).
- **`qwen3_8b+qwen2.5_7b`**: tests whether the RAG-fusion win scales with model size (leaves
  the small-receiver efficiency thesis behind).

## Second benchmark: SQuAD v1.1 — REPLICATES (2026-06-11)

Hardened the harness to be **dataset-generic**: `common.paths(name)` resolves
`<name>_data.json` / `<name>_corpus_emb.npy` / `<name>_results/` / `<name>_kvcache/`, and
`rag_trivia.py` / `rag_trivia_cached.py` / `judge_trivia.py` / `rescore_trivia.py` all take
`--data` (default `trivia`, behaviour unchanged). A new dataset = one prep script emitting the
shared schema. `squad_prep.py` builds SQuAD v1.1 (500 Qs over 2067 dev contexts; single-hop,
whole-paragraph chunks, gold = own context; retrieval top-1 hit 61%). Also fixed: `rescore`'s
glob was matching `*.judge.json` (now excluded).

**Result — LLM-judge (primary), original prompts, n≈490 (a few dup question strings collapse):**
| SQuAD | text_rag | c2c/doc_only | Δ | McNemar |
|---|--:|--:|--:|--|
| retrieved | 51.2% | **58.4%** | +7.1 | 61/26  p=0.0002 |
| oracle | 70.8% | **83.7%** | +12.9 | 93/30  p<0.0001 |

Same story as TriviaQA, **larger margins**. Lexical (copula) shows the classic split again —
C2C wins `contains` (oracle 79.0 vs 68.8, +10.2) but loses EM/F1 to verbosity (oracle EM 30.4
vs 43.6) — i.e. SQuAD independently reproduces the **encode/decode decomposition** (fusion lifts
recall/`contains`≫EM; verbosity-robust judge converts it to a correctness win). receiver-only is
~0% here (SQuAD answers aren't in the 0.6B's parametric memory), which sharpens the RAG framing.

## Receiver-own-KV control — fusion is the cause (SQuAD, 2026-06-11)

Ablation: run the *identical* doc-only-cache pipeline but build the doc cache with fusion OFF
(mask -1) → the receiver's OWN KV (`rag_trivia_cached.py --recv_kv`; `kvcache.build_doc_cache
(fuse=False)`; tag `recvkv_*`). Isolates fusion (sharer knowledge) from the cache/sectioning
machinery. LLM-judge:

| SQuAD | text_rag | recv_kv (own KV, c2c pipeline) | c2c (fusion) |
|---|--:|--:|--:|
| retrieved | 51.2 | 51.0 | 58.4 |
| oracle | 70.8 | 69.0 | 83.7 |

- **recv_kv ≈ text_rag** (hypothesis confirmed): retrieved McNemar 2/3 p=1.0 (identical), oracle
  4/13 p=0.049 (within ~2pts; the borderline p is just the prefill-chunking fp nondeterminism —
  94% exact-gen match to text_rag — recv_kv has no info advantage over text).
- **c2c vs recv_kv** (same pipeline, only the fused KV source differs): +7.4 retrieved p=0.0002,
  +14.7 oracle p<0.0001. So the win is **entirely the fusion of the sharer's KV**, not the
  doc-only-cache plumbing. Strong within-pipeline isolation. (Distinct from the not-yet-run
  weak-sharer 0.5B baseline, which would further split sharer *knowledge* vs *extra compute*.)

## Theory (for the paper, 2026-06-11)

Why KV fusion helps — framings, ranked by how well our data supports them:
1. **Information bottleneck / DPI:** text is a low-bandwidth serialization; the receiver
   re-encoding text recovers only what its params allow, whereas the sharer's KV is a function
   the receiver *cannot compute*. `text_rag` (same tokens/positions, only encoder differs) is the
   clean test. (Extends C2C's thesis — cite.)
2. **Encode/decode decomposition (OURS, best-supported):** fusion edits the KV (read/memory),
   not the output head, so it lifts *what's accessible* (`contains`) ≫ *how it generates* (EM) —
   exactly our verbosity pattern, on both datasets; and the GSM8K negative (reasoning is a
   decode-time computation fusion can't supply) is the same coin. Foreground this.
3. **Attention as associative memory:** sharer KV = pre-digested passage (entities/coref/salience
   resolved); receiver attends over richer keys/values (paper's effective-rank bump). Residual
   KV-space steering, input-conditioned by a stronger model.
4. **Cross-model representational alignment** lets a low-complexity projector translate
   sharer→receiver KV (model-stitching / Platonic universality; paper's T-SNE). Borrowed — cite.

Killer validation experiment (proposed, cheap): **linear answer-probe** on the receiver cache
with vs without fusion, per layer — fusion should make the gold answer more linearly decodable.
Plus weak-sharer (knowledge-vs-compute) and attention-mass-on-answer.

## Offline-cacheable doc-only path — IMPLEMENTED + validated (2026-06-11)

Turned the README's "doc_only is mathematically equivalent to an offline cache" claim into a
real persisted-cache artifact. Modules: `kvcache.py` (build/save/load/decode primitives),
`rag_trivia_cached.py` (two-phase runner), `spike_cache.py` (equivalence spike).

**Mechanism** (verified by reading `rosetta/model/wrapper.py`): `forward()` builds the fused base
KV section-by-section and returns it as `output.past_key_values`; fusion only applies to
NON-final sections. So: build = prefill `[preamble+doc, 1-dummy]` with sections `[fused, nofuse]`,
take `past_key_values`, truncate the dummy → the query-independent fused doc KV. Query = inject
that DynamicCache into the **base receiver** (`model.model_list[0]`) and manually prefill+greedy-
decode the question — **no 4B, no fuser at query time**. (HF `.generate` with a prefilled
multi-token cache is broken in transformers 4.52.4 → manual decode loop.)

**Validated (n=500) on BOTH datasets:** cached vs inline `c2c doc_only`, judge metric —
TriviaQA retrieved 43.6 vs 43.0, oracle 57.4 vs 58.2; SQuAD retrieved 57.2 vs 58.4, oracle 83.4
vs 83.7. F1/contains ~identical; 91–93% token-exact generations on both. The ~7–9% divergence is
greedy fp nondeterminism (inline splits the question prefill into 2 chunks vs the cache's 1), all
correctness-preserving. **Equivalence holds** — the precomputed-fused (4B-offline) top-1 result
stands on both benchmarks.

**Timing (A100, n=500, ~130-tok docs):** offline build ~0.1–0.2 s/doc (one-time, query-agnostic,
amortized over every query that retrieves it); online query ~0.52 s/q **receiver-only**. Per-query
wall-clock is decode-dominated (24 greedy steps) so not dramatically faster at unique-doc n=500 —
the real win is (a) the 4B+fuser leave the query-time critical path *and memory*, and (b)
amortization across repeated retrievals; both grow with sharer size, doc length, and reuse.
Footprint: ~18 MB/doc fp16 KV (28 layers × 8 KV heads × 128 dim) → ~9 GB / 500 docs (gitignored).
TODO: top-k / multi-chunk needs TurboRAG position rearrangement (HotpotQA will motivate it).

## LLM-as-judge — verbosity-robust, on the ORIGINAL prompts (2026-06-11)

Second, independent confirmation that doesn't need the terse hack. `judge_trivia.py` scores the
**original (non-terse, verbose) base generations** with the `claude` CLI (`-p --output-format
json --json-schema`, reference-based pointwise: judge sees question + answer + gold aliases,
returns correct/incorrect). The judge reads the whole answer so verbosity doesn't matter.

| setting | text_rag | c2c/doc_only | Δ | paired McNemar |
|---|--:|--:|--:|--|
| retrieved | 37.6% | **43.0%** | +5.4 | 52/25  p=0.0028 |
| oracle | 52.6% | **58.2%** | +5.6 | 58/30  p=0.0037 |

Validation: 86–89% agreement with `contains`; the judge fixes verbosity/paraphrase false-negs
(more for C2C, as expected) and rejects spurious alias-matches. Spot-checked — sound. Judge =
Claude Haiku 4.5, ~$3.70 for 2000 items (batched 40/call; ~24k cached system tokens/call). Note
`judge_trivia.py`'s internal `contains` uses only `aliases[:6]` (a cruder proxy than rescore's
full `norm_aliases`) — it's only for the agreement check, not a headline number.

**Both metric families agree** → the headline is robust: C2C-RAG > text-RAG on (a) lexical
EM/F1/contains under a short-answer protocol, and (b) LLM-judge correctness on the original
prompts. F1 significance is a paired bootstrap (McNemar is binary-only): retrieved Δ+3.90
p=0.018, oracle Δ+7.06 p=0.0007.

## Verbosity fix — SOLVED (TriviaQA, 2026-06-11)

Root cause: C2C **restates the question as a copular sentence with the answer as the predicate
nominal** ("The highest mountain in Africa is Kilimanjaro.") — avg ~10 words vs text's ~5. So
`contains` passes but the restating tokens tank F1 precision and block EM.

**What did NOT work** (all dead ends, measured):
- *Few-shot Q->A exemplars*: the 0.6B receiver **parrots** the exemplar answers ("Paris", "Au")
  regardless of the question — collapses text to ~10% contains. For C2C the exemplars sit in the
  fused preamble so they neither parrot nor help format.
- *Answer-priming* ("The answer is …" assistant prefix): greedy decoding **hijacks** to a salient
  date/number from the passage ("1993", "1978"). Tanks contains for both channels.

**What worked — two pieces, both needed, applied uniformly to every method:**
1. **Copula extraction at scoring** (`extract_answer` in `rag_trivia.py` / `rescore_trivia.py`):
   strip a leading "the answer is", then if the response is "<long restatement> is <ANSWER>",
   keep the predicate-nominal tail. Recovers EM/F1 from high-`contains` verbose output **without
   forcing the model terse** (so `contains` is preserved). Tuned offline in `extract_dev.py`.
2. **Terse instruction** (`--terse`): a short-answer directive in the prompt. Alone it costs the
   *text* channel `contains` (text is already terse, so brevity makes it commit wrong), but it is
   what lets C2C stop restating so copula extraction can isolate the entity. base+copula leaves
   C2C at 5.8 words (still loses); terse+copula → 3.7 words (wins).

**Result (n=500, copula extraction; matched terse protocol on both channels):**
| retrieved | EM | F1 | contains |   | oracle | EM | F1 | contains |
|---|--:|--:|--:|---|---|--:|--:|--:|
| text terse | 25.0 | 33.4 | 33.4 |   | text terse | 35.2 | 43.9 | 45.8 |
| **c2c doc_only terse** | **29.2** | **37.3** | **39.2** |   | **c2c doc_only terse** | **41.8** | **50.9** | **55.8** |

C2C now wins **EM, F1 and contains** (was: wins contains, loses EM/F1). Paired McNemar:
EM retrieved 51/30 p=0.026*, EM oracle 79/46 p=0.004*; contains retrieved 55/26 p=0.002*,
oracle 92/42 p<0.001*. **Robustness:** c2c-terse also beats text's *own best* config (base):
retrieved EM 29.2>25.8, F1 37.3>36.1; oracle EM 41.8>38.0, F1 50.9>48.3. doc_q (upper bound)
terse: retrieved 29.8/39.7/44.4, oracle 41.8/52.5/60.8.

Caveat to revisit: distilled-text baseline not yet re-run under this protocol (skipped for now).

## Where things live (on the original machine)

- This repo: experiment scripts + README. Run from `scripts/`.
- C2C package (the `rosetta` library + RosettaModel): cloned separately, `pip install -e`.
- Fuser checkpoints / base models: Hugging Face cache (auto-downloaded; `common.get_fuser_dir()`).
- **GSM8K scripts are NOT in this repo** (deliberately). On the original box they were in
  `C2C/rag_exp/` as `gen_demo_bank.py`, `run_eval.py`, `smoke_test.py` (+ a GSM8K-flavored
  `common.py`). Re-create from the OBQA/Trivia harnesses if needed.
- Results JSON (every generation saved) land in `scripts/obqa_results/` and
  `scripts/trivia_results/` — gitignored; regenerate by re-running.
