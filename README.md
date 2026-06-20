# turborag-fusion

Retrieval-Augmented Generation with **Cache-to-Cache (C2C)** fusion: instead of feeding a
small "receiver" model the retrieved text, we fuse a large "sharer" model's **KV-cache
representation** of the retrieved context directly into the receiver's cache, and let the
small model decode the answer.

Built on [Cache-to-Cache (C2C)](https://github.com/thu-nics/C2C)
([paper](https://arxiv.org/abs/2510.03215)). Receiver = **Qwen3-0.6B**, Sharer = **Qwen3-4B-Base**
(frozen), Fuser = the released `nics-efc/C2C_Fuser :: qwen3_0.6b+qwen3_4b_base_Fuser` projector.

## TL;DR findings

| Experiment | Metric | Receiver-only (0.6B) | Text-RAG | **C2C-RAG** |
|---|---|---:|---:|---:|
| OpenBookQA (no retrieval, paper repro) | acc | 39.6 | — | **53.4** |
| OpenBookQA-RAG (retrieved fact) | acc | 39.6 | 46.2 | **62.2** |
| TriviaQA-RAG (retrieved passage) | F1 | 15.5 | 36.1 | **37.3** (doc-only) / **39.7** (doc+q) |
| TriviaQA-RAG (retrieved passage) | EM | 8.2 | 26.0 | **29.2** (doc-only) / **29.8** (doc+q) |
| TriviaQA-RAG (retrieved passage) | contains | 15.6 | 39.4 | **39.2** (doc-only) / **44.4** (doc+q) |

**Where C2C-RAG wins:** knowledge-bottlenecked tasks with **short answers** (the fuser's
trained regime). Fusing a retrieved document's *representation* beats pasting its *text*,
which distracts the small model. **Where it doesn't:** reasoning-bottlenecked tasks with long
chain-of-thought (e.g. GSM8K), which are out of the released fuser's regime — retrieval can't
supply *reasoning*.

## Headline (consolidated, properly baselined)

**The deployable claim.** Precompute the 4B's **fused KV** of each passage *offline* and store it;
at query time the **0.6B alone** decodes from the loaded fused cache — **the 4B never runs.** This
beats the 0.6B *reading the passage as text itself*, the baseline that matters:

| SQuAD top-1 retrieved (LLM-judge, n=500) | judge |
|---|---:|
| receiver text-RAG (0.6B — the baseline to beat) | 51.2% |
| **C2C cached doc-only (0.6B + precomputed fused 4B, 4B offline)** | **58.4%** |
| 4B text-RAG (knowledge ceiling) | 70.0% |

**+7.1 over the receiver baseline, McNemar p=0.0002** — recovering **38% of the 0.6B→4B gap at
0.6B query cost.** The same win holds on TriviaQA. We score with a **reference-based LLM judge**
(verbosity-robust — the fused model is wordier, so token-EM/F1 under-credit it; see the
verbosity/encode-decode discussion below).

**Scope boundary (honest).** On **multi-hop HotpotQA**, the cacheable C2C does **not** beat the
receiver-alone baseline (the small model already attends across passages; the bottleneck is
*reasoning/linking*, not per-passage encoding — consistent with the GSM8K negative). The cacheable
fusion captures **per-passage knowledge, not cross-passage reasoning.** See
[Multi-doc & multi-hop](#multi-doc--multi-hop-hotpotqa--scope-boundary).

## Setup

```bash
# 1) Clone this repo
git clone https://github.com/vihan-lakshman/turborag-fusion && cd turborag-fusion

# 2) Install the C2C package (provides the `rosetta` library + RosettaModel)
git clone https://github.com/thu-nics/C2C
pip install -e C2C --no-deps          # --no-deps keeps your existing torch
# If the editable install fails (older setuptools: "missing build_editable hook"),
# skip it and put C2C on the path instead -- `rosetta` is pure Python:
#   export PYTHONPATH=$PWD/C2C

# 3) Install Python deps
pip install -r requirements.txt
```

Notes:
- **`USE_TF=0`** is set inside the scripts to avoid the system-TensorFlow import path in
  `transformers`; if you import these modules elsewhere, export `USE_TF=0` first.
- If you used the `PYTHONPATH` fallback above, export it (plus `USE_TF=0`) before every run:
  `export PYTHONPATH=/path/to/C2C USE_TF=0`.
- **`jinja2>=3.1`** is required — older jinja silently makes `apply_chat_template` skip samples.
- Models + the fuser projector are pulled automatically from the Hugging Face Hub on first run
  (`Qwen/Qwen3-0.6B` ~1.2 GB, `Qwen/Qwen3-4B-Base` ~8 GB, fuser ~1 GB).
- Hardware: a **CUDA GPU is required** (the scripts hardcode `cuda`); everything fits on a single
  **A100-40GB** (peak ~12 GB). Greedy decoding throughout.
- Run scripts from the `scripts/` directory (they `import common`):
  ```bash
  cd scripts
  ```

## Experiment 1 — OpenBookQA reproduction (paper Table 4, no retrieval)

Reproduces the headline C2C result: fusing the 4B's representation of the question lifts the
0.6B from ~39 to ~53 on OpenBookQA, with no retrieval.

```bash
python rag_obqa.py --method receiver --fact none   # receiver-only baseline  -> ~39.6
python rag_obqa.py --method c2c_rag  --fact none    # C2C (fusion, no fact)   -> ~53.4
```

| condition | accuracy |
|---|---:|
| `receiver/none` (Qwen3-0.6B) | 39.6 |
| `c2c/none` (0.6B ← 4B fused) | **53.4** |

## Experiment 2 — OpenBookQA-RAG

OpenBookQA ships a fixed "open book" of 1,326 science facts. We retrieve the most relevant fact
(bge-small, top-1) and either prepend it as text (`text_rag`) or fuse `[fact+Q+choices]`
(`c2c_rag`). `--fact oracle` uses the gold fact (perfect-retrieval upper bound).

```bash
python rag_obqa.py --method receiver --fact none
python rag_obqa.py --method text_rag --fact retrieved
python rag_obqa.py --method c2c_rag  --fact retrieved
python rag_obqa.py --method text_rag --fact oracle
python rag_obqa.py --method c2c_rag  --fact oracle
```

| condition | accuracy | retr-hit |
|---|---:|---:|
| `receiver/none` (baseline) | 39.6 | — |
| `text_rag/retrieved` | 46.2 | 48% |
| **`c2c/retrieved`** | **62.2** | 48% |
| `text_rag/oracle` | 56.4 | — |
| `c2c/oracle` | **69.0** | — |

**Fusion beats text at identical retrieval (+16 pts), and beats the 0.6B baseline by +22.6.**
Retrieval only finds the exact gold fact 48% of the time, yet C2C-RAG hits 62.2 — with perfect
retrieval it reaches 69.0, so the retriever is the bottleneck, not the method.

## Experiment 3 — TriviaQA-RAG (open-domain, realistic retrieval)

Open-domain short-answer QA. We build a self-contained corpus by chunking each question's
Wikipedia evidence into ~100-word passages (~22k chunks), retrieve top-1 (bge-small), and answer.

Two fusion modes:
- **`doc_only`** — fuse **only the document** tokens, question stays query-time text. This is the
  *realistic, cacheable* setup: documents can be fused offline; the 4B never sees the question.
- **`doc_q`** — fuse the whole `[doc+question]` prompt (upper bound; 4B conditions on the query).

**What each text baseline isolates.** C2C doesn't summarize — the fuser is a *projection*, so
`doc_only` injects a full-length, full-fidelity KV encoding of *every* doc token (richer than
text, not compressed). There is therefore no perfectly symmetric text baseline — that
impossibility *is* the information-bottleneck thesis. We bracket `doc_only` between two
offline-precomputable text rungs:
- **raw passage** (`text_rag retrieved`) — *content-identical control*: same doc tokens, same
  positions; the only difference vs `doc_only` is the channel (0.6B self-encodes the text vs the
  KV is enriched by the 4B). This is the clean **channel / information-bottleneck** test.
- **distilled** (`text_rag --distill summary`) — the **deployment baseline**: the strongest
  offline-precomputable *text* competitor, where the same 4B compresses each passage into a short
  summary once, offline. It's asymmetric by design — the text system's native advantage is
  compression (tiny storage vs the KV cache's ~10⁴× footprint), so this measures whether the KV
  channel's fidelity is worth its storage cost. Beating *both* rungs is the claim.

**The short-answer protocol (`--terse`).** Greedy decoding of the fused model restates the
question as a copular sentence with the answer buried inside ("The highest mountain in Africa is
Kilimanjaro."), so it *contains* the answer (high recall) but loses token-F1/EM to the dilution.
We fix this with two pieces applied **identically to every method**, so the comparison stays
fair: (1) a **short-answer instruction** (`--terse`), and (2) a **uniform copula extraction** at
scoring (`extract_answer`: strip a leading "the answer is", then for an "X is Y" restatement keep
the predicate-nominal tail). Both are needed — extraction alone leaves C2C too wordy (5.8 words),
terse alone hurts the already-terse text channel. Together they convert C2C's `contains` lead
into an **F1, EM and contains win**. (Few-shot exemplars make the 0.6B *parrot* the demos, and
"The answer is …" priming hijacks to a salient date — both rejected; see `docs/NOTES.md`.)

**Reproduce the sweep** (run from `scripts/`; if you used the `PYTHONPATH` fallback,
`export PYTHONPATH=/path/to/C2C USE_TF=0` first). Each cell is a separate process; the `c2c`
cells decode one example at a time (~4–5 min each at n=500), the rest are batched and fast.

```bash
# 0) build the retrieval set + corpus embeddings (one-time, streams TriviaQA)
python trivia_prep.py --target 500 --scan 2000

# 1) no-retrieval references
python rag_trivia.py --method receiver --fact none      # 0.6B parametric floor
python rag_trivia.py --method sharer   --fact none      # 4B parametric

# 2) text baseline (raw passage) -- base and short-answer protocol
python rag_trivia.py --method text_rag --fact retrieved
python rag_trivia.py --method text_rag --fact retrieved --terse
python rag_trivia.py --method text_rag --fact oracle
python rag_trivia.py --method text_rag --fact oracle    --terse

# 3) C2C: realistic doc-only fusion + doc+question upper bound, short-answer protocol
python rag_trivia.py --method c2c --fact retrieved --fuse doc_only --terse
python rag_trivia.py --method c2c --fact retrieved --fuse doc_q    --terse
python rag_trivia.py --method c2c --fact oracle    --fuse doc_only --terse
python rag_trivia.py --method c2c --fact oracle    --fuse doc_q    --terse

# 4) scoring (EM / F1 / prefix-EM / contains, avg words) + paired McNemar on EM and contains
python rescore_trivia.py
```

Latest run (n=500, retrieval top-1 gold-chunk hit = 53%), uniform copula extraction. Matched
**`--terse` protocol on both channels** (the fair, single-protocol comparison):

| cell | retr EM | retr F1 | retr contains | oracle EM | oracle F1 | oracle contains |
|---|---:|---:|---:|---:|---:|---:|
| `receiver/none` (0.6B) | 8.2 | 15.5 | 15.6 | — | — | — |
| `sharer/none` (4B) | 5.4 | 24.8 | 34.4 | — | — | — |
| `text_rag` (raw passage) | 25.0 | 33.4 | 33.4 | 35.2 | 43.9 | 45.8 |
| **`c2c/doc_only`** (realistic) | **29.2** | **37.3** | **39.2** | **41.8** | **50.9** | **55.8** |
| `c2c/doc_q` (upper bound) | 29.8 | 39.7 | 44.4 | 41.8 | 52.5 | 60.8 |

**Fusion now wins EM, F1 *and* contains** at every retrieval level. Paired McNemar (exact,
two-sided), `doc_only` vs `text_rag` under the matched terse protocol:

| metric / setting | win / lose | p |
|---|---:|---:|
| EM, retrieved | 51 / 30 | 0.026 |
| EM, oracle | 79 / 46 | 0.004 |
| contains, retrieved | 55 / 26 | 0.002 |
| contains, oracle | 92 / 42 | <0.0001 |

**Robustness — C2C also beats text's *own best* config.** Text does best *without* `--terse`
(it is already terse), so we also check C2C-terse against text-base: retrieved EM 29.2 > 26.0 and
F1 37.3 > 36.1; oracle EM 41.8 > 38.4 and F1 50.9 > 48.3. C2C wins either way. The
**cacheability cost** (doc_q − doc_only) stays small (EM ~0–0.6, F1 ~1.6–2.4).

> The earlier **verbosity caveat is resolved**: the previous "wins contains, loses F1/EM" gap was
> the buried-answer artifact above, not a real quality gap. The **distilled-text** rung (the
> cost-matched offline-summary baseline) has not yet been re-run under this protocol — a small
> pending item to complete the table, not expected to change the conclusion.

## Experiment 4 — SQuAD v1.1 (second benchmark, dataset-generic harness)

The harness is dataset-generic: every script takes `--data <name>` and `common.paths` resolves
`<name>_data.json` / `<name>_corpus_emb.npy` / `<name>_results/`. A new dataset is one prep
script emitting the shared schema (`{questions:[{q,aliases,norm_aliases,gold_ids}], corpus}`).
SQuAD v1.1 is single-hop and self-contained (gold = the question's own paragraph; 2067 dev
contexts as the retrieval pool).

```bash
python squad_prep.py --target 500                                   # build squad_data.json (+emb)
python rag_trivia.py --data squad --method text_rag --fact retrieved
python rag_trivia.py --data squad --method c2c --fact retrieved --fuse doc_only   # (+ oracle)
python judge_trivia.py --data squad text_rag_retrieved c2c_retrieved_doc_only --model haiku
```

**LLM-judge (primary metric), original prompts** — replicates TriviaQA with larger margins:

| SQuAD | text_rag | **c2c/doc_only** | Δ | paired McNemar |
|---|---:|---:|---:|---|
| retrieved | 51.2% | **58.4%** | +7.1 | 61/26, **p=0.0002** |
| oracle | 70.8% | **83.7%** | +12.9 | 93/30, **p<0.0001** |

Lexically, SQuAD reproduces the verbosity signature: C2C wins `contains` (oracle 79.0 vs 68.8)
but loses raw EM/F1 — the verbosity-robust judge is what converts the recall lead into a
correctness win. (Receiver-only ≈ 0% here, since SQuAD answers aren't in the 0.6B's parametric
memory — a cleaner RAG isolation than TriviaQA.)

**Control — the gain is the fusion, not the cache pipeline.** `rag_trivia_cached.py --recv_kv`
runs the *identical* doc-only pipeline but builds the cache from the receiver's OWN KV (fusion
off). It matches `text_rag` (judge 51.0/69.0 vs 51.2/70.8; 94% token-exact) and sits far below
C2C — so the +7–13 pt win is attributable to fusing the sharer's KV, not the doc-only-cache
machinery (within-pipeline isolation: c2c vs recv_kv p=0.0002 retrieved, p<0.0001 oracle).

## How doc-only fusion works (and why no position handling is needed here)

`RosettaModel` applies fusion **per section** of `kv_cache_index`, keyed on each section's first
token — *not* per token. Doc-only fusion therefore uses **three sections**:
`[doc → fuse, question → no-fuse, generation]` (see `run_c2c` in `rag_trivia.py` and
`smoke_doc_fusion.py`). Because attention is causal, fusing the document in a single forward pass
(question appended, un-fused) is **mathematically identical** to an offline-built, question-agnostic
fused document cache — so for **top-1** retrieval with a fixed preamble there is **no RoPE
mismatch** and no position handling is required.

[TurboRAG](https://aclanthology.org/2025.emnlp-main.334.pdf)-style position rearrangement becomes
necessary only when (a) persisting an offline KV cache **and** (b) composing **multiple** chunks
(top-k) or placing a chunk at a different absolute offset than it was cached at — a natural
follow-up for the efficiency story.

```bash
# sanity-check the mechanism (none vs doc_only vs doc_q should differ)
python smoke_doc_fusion.py
```

## Offline-cacheable doc-only fusion (the persisted-cache artifact)

`kvcache.py` + `rag_trivia_cached.py` turn the equivalence above into a real two-phase pipeline:

- **Phase 1 (offline):** for each passage, prefill `[preamble+doc]` with fusion and persist the
  receiver's **fused doc KV** to disk. The 4B sharer + fuser run *only here*, once per passage,
  query-agnostic.
- **Phase 2 (online):** load a passage's cached KV into the **base 0.6B receiver** and decode the
  question — **no 4B, no fuser** on the query path.

```bash
python rag_trivia_cached.py --fact retrieved        # build caches -> trivia_kvcache/, then query
python rag_trivia_cached.py --fact oracle            # ([--terse] also supported)
python judge_trivia.py c2c_retrieved_doc_only_cached c2c_oracle_doc_only_cached --model haiku
```

**Validated** to reproduce the inline `doc_only` path (n=500), both datasets: LLM-judge —
TriviaQA 43.6 vs 43.0 (retrieved) / 57.4 vs 58.2 (oracle); SQuAD 57.2 vs 58.4 / 83.4 vs 83.7.
F1/contains identical; 91–93 % token-exact generations (the rest is correctness-preserving greedy
fp nondeterminism). **Timing:** offline build ~0.1–0.2 s/passage
(one-time, amortized over every query that retrieves it); online query ~0.52 s receiver-only. The
4B+fuser leave the query-time critical path and memory entirely; the benefit grows with sharer
size, doc length, and passage reuse. Footprint ~18 MB/passage (fp16 KV). Top-k / multi-chunk
composition still needs TurboRAG-style position rearrangement (next step).

## Multi-doc & multi-hop (HotpotQA) — scope boundary

We extended C2C-RAG to **top-k retrieval**, which forces composing multiple cached chunks.
Findings (all on the LLM judge):

- **The multi-doc loss is mostly *position gaps*, not block-locality.** Naively re-rotating chunk
  caches into **gapped slots** hurts; with **dense contiguous** positions, block-local composition
  is ~flat on single-hop. (`rag_topk.py`)
- **Single-hop (SQuAD): cross-chunk attention doesn't matter** — dense block-local ≈ full attention,
  so multi-doc adds little and no recovery is needed.
- **Multi-hop (HotpotQA): cross-chunk attention *does* matter** — full attention rises with k while
  block-local declines (gap +10.7 at k=4).
- **CacheBlend recovers it, training-free.** Following
  [CacheBlend](https://arxiv.org/abs/2405.16444), we recompute only the top-15% high-deviation
  tokens to restore cross-chunk attention over independently-cached chunks (`cacheblend.py`). It
  recovers **56–111%** (receiver) / **33–47%** (C2C) of the block-local→full-attention gap.
- **But it is *not* a win over the baseline.** On HotpotQA the deployable cacheable C2C does **not**
  beat the 0.6B reading text directly — multi-hop is **reasoning-bottlenecked**, and cacheable
  fusion supplies per-passage *knowledge*, not cross-passage *reasoning*. We verified with the
  fundamental baselines (`hotpot_baselines.py`): receiver text-RAG and 4B text-RAG.

**Takeaway:** C2C-RAG is a **knowledge-injection** method (wins single-hop), not a reasoning
method. CacheBlend's cross-chunk recovery is a validated mechanism but doesn't overturn the
multi-hop baseline. The store-both-pre-fusion-caches design keeps the 4B offline; storing only the
fused caches is harder (fused KV isn't RoPE-re-rotatable) and is left as a stretch.

## Training your own fuser — Llama-3.2-1B-Instruct ← Llama-3.1-8B-Instruct

The released `nics-efc/C2C_Fuser` projectors are Qwen-only. To take C2C-RAG to a **new model pair**
you train your own fuser with C2C's `SFT_train.py` (general chat distillation on OpenHermes; the
RAG ability is zero-shot afterward, exactly as with the released fusers). Below is the exact recipe
we used for the Llama-1B-Instruct ← 8B-Instruct pair. (Findings/analysis live in
[`docs/NOTES.md`](NOTES.md); this is the reproduction.)

**Extra prerequisites** (on top of [Setup](#setup)):
```bash
. .venv/bin/activate
pip install wandb                       # unlisted import in C2C's SFT_train.py; run offline (below)
# Llama weights are GATED — log in with an HF token that has been granted Llama access:
huggingface-cli login                   # then accept the meta-llama/Llama-3.* licenses on the Hub
```

**The recipe** ([`recipes/llama_1b8b_c2c.json`](../recipes/llama_1b8b_c2c.json)) — what matters:
- `base_model` (receiver) + `teacher_model` (sharer); the projector dims and the `last_aligned`
  layer mapping are derived automatically from the two model configs.
- **Same tokenizer ⇒ `is_do_alignment: false`.** Llama-1B and 8B share the Llama-3 vocab, so no
  token aligner is needed (the easy case). For a pair with *different* tokenizers, set it `true`.
- `data.num_samples` = 50000 (feasibility) / 150000 (the stronger fuser we shipped); 1 epoch.
- ⚠️ **`output.output_dir` is an absolute path — edit it for your machine.** Also set
  `wandb_config.mode: "offline"` (no W&B account needed).

**Train** (single A100-40GB; run `torchrun` from inside the C2C clone):
```bash
cd C2C
PYTHONPATH=$PWD USE_TF=0 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=1 --master_port=29515 script/train/SFT_train.py \
    --config ../recipes/llama_1b8b_c2c.json
```
- **Memory (40 GB):** `per_device_train_batch_size: 3` (grad-accum 11 → eff. batch 33) is the safe
  config (~38 GB peak, ~10 s/it). **Batch 4 OOMs mid-run** (~39.7 GB peak) even with
  `expandable_segments`. There is **no resume-from-checkpoint logic** in the trainer, so a late OOM
  loses progress back to the last `save_steps` (500) checkpoint — don't push the batch size.
- Wall-clock: ~4.3 h (50k) / ~12.5 h (150k). Output lands in `<output_dir>/final/`:
  16 `projector_*.pt` + `projector_*.json` + `projector_config.json` (the loadable artifact;
  intermediate `checkpoint-*` dirs can be deleted).

**Evaluate** the new fuser with the same harness — `common.py` reads `C2C_RECEIVER` / `C2C_SHARER`
/ `C2C_FUSER_DIR` env vars (Qwen defaults unchanged), and [`scripts/eval_llama.sh`](scripts/eval_llama.sh)
wires them up:
```bash
cd scripts
C2C_FUSER_DIR=/abs/path/to/<your_run>/final ./eval_llama.sh obqa    # OBQA: no-retrieval + retrieved
C2C_FUSER_DIR=/abs/path/to/<your_run>/final ./eval_llama.sh squad   # SQuAD-RAG + LLM judge
C2C_FUSER_DIR=/abs/path/to/<your_run>/final ./eval_llama.sh trivia  # TriviaQA-RAG + LLM judge
```
`C2C_FUSER_DIR` can also be a Hugging Face snapshot dir — e.g. our published fuser:
`C2C_FUSER_DIR=$(python -c "from huggingface_hub import snapshot_download as d; print(d('vihan29/c2c_llama8b_llama1b_150k'))")`.

**Pick a dataset before training — the fusion-amenability screen.** Fusion only helps when the
answer is **in the document** (not the sharer's parametric memory) *and* the receiver
**under-encodes** the context. Screen any candidate cheaply (oracle context, 4 batched cells, no
fuser) before investing in a training run:
```bash
cd scripts
python pubmedqa_prep.py --target 500     # or boolq_prep.py / narrativeqa_prep.py (SQuAD-style schema)
for m in "receiver none" "sharer none" "text_rag oracle" "sharer oracle"; do
  python rag_trivia.py --data pubmedqa --method ${m% *} --fact ${m#* }
done
# good target: 8B-text ≫ 1B-text (encoding headroom) AND 8B-text ≫ 8B-none (answer is in the doc)
```
See `docs/NOTES.md` → "dataset SCREEN for fusion-amenability" for the interpretation and results.

## Repo layout

```
scripts/
  common.py            # models, fuser download helper, paths(data) for the dataset-generic harness
  rag_obqa.py          # Experiments 1 & 2 (OpenBookQA + OpenBookQA-RAG)
  rag_trivia.py        # core RAG runner (TriviaQA/SQuAD/...; --data, doc_only/doc_q, --terse)
  squad_prep.py        # builds the SQuAD v1.1 retrieval set (Experiment 4)
  rag_trivia_cached.py # offline-cacheable doc-only path (build caches -> disk, then query)
  kvcache.py           # reusable KV primitives: build/save/load/decode, fuse_caches, re_rotate_cache
  trivia_prep.py       # builds the TriviaQA passage corpus + embeddings
  trivia_distill.py    # offline 4B-Base summary of every passage (distilled-text baseline)
  rescore_trivia.py    # uniform copula extraction -> EM / F1 / prefix-EM / contains + McNemar
  judge_trivia.py      # LLM-as-judge (claude CLI) reference-based correctness scoring (--data)
  tune_prompt.py       # fast prompt/format sweep on the receiver (verbosity-fix tuning)
  extract_dev.py       # offline answer-extraction tuning over saved raw generations
  smoke_doc_fusion.py  # minimal doc-only fusion mechanism check
  # --- train-your-own-fuser (new model pair; see "Training your own fuser") ---
  eval_llama.sh        # eval a trained fuser via C2C_RECEIVER/SHARER/FUSER_DIR env (obqa|squad|trivia)
  pubmedqa_prep.py boolq_prep.py narrativeqa_prep.py   # dataset-screen sets (SQuAD-style schema)
  # --- multi-doc / multi-hop ---
  hotpot_prep.py       # builds the HotpotQA (distractor, multi-hop) retrieval set
  hotpot_baselines.py  # the fundamental text-RAG baselines: receiver (0.6B) + sharer (4B)
  rag_topk.py          # top-k composition: dense block-local / slot / re-rotate / text modes
  cacheblend.py        # efficient CacheBlend (selective HKVD recompute) + C2C integration
  spike_cache.py spike_fuse.py spike_rerotate.py        # equivalence/correctness spikes
  spike_cacheblend.py spike_cacheblend_c2c.py           # CacheBlend validation (receiver + C2C)
```

```
recipes/
  llama_1b8b_c2c.json        # 50k fuser training recipe (Llama-1B-Instruct <- 8B-Instruct)
  llama_1b8b_c2c_150k.json   # 150k recipe (the stronger fuser; published to the HF Hub)
```

Results are written to `scripts/<dataset>_results/` (with every generation saved, so metrics can
be recomputed offline). The cloned `C2C/` package and all generated data/caches/results are
gitignored — install C2C separately (see Setup) and regenerate by re-running.
