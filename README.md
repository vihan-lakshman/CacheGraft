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
| TriviaQA-RAG (retrieved passage) | contains | 17.0 | 40.8 | **48.2** (doc+q) / **44.6** (doc-only) |

**Where C2C-RAG wins:** knowledge-bottlenecked tasks with **short answers** (the fuser's
trained regime). Fusing a retrieved document's *representation* beats pasting its *text*,
which distracts the small model. **Where it doesn't:** reasoning-bottlenecked tasks with long
chain-of-thought (e.g. GSM8K), which are out of the released fuser's regime — retrieval can't
supply *reasoning*.

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

**Reproduce the full sweep** (run from `scripts/`; if you used the `PYTHONPATH` fallback,
`export PYTHONPATH=/path/to/C2C USE_TF=0` first). The 10 cells below each run as a separate
process, so GPU memory is freed between them; the four `c2c` cells decode one example at a time
(~5–6 min each), the rest are batched and fast. End-to-end on one A100-40GB is ~1.5 h, dominated
by the offline distillation of the full 21,812-chunk corpus (~50 min).

```bash
# 0) build the retrieval set + corpus embeddings (one-time, streams TriviaQA)
python trivia_prep.py --target 500 --scan 2000

# 1) offline 4B distillation of every passage -> trivia_distilled.json (the distilled baseline)
python trivia_distill.py

# 2) no-retrieval references
python rag_trivia.py --method receiver --fact none      # 0.6B parametric floor
python rag_trivia.py --method sharer   --fact none      # 4B parametric

# 3) text baselines: raw passage, and the offline-summarized (distilled) passage
python rag_trivia.py --method text_rag --fact retrieved
python rag_trivia.py --method text_rag --fact oracle
python rag_trivia.py --method text_rag --fact retrieved --distill summary
python rag_trivia.py --method text_rag --fact oracle    --distill summary

# 4) Tacit / C2C: realistic doc-only fusion, and the doc+question upper bound
python rag_trivia.py --method c2c --fact retrieved --fuse doc_only
python rag_trivia.py --method c2c --fact oracle    --fuse doc_only
python rag_trivia.py --method c2c --fact retrieved --fuse doc_q
python rag_trivia.py --method c2c --fact oracle    --fuse doc_q

# 5) scoring (contains / F1 / EM / prefix-EM, avg words) + paired McNemar on `contains`
python rescore_trivia.py
```

Latest run (n=500, retrieval top-1 gold-chunk hit = 53%). `contains` is the headline metric;
F1/EM are verbosity-penalized (see caveat). Distilled here is the **leaky base-4B** version
(faithful re-run pending).

| cell | retrieved contains | retrieved F1 | oracle contains | oracle F1 |
|---|---:|---:|---:|---:|
| `receiver/none` (0.6B param) | 17.0 | 14.7 | — | — |
| `sharer/none` (4B param) | 35.8 | 20.7 | — | — |
| `text_rag` (raw passage) | 40.8 | 33.5 | 56.0 | 44.3 |
| `text_rag --distill summary` | 39.8 | 34.6 | 47.6 | 40.5 |
| **`c2c/doc_only`** (realistic) | **44.6** | 22.6 | **60.4** | 29.5 |
| `c2c/doc_q` (upper bound) | 48.2 | 28.0 | 63.0 | 36.4 |

Paired McNemar on `contains` (exact, two-sided) — `doc_only` beats **both** text rungs in both
settings, and the margin over distilled grows in oracle (compression loss fully exposed):

| comparison | win / lose | p |
|---|---:|---:|
| `doc_only` vs raw (retrieved) | 43 / 24 | 0.027 |
| `doc_only` vs distilled (retrieved) | 60 / 36 | 0.018 |
| `doc_only` vs distilled (oracle) | 104 / 40 | <0.0001 |

**On `contains`, fusion wins at every retrieval level**, all significant, and the
**cacheability cost** (doc_q − doc_only) is a modest 2.6–3.6 pts.

> **Verbosity caveat.** On F1 and strict EM, the text baselines lead — the fused model is wordier
> (~10 words vs ~5), so it *contains* the right answer but the extra tokens hurt token-precision
> (F1) and exact-match (EM). `rescore_trivia.py` reports a verbosity-robust **prefix-EM**; a
> constrained short-answer prompt is the planned fix to convert the `contains` win into an F1/EM win.

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

## Repo layout

```
scripts/
  common.py            # models, fuser download helper (get_fuser_dir)
  rag_obqa.py          # Experiments 1 & 2 (OpenBookQA + OpenBookQA-RAG)
  rag_trivia.py        # Experiment 3 (TriviaQA-RAG, doc_only / doc_q fusion, --distill)
  trivia_prep.py       # builds the TriviaQA passage corpus + embeddings
  trivia_distill.py    # offline 4B-Base summary of every passage (distilled-text baseline)
  rescore_trivia.py    # verbosity-corrected EM / prefix-EM / contains + paired McNemar
  smoke_doc_fusion.py  # minimal doc-only fusion mechanism check
```

Results are written to `scripts/obqa_results/` and `scripts/trivia_results/` (with every
generation saved, so metrics can be recomputed offline).
