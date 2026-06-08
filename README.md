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

# 3) Install Python deps
pip install -r requirements.txt
```

Notes:
- **`USE_TF=0`** is set inside the scripts to avoid the system-TensorFlow import path in
  `transformers`; if you import these modules elsewhere, export `USE_TF=0` first.
- Models + the fuser projector are pulled automatically from the Hugging Face Hub on first run
  (`Qwen/Qwen3-0.6B` ~1.2 GB, `Qwen/Qwen3-4B-Base` ~8 GB, fuser ~1 GB).
- Hardware: everything fits on a single **A100-40GB** (peak ~12 GB). Greedy decoding throughout.
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

```bash
# build the retrieval set + corpus embeddings (one-time, streams TriviaQA)
python trivia_prep.py --target 500 --scan 2000

# baselines
python rag_trivia.py --method receiver --fact none
python rag_trivia.py --method text_rag --fact retrieved
python rag_trivia.py --method text_rag --fact oracle

# C2C, realistic doc-only fusion
python rag_trivia.py --method c2c --fact retrieved --fuse doc_only
python rag_trivia.py --method c2c --fact oracle    --fuse doc_only

# C2C, doc+question upper bound
python rag_trivia.py --method c2c --fact retrieved --fuse doc_q
python rag_trivia.py --method c2c --fact oracle    --fuse doc_q

# verbosity-corrected scoring (EM / prefix-EM / contains) from saved generations
python rescore_trivia.py
```

| cell | contains | EM | retr-hit |
|---|---:|---:|---:|
| `receiver/none` (0.6B baseline) | 17.0 | 6.2 | — |
| `text_rag/retrieved` | 40.8 | 21.8 | 53% |
| **`c2c/retrieved/doc_only`** (realistic) | **44.6** | 7.4 | 53% |
| `c2c/retrieved/doc_q` (upper bound) | 48.2 | 12.8 | 53% |
| `text_rag/oracle` | 56.0 | 32.2 | — |
| `c2c/oracle/doc_only` | 60.4 | 12.0 | — |
| `c2c/oracle/doc_q` | 63.0 | 20.4 | — |

**On `contains`, fusion wins at every retrieval level** (doc-only 44.6 > text 40.8; doc+q 48.2;
oracle 60.4/63.0 > text-oracle 56.0), and all C2C cells crush the 17.0 baseline. The
**cacheability cost** (doc_q − doc_only) is a modest 2.6–3.6 pts.

> **EM caveat.** On strict exact-match, text-RAG leads — the fused model is wordier (~10 words vs
> ~6), so it *contains* the right answer but rarely *equals* it. `rescore_trivia.py` reports a
> verbosity-robust **prefix-EM** that narrows the gap; a constrained short-answer prompt is the
> obvious fix to convert the `contains` win into an EM win.

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
  rag_trivia.py        # Experiment 3 (TriviaQA-RAG, doc_only / doc_q fusion)
  trivia_prep.py       # builds the TriviaQA passage corpus + embeddings
  rescore_trivia.py    # verbosity-corrected EM / prefix-EM / contains
  smoke_doc_fusion.py  # minimal doc-only fusion mechanism check
```

Results are written to `scripts/obqa_results/` and `scripts/trivia_results/` (with every
generation saved, so metrics can be recomputed offline).
