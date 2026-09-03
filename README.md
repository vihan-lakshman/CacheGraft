# CacheGraft — precomputed cross-model KV enrichment for RAG

Standard RAG hands a retrieved passage to the reader as **text**, so a small reader must
re-encode that passage with its own limited parameters on every query. CacheGraft delivers the
passage as a **larger model's KV-cache representation of it**, projected into the small model's
cache by a trained fuser.

The restriction that makes it deployable: fuse **only the document section** of the prompt and
leave the question as ordinary text. By causality the fused document cache then depends on the
document alone — it is **query-independent**, so it can be built once per passage offline and
reused by every future query that retrieves it. At query time only the small receiver runs.

Built on [Cache-to-Cache (C2C)](https://github.com/thu-nics/C2C)
([paper](https://arxiv.org/abs/2510.03215)). Receiver **Qwen3-0.6B**, sharer **Qwen3-4B-Base**
(frozen), fuser = the released `nics-efc/C2C_Fuser` projector, which is trained on general chat
distillation and never sees retrieval data — so every result here is zero-shot with respect to
fuser training.

## Metrics

Four instruments, all reproducible from this repository with **no API in the loop**:

| metric | what it is | bias |
|---|---|---|
| **phi-4** | primary judge: `microsoft/phi-4`, local, greedy, deterministic | roughly neutral |
| **Gemma-2** | second judge, different family: `google/gemma-2-9b-it`, same prompt | more lenient overall |
| **contains** | a gold alias appears in the raw generation; no model in the loop | length-permissive |
| **EM / F1** | against a uniform answer extraction applied to every system | length-penalising |

Both judges read the raw unmodified generation and are blind to which system produced it. All
metrics score against the dataset's **full** alias list.

Two metrics were tested and **rejected**; the scripts are kept so the checks are reproducible.

* **An API judge is not reproducible.** Re-judging byte-identical generations with Claude Haiku
  flipped 8–9% of verdicts and moved a paired delta from +5.3 (p=0.004) to +2.8 (n.s.). Available
  via `judge_trivia.py`, reported only where explicitly requested.
* **BERTScore measures length here.** Under the CAG protocol (arXiv:2412.15605) it ranks the 4B
  sharer *below* the 0.6B receiver in every variant tested (F1/recall × raw/extracted), and
  within every cell it correlates more strongly with output length (ρ = −0.37 to −0.53) than with
  correctness (ρ = +0.29 to +0.39). With a 1–5 token gold span its precision term dominates.
  See `score_bertscore.py`.

## Results

Six eval sets, one question per gold passage so questions are independent units. Retrieval is
bge-small top-1.

### SQuAD v1.1 — n=800 over 800 distinct paragraphs, top-1 hit 68.1%

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver only, no passage | 7.1 | 16.9 | 5.1 | 2.4 | 7.4 |
| sharer only, no passage | — | — | 12.2 | 1.8 | 9.1 |
| *retrieved* | | | | | |
| receiver text-RAG *(the baseline that matters)* | 51.8 | 59.2 | 48.6 | 25.8 | 6.9 |
| + offline sharer distillation *(cost-matched text)* | 37.8 | 46.4 | 36.8 | 22.5 | 6.8 |
| receiver-own-KV control *(same pipeline, fusion off)* | 52.1 | 59.8 | 48.8 | 26.2 | 6.9 |
| CacheGraft, inline | 59.6 | 62.5 | 55.5 | 16.9 | 12.4 |
| **CacheGraft, persisted cache (4B fully offline)** | **60.0** | **63.1** | **56.2** | 17.1 | 12.4 |
| sharer text-RAG *(ceiling)* | 66.4 | 70.8 | 61.1 | 4.5 | 10.9 |
| *oracle* | | | | | |
| receiver text-RAG | 73.0 | 78.6 | 69.9 | 38.8 | 6.4 |
| + offline sharer distillation | 52.4 | 59.8 | 50.9 | 31.9 | 6.5 |
| receiver-own-KV control | 72.5 | 77.9 | 69.5 | 39.0 | 6.3 |
| CacheGraft, inline | 79.4 | 81.8 | 76.9 | 24.9 | 11.8 |
| **CacheGraft, persisted cache** | **79.6** | **82.1** | **77.4** | 25.4 | 11.7 |
| sharer text-RAG *(ceiling)* | 86.5 | 89.4 | 81.2 | 5.6 | 9.5 |

### AdversarialQA — n=416, top-1 hit 59.4%

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver only, no passage | 10.6 | 18.5 | 7.7 | 4.1 | 7.9 |
| sharer only, no passage | — | — | 13.7 | 0.2 | 9.3 |
| receiver text-RAG, retrieved | 30.8 | 34.9 | 24.5 | 7.2 | 9.1 |
| receiver-own-KV control | 31.0 | 34.4 | 24.0 | 7.2 | 9.1 |
| **CacheGraft, retrieved** | **36.8** | **38.9** | **30.3** | 5.5 | 14.4 |
| sharer text-RAG, retrieved | — | 48.3 | 33.7 | 3.4 | 12.7 |
| receiver text-RAG, oracle | 42.1 | 44.2 | 33.9 | 9.9 | 8.8 |
| receiver-own-KV control | 41.3 | 43.3 | 33.4 | 9.9 | 8.9 |
| **CacheGraft, oracle** | **49.3** | **51.4** | **42.5** | **9.9** | 13.9 |
| sharer text-RAG, oracle | 59.1 | 63.9 | 48.3 | 5.8 | 11.9 |

### CovidQA — n=800, top-1 hit 53.2% (within-document retrieval)

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver only, no passage | 3.7 | 13.8 | 0.9 | 0.4 | 10.4 |
| sharer only, no passage | — | — | 3.2 | 0.5 | 11.3 |
| receiver text-RAG, retrieved | 33.1 | 50.8 | 21.0 | 8.3 | 10.8 |
| receiver-own-KV control | 33.5 | 50.0 | 21.1 | 8.1 | 10.8 |
| **CacheGraft, retrieved** | **36.9** | **53.8** | 22.6 | 5.1 | 15.2 |
| receiver text-RAG, oracle | 53.7 | 71.7 | 32.6 | 12.0 | 10.7 |
| receiver-own-KV control | 53.8 | 72.5 | 32.8 | 11.5 | 10.7 |
| **CacheGraft, oracle** | **59.1** | **76.9** | 35.4 | 6.7 | 15.0 |
| sharer text-RAG, oracle *(ceiling)* | 61.4 | 80.1 | 37.4 | 3.5 | 13.2 |

### QASPER — n=583, top-1 hit 36.2% (within-document retrieval)

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver only, no passage | 1.6 | 10.3 | 2.9 | 1.3 | 10.5 |
| receiver text-RAG, retrieved | 15.3 | 35.9 | 17.7 | 4.5 | 9.7 |
| receiver-own-KV control | 15.5 | 36.3 | 17.3 | 4.3 | 9.7 |
| CacheGraft, retrieved | 15.7 | 33.9 | 18.8 | 2.9 | 14.5 |
| receiver text-RAG, oracle | 34.7 | 57.4 | 32.7 | 12.5 | 8.9 |
| receiver-own-KV control | 34.7 | 57.9 | 33.0 | 12.5 | 8.9 |
| CacheGraft, oracle | 37.5 | 60.8 | 39.0 | 7.6 | 14.4 |
| sharer text-RAG, oracle *(ceiling)* | 37.4 | 59.0 | 34.5 | 3.1 | 14.9 |

### TriviaQA — n=800, 34,064 chunks, top-1 hit 54.3%

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver only, no passage | 15.0 | 24.6 | 18.8 | 9.5 | 6.5 |
| sharer only, no passage | — | — | 40.8 | 5.2 | 7.6 |
| receiver text-RAG, retrieved | 38.1 | 49.2 | 45.9 | 27.4 | 5.8 |
| receiver-own-KV control | 37.9 | 48.5 | 45.5 | 26.6 | 5.8 |
| **CacheGraft, retrieved** | **41.6** | **52.4** | **48.9** | 19.0 | 11.4 |
| sharer text-RAG, retrieved *(ceiling)* | 57.8 | 74.1 | 65.5 | 5.8 | 7.5 |
| receiver text-RAG, oracle | 50.8 | 62.9 | 61.0 | 40.0 | 5.4 |
| receiver-own-KV control | 51.2 | 62.8 | 60.9 | 40.0 | 5.4 |
| CacheGraft, oracle | 54.0 | 63.4 | 62.8 | 27.6 | 11.2 |

### TriviaQA-500 — n=500, top-1 hit 53.0% (replication of a smaller sampling of the same corpus)

| system | phi-4 | Gemma | contains | EM | words |
|---|---:|---:|---:|---:|---:|
| receiver text-RAG, retrieved | 34.8 | 45.6 | 41.0 | 26.0 | 6.2 |
| **CacheGraft, retrieved** | **40.6** | **50.2** | **44.0** | 19.6 | 11.6 |
| receiver text-RAG, oracle | 48.0 | 59.8 | 56.2 | 38.0 | 5.7 |
| **CacheGraft, oracle** | **53.2** | 61.0 | **60.0** | 29.0 | 11.5 |

Kept as a sampling-sensitivity check: the same corpus sampled at n=500 gives +5.8/+5.2 (phi-4)
where n=800 gives +3.5/+3.2. Effect sizes on TriviaQA are sample-dependent; report intervals.

### Paired significance (CacheGraft vs receiver text-RAG)

Exact two-sided McNemar over questions. **21 of 22** judge × dataset × setting deltas are
positive; 11 are significant at p<0.05.

| dataset | setting | phi-4 | Gemma | contains |
|---|---|---|---|---|
| AdversarialQA | retrieved | +6.0 p=.010 | +4.1 p=.093 | +5.8 p=.006 |
| AdversarialQA | oracle | +7.2 p=.007 | +7.2 p=.007 | +8.7 p=4e-4 |
| SQuAD | retrieved | +7.9 p=2e-7 | +3.2 p=.037 | +6.9 p=4e-6 |
| SQuAD | oracle | +6.4 p=1e-4 | +3.1 p=.061 | +7.0 p=4e-5 |
| CovidQA | retrieved | +3.8 p=.011 | +3.0 p=.062 | +1.6 n.s. |
| CovidQA | oracle | +5.4 p=.002 | +5.2 p=.001 | +2.8 p=.084 |
| TriviaQA | retrieved | +3.5 p=.021 | +3.1 p=.041 | +3.0 p=.028 |
| TriviaQA | oracle | +3.2 p=.051 | +0.5 n.s. | +1.8 n.s. |
| QASPER | retrieved | +0.4 n.s. | −2.0 n.s. | +1.1 n.s. |
| QASPER | oracle | +2.9 n.s. | +3.4 n.s. | +6.3 p=5e-4 |

Magnitude is grader-dependent — Gemma is systematically more lenient, which compresses deltas —
so the effect is best stated as a range, **+3 to +8 depending on grader**, rather than a point
estimate. Inter-judge Cohen's κ is 0.79 (SQuAD), 0.70–0.74 (AdversarialQA), 0.61–0.62 (CovidQA),
0.67–0.70 (TriviaQA) and 0.47 (QASPER); it tracks gold-span length, not effect size.

### Gap recovery

How much of the receiver→sharer headroom CacheGraft captures at receiver-only query cost:

| dataset | receiver text-RAG | sharer text-RAG | CacheGraft | recovered |
|---|---:|---:|---:|---:|
| CovidQA (oracle) | 53.7 | 61.4 | 59.1 | **70%** |
| SQuAD (retrieved) | 51.8 | 66.4 | 60.0 | **52%** |
| AdversarialQA (oracle) | 42.1 | 59.1 | 49.3 | **42%** |
| TriviaQA (retrieved) | 38.1 | 57.8 | 41.6 | 18% |

QASPER has essentially no headroom to recover (sharer 37.4 vs receiver 34.7, a 2.7-point
ceiling), which is why it is flat there rather than negative — CacheGraft reaches the ceiling.

### Three robustness results

* **The receiver-own-KV control is null everywhere.** The identical two-phase cache pipeline with
  fusion switched off lands on the text baseline in every dataset and setting. The gain is
  attributable to *whose* KV fills the document section, not to the caching machinery.
* **Spending the sharer's offline budget on text is worse than not spending it.** The same 4B, run
  once per passage offline and question-agnostically, writing text instead of KV: 37.8 vs 51.8
  (retrieved) and 52.4 vs 73.0 (oracle) on SQuAD. A query-agnostic summary must be written before
  the query is known, so it discards the span some future question wanted.
* **Verbosity is not the explanation.** Fused output runs ~2× longer, which costs EM. Under a
  matched short-answer protocol applied identically to both channels, CacheGraft wins EM (28.1 vs
  23.1), F1 (43.5 vs 36.7) and contains (+13.2, p=7e-15) on SQuAD — and still beats the text
  channel's *own best* configuration on all three. On AdversarialQA oracle, where gold spans are
  longer, EM is neutral (9.9 vs 9.9) with no protocol change at all.

### A fuser-free screen predicts amenability

Four batched cells per dataset, no fusion and no projector:

```
encoding headroom = sharer/oracle - text_rag/oracle    # does the receiver under-encode?
answer-in-doc     = sharer/oracle - sharer/none        # is the answer in the passage at all?
```

| dataset | encoding headroom | answer-in-doc | fusion gain (oracle, phi-4) |
|---|---:|---:|---:|
| AdversarialQA | **+14.4** | +34.6 | +7.2 |
| SQuAD | +11.3 | +69.0 | +6.4 |
| CovidQA | +4.8 | +34.2 | +5.4 |
| QASPER | +1.8 | +29.6 | +2.9 n.s. |
| TriviaQA | +19.6 | **+24.7** ← fails | +3.2 |

Both preconditions have to hold. TriviaQA has the most encoding headroom of any dataset here and
still captures the least of it, because its answers live in the sharer's parameters rather than
in the retrieved passage. Screen terms are `contains`, oracle context; TriviaQA's are computed at
retrieved (no sharer/oracle cell).

## Setup

```bash
# 1. a virtualenv is required -- the system Python will not have these deps, and every
#    command below assumes this environment is active
python3 -m venv .venv
source .venv/bin/activate

# 2. the C2C package supplies the `rosetta` library and RosettaModel
git clone https://github.com/thu-nics/C2C
pip install -e C2C --no-deps          # --no-deps keeps your existing torch
# if the editable install fails (older setuptools), put C2C on the path instead:
#   export PYTHONPATH=$PWD/C2C        # rosetta is pure Python
pip install -r requirements.txt

# 3. every command in this README is run from scripts/ with the venv active
cd scripts
export PYTHONPATH=$(cd .. && pwd)/C2C USE_TF=0
```

* Set `USE_TF=0` (scripts do it internally; export it if importing them elsewhere).
* `jinja2>=3.1` is required — older jinja silently makes `apply_chat_template` skip samples.
* Models and the fuser download from the Hub on first run (~10 GB).
* A CUDA GPU is required; everything fits on one **A100-40GB** (peak ~12 GB). Greedy decoding
  throughout, with `common.set_determinism()` (seed 0) called by both runners.
* The judges download `microsoft/phi-4` (~28 GB) and `google/gemma-2-9b-it` (~18 GB, **gated** —
  accept the licence on the Hub and `huggingface-cli login`). Each wants the GPU to itself.
* **Run everything from `scripts/`.**

## Reproducing the results

### 1. Build the eval sets

Two design choices are enforced because they change the statistics:

* **One question per gold passage.** SQuAD-style corpora ask many questions about the same
  passage (AdversarialQA: 3,000 questions over 416 contexts), and scoring all of them treats
  correlated items as independent. One question per gold passage makes the question the
  independent unit, so no cluster correction is needed.
* **Chunking long contexts** (`--chunk_words`). CovidQA and QASPER ship whole papers; chunking
  keeps passage length, encoding burden and KV footprint comparable across datasets. For those
  two, `--scope document` restricts retrieval to the question's own paper — the realistic setting
  for QA over a known document, and what makes paragraph-level retrieval tractable at all
  (QASPER top-1 goes from 4.8% to 36.2%, CovidQA from 32.5% to 53.2%).

```bash
python prep.py --data squad         --target 800
python prep.py --data adversarialqa --target 800                                    # yields 416
python prep.py --data covidqa       --target 800 --chunk_words 140 --scope document
python prep.py --data qasper        --target 800 --chunk_words 140 --scope document # yields 583
python trivia_prep.py --target 800 --scan 4000
python trivia_prep.py --target 500 --scan 2000 --out trivia500
```

Expected: `squad` 800 q / 2067 chunks, `adversarialqa` 416/416, `covidqa` 800/4295,
`qasper` 583/6460, `trivia` 800/34064, `trivia500` 500/21812. `prep.py --data squad --target 800`
is deterministic and regenerates the exact eval set these numbers were produced on.

Prepending the document title/abstract to each chunk (`--header`) was tested and **hurt**
retrieval in all four comparisons, because a shared prefix makes chunks of the same document
harder to tell apart — which is exactly the discrimination within-document retrieval needs.

### 2. Run the matrix

One command per dataset: the four fuser-free screen cells, then the text-RAG baseline,
CacheGraft doc-only fusion, the receiver-own-KV control and the sharer ceiling, at both
`retrieved` and `oracle`.

```bash
./run_matrix.sh squad         800
./run_matrix.sh adversarialqa 416
./run_matrix.sh covidqa       800
./run_matrix.sh qasper        583
./run_matrix.sh trivia        800
./run_matrix.sh trivia500     500
```

Roughly 40 min per dataset on an A100; the fusion cells decode one example at a time. Budget ~4 h for step 2 across all six datasets, ~45 min for step 3, and ~3 h per judge for step 4 (both judges over all six datasets), so a full reproduction is roughly a 10-hour unattended run on a single A100-40GB.

### 3. Add the two SQuAD-only baselines

The cost-matched offline **text** channel, and the persisted two-phase cache that keeps the 4B
off the query path entirely:

```bash
python trivia_distill.py --data squad --style detailed --tag _strong --n_test 800
python rag_trivia.py --data squad --method text_rag --fact retrieved \
                     --distill summary --distill_tag _strong --n_test 800
python rag_trivia.py --data squad --method text_rag --fact oracle    \
                     --distill summary --distill_tag _strong --n_test 800

python rag_trivia_cached.py --data squad --fact retrieved --n_test 800
python rag_trivia_cached.py --data squad --fact oracle    --n_test 800
```

The persisted path reproduces the inline one to 91–94% token-exact generations, the residual
being floating-point nondeterminism under greedy decoding. It writes ~18 MB per passage, so
budget ~15 GB per condition and delete `squad_kvcache/` afterwards.

The matched short-answer protocol (the length-controlled, judge-free comparison) adds `--terse`.
It must be applied to *every* method or the comparison is not fair:

```bash
python rag_trivia.py --data squad --method text_rag --fact retrieved --terse --n_test 800
python rag_trivia.py --data squad --method text_rag --fact oracle    --terse --n_test 800
python rag_trivia.py --data squad --method c2c --fact retrieved --fuse doc_only --terse --n_test 800
python rag_trivia.py --data squad --method c2c --fact oracle    --fuse doc_only --terse --n_test 800
```

These cells are scored lexically (EM / F1 / contains) and need no judge.

### 4. Judge and report

SQuAD carries the full ladder, so it has more judged cells than the others. Both lists below
cover every judged number reported in the paper.

```bash
SQUAD_CELLS="receiver_none text_rag_retrieved text_rag_retrieved_distill_strong \
recvkv_retrieved_doc_only c2c_retrieved_doc_only c2c_retrieved_doc_only_cached sharer_retrieved \
text_rag_oracle text_rag_oracle_distill_strong recvkv_oracle_doc_only c2c_oracle_doc_only \
c2c_oracle_doc_only_cached sharer_oracle"

OTHER_CELLS="receiver_none text_rag_retrieved recvkv_retrieved_doc_only c2c_retrieved_doc_only \
sharer_retrieved text_rag_oracle recvkv_oracle_doc_only c2c_oracle_doc_only sharer_oracle"

for M in "microsoft/phi-4 phi-4" "google/gemma-2-9b-it gemma2"; do
  set -- $M
  python judge_open.py --data squad --model "$1" --short "$2" --batch 8 $SQUAD_CELLS
  for D in adversarialqa covidqa qasper trivia trivia500; do
    python judge_open.py --data $D --model "$1" --short "$2" --batch 8 $OTHER_CELLS
  done
done

python report_table.py --data squad --baseline text_rag_retrieved --cells $SQUAD_CELLS
python report_table.py --data squad --baseline text_rag_oracle    --cells $SQUAD_CELLS
```

`report_table.py` prints per-cell metrics, paired exact McNemar, question-level bootstrap CIs and
inter-judge Cohen's κ. Use `--batch 8` on TriviaQA, whose long alias lists will otherwise OOM at
40 GB.

### 5. Verify a re-run against a reference run

Every number above should survive a clean re-run of the commands in this section. To check that,
keep the previous results directory and compare:

```bash
mv squad_results squad_results.v1        # before re-running
# ... re-run steps 1-4 ...
python check_repro.py --data squad --ref squad_results.v1
```

`check_repro.py` compares cell by cell at three levels: byte-identity of the generations, the
metrics computed from them, and per-judge verdict agreement. Both judges decode greedily, so an
identical generation must yield an identical verdict — drift there is a bug, not noise.

**Measured result of a full clean-slate re-run** (5.5 h, all six datasets, both judges):

| level | result |
|---|---|
| eval sets | all six rebuild **byte-identically**, including the two that stream TriviaQA |
| generations | **100.0% byte-identical** across all 68 cells |
| contains / EM / F1 | identical to **0.0 points** |
| judge verdicts | **0 flips / 10,400** with `set_determinism()`; 19 / 67,554 without it |

Exact reproduction depends on `common.set_determinism()`, which is called by both runners *and*
by `judge_open.py`. Without it in the judge, greedy judging still drifts on ~0.03% of verdicts
because CUDA reductions are not bitwise-stable across processes. Note this is reproduction of the
*same* configuration; the two different implementations of doc-only fusion (inline vs persisted
cache) agree to 93–94% token-exact, which is expected and correctness-preserving.

### Reading the screen

```bash
python report_table.py --data covidqa \
  --cells receiver_none sharer_none text_rag_oracle sharer_oracle
```

`sharer_oracle − text_rag_oracle` is encoding headroom; `sharer_oracle − sharer_none` is
answer-in-document. Both need to be large.

## How doc-only fusion works

`RosettaModel` applies fusion **per section** of `kv_cache_index`, keyed on each section's first
token. Doc-only fusion therefore uses three sections: `[doc → fuse, question → no-fuse,
generation]`. Because attention is causal, fusing the document in a single forward pass with the
question appended un-fused is **mathematically identical** to loading an offline-built,
question-agnostic fused cache — so for top-1 retrieval with a fixed preamble there is no RoPE
mismatch and no position handling is required.

`rag_trivia_cached.py` implements that as a real two-phase pipeline. Phase 1 prefills
`[preamble+doc]` with fusion and persists the receiver-space KV, keyed by a content hash of the
document. Phase 2 loads it into the **base** receiver and decodes the question with no sharer and
no projector on the query path. `--recv_kv` runs the same pipeline with fusion off — the control.

Cost: offline build 0.05–0.19 s and ~18 MB (fp16) per ~130-token passage, once per passage;
online 0.58 s per query, receiver-only.

## Limitations

* **Storage.** ~18 MB per passage, ~2.6×10⁴ times the passage text. The cheap text alternatives
  are *worse*, not merely cheaper, so the trade is real — but it is a trade, and KV quantization
  is not yet measured.
* **Scale and scope.** 0.6B receiver, top-1 retrieval, n≤800 per cell, greedy decoding. The
  Llama-1B←8B pair suggests gains shrink as the receiver gets stronger.
* **Multi-chunk.** Fused keys are not RoPE-re-rotatable, so top-k composition needs a cache per
  (chunk, slot) pair. Multi-hop is a separate failure: fusion supplies per-passage knowledge, not
  cross-passage reasoning.
* **Effect size depends on the grader and on the sample.** Gemma reports roughly half of phi-4's
  delta on SQuAD, and TriviaQA moves by ~2.5 points between samples of the same corpus. Report
  ranges and intervals, not point estimates.
* **No headroom, no gain.** Where the sharer is barely better than the receiver at reading the
  passage (QASPER: 37.4 vs 34.7), there is nothing for fusion to transfer.

## Repo layout

```
scripts/
  common.py             # models, fuser resolution, per-dataset paths, determinism, retrieval
  prep.py               # eval-set builder: squad | covidqa | adversarialqa | qasper
  trivia_prep.py        # TriviaQA passage corpus + embeddings (--out for a second setting)
  run_matrix.sh         # screen + full matrix for one dataset
  rag_trivia.py         # core runner (--data, text_rag/c2c/receiver/sharer, doc_only/doc_q, --terse)
  rag_trivia_cached.py  # persisted two-phase pipeline + the receiver-own-KV control
  kvcache.py            # KV primitives: build / save / load / decode
  trivia_distill.py     # offline sharer summaries (cost-matched text channel)
  judge_open.py         # the two judges: any local open-weights model (phi-4, gemma-2)
  check_repro.py        # compare a re-run against a reference results dir (generations/metrics/judges)
  judge_trivia.py       # rejected: Claude Haiku via CLI (8-9% verdict instability)
  score_bertscore.py    # rejected: BERTScore (length-dominated on short-answer QA)
  report_table.py       # metrics + paired McNemar + bootstrap CIs + inter-judge kappa
  eval_llama.sh         # evaluate a locally trained fuser (Llama pair) via env vars
recipes/                # fuser training recipes for Llama-3.2-1B <- Llama-3.1-8B
```

Generated data, caches and results are gitignored; regenerate by re-running.
`docs/NOTES.md` is the working log.
