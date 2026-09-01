# CacheGraft — precomputed cross-model KV enrichment for RAG

Standard RAG hands a retrieved passage to the reader as **text**, so a small reader must
re-encode that passage with its own limited parameters on every query. CacheGraft delivers the
passage as a **larger model's KV-cache representation of it**, fused into the small model's cache
by a trained projector.

The restriction that makes this deployable: fuse **only the document section** of the prompt and
leave the question as ordinary text. By causality the fused document cache then depends on the
document alone — it is **query-independent**, so it can be computed once per passage offline and
reused for every future query that retrieves it. At query time only the small receiver runs.

Built on [Cache-to-Cache (C2C)](https://github.com/thu-nics/C2C)
([paper](https://arxiv.org/abs/2510.03215)). Receiver = **Qwen3-0.6B**, sharer =
**Qwen3-4B-Base** (frozen), fuser = the released `nics-efc/C2C_Fuser` projector. The fuser is
trained on general chat distillation and never sees retrieval data, so every result here is
zero-shot with respect to fuser training.

## Headline result

SQuAD v1.1, n=800 questions over **800 distinct paragraphs**, top-1 retrieval, scored by two
independent judge families plus a metric with no model in the loop.

| system | Haiku | phi-4 | contains |
|---|---:|---:|---:|
| receiver only, no passage | 7.4 | 7.1 | 5.1 |
| receiver text-RAG *(the baseline that matters)* | 55.5 | 51.8 | 48.6 |
| + offline sharer distillation *(cost-matched text channel)* | 42.2 | 37.8 | 36.8 |
| receiver-own-KV control *(same pipeline, fusion off)* | 55.2 | 52.1 | 48.8 |
| **CacheGraft, persisted cache (4B fully offline)** | **62.0** | **60.0** | **56.2** |
| sharer text-RAG *(ceiling)* | 67.9 | 66.4 | 61.1 |

Δ vs baseline: **+6.5** (p=1.8e-5), **+8.2** (p=2.8e-8), **+7.6** (p=1.9e-7) — recovering 52–61%
of the receiver→sharer gap at receiver-only query cost. With oracle passages: +6.8 / +6.6 / +7.5.
The control is null on every metric (−0.2 / +0.4 / +0.1), so the gain is attributable to *whose*
KV fills the document section, not to the caching machinery. Inter-judge Cohen's κ = 0.78–0.88,
and does not differ between the verbose fused cell and the terse text cell.

Two further findings:

* **Spending the sharer's offline budget on text is worse than not spending it.** A
  query-agnostic summary must be written before the query is known, so it discards the span some
  future question wanted: −13.2 / −19.2 judge points against raw text.
* **Verbosity is not the explanation.** Fused output is ~2× longer, which costs EM (−8.6,
  p=2e-9). Under a matched short-answer protocol applied identically to both channels,
  CacheGraft wins EM (28.1 vs 23.1), F1 (43.5 vs 36.7) and contains (+13.2, p=7e-15) — and still
  beats the text channel's *own best* configuration on all three.

## Scope — where it does not work

| dataset | setting | Δ vs receiver text-RAG |
|---|---|---|
| SQuAD | top-1 retrieved / oracle | **+6.5 / +6.8** (p<1e-4) |
| TriviaQA | top-1 retrieved | **+4.0** (p=0.006) |
| TriviaQA | oracle | +1.8 (n.s.) |
| OpenBookQA | fact-only fusion (cacheable) | −0.2 / −1.8 (n.s.) |
| OpenBookQA | full-prompt fusion (**not** cacheable) | +16.6 / +12.6 (p<1e-8) |

OpenBookQA is the informative negative: fusing a 15-token science fact query-independently gives
nothing, and the large OpenBookQA gain reported for C2C-style fusion requires fusing the
**question**, which precomputation forbids. The gain tracks the *answer-in-document* term — how
much of the answer lives in the passage rather than in the sharer's parameters — rather than raw
encoding headroom: TriviaQA has more headroom than SQuAD (+19.6 vs +11.3) and gets less than half
the gain.

## Setup

```bash
git clone https://github.com/thu-nics/C2C
pip install -e C2C --no-deps          # or: export PYTHONPATH=$PWD/C2C  (rosetta is pure Python)
pip install -r requirements.txt
```

* Set `USE_TF=0` (the scripts do it internally; export it if importing them elsewhere).
* `jinja2>=3.1` is required — older jinja silently makes `apply_chat_template` skip samples.
* Models and the fuser download from the Hub on first run (~10 GB).
* A CUDA GPU is required; everything fits on one **A100-40GB** (peak ~12 GB). Greedy decoding.
* Run everything from `scripts/`.

## Building an eval set

`prep.py` builds any supported reading-comprehension dataset into the shared schema. Two design
choices are enforced because they affect the statistics:

* **One question per source passage.** SQuAD-style corpora ask many questions per paragraph;
  scoring all of them treats correlated items as independent and lets a few passages dominate.
  One question per passage makes the question the independent unit — no cluster correction
  needed.
* **Chunking long contexts** (`--chunk_words`). CovidQA and QASPER ship whole papers; chunking
  keeps passage length, encoding burden and KV footprint comparable across datasets.

```bash
python prep.py --data squad         --target 800                    # 800 paragraphs, 2067-doc pool
python prep.py --data covidqa       --target 800 --chunk_words 140
python prep.py --data adversarialqa --target 800
python prep.py --data qasper        --target 800 --chunk_words 140
python trivia_prep.py --target 800 --scan 4000                      # TriviaQA builds its own corpus
```

## Running the experiments

Every runner is dataset-generic via `--data`.

```bash
# reference points
python rag_trivia.py --data squad --method receiver --fact none --n_test 800   # parametric floor
python rag_trivia.py --data squad --method sharer   --fact none --n_test 800   # answer-in-doc term
python rag_trivia.py --data squad --method sharer   --fact oracle --n_test 800 # ceiling

# the baseline that carries the claim: the same receiver reading the same passage as text
python rag_trivia.py --data squad --method text_rag --fact retrieved --n_test 800
python rag_trivia.py --data squad --method text_rag --fact oracle    --n_test 800

# cost-matched offline TEXT channel: same sharer, once per passage, question-agnostic
python trivia_distill.py --data squad --style detailed --tag _strong --n_test 800
python rag_trivia.py --data squad --method text_rag --fact retrieved \
                     --distill summary --distill_tag _strong --n_test 800

# CacheGraft, inline (mathematically equivalent to the offline cache)
python rag_trivia.py --data squad --method c2c --fact retrieved --fuse doc_only --n_test 800
python rag_trivia.py --data squad --method c2c --fact retrieved --fuse doc_q    --n_test 800  # upper bound

# CacheGraft, persisted two-phase pipeline: build caches to disk, then decode receiver-only
python rag_trivia_cached.py --data squad --fact retrieved --n_test 800
# CONTROL: identical pipeline, cache built from the receiver's OWN KV (fusion off)
python rag_trivia_cached.py --data squad --fact retrieved --recv_kv --n_test 800

# matched short-answer protocol (adds --terse to every method, for the no-judge comparison)
python rag_trivia.py --data squad --method text_rag --fact retrieved --terse --n_test 800
python rag_trivia.py --data squad --method c2c --fact retrieved --fuse doc_only --terse --n_test 800
```

OpenBookQA is separate because it is multiple-choice (accuracy, so no judge and no length
confound). `--fuse fact_only` is the cacheable configuration; `--fuse all` is C2C's original,
non-precomputable one.

```bash
python rag_obqa.py --method text_rag --fact retrieved
python rag_obqa.py --method c2c_rag  --fact retrieved --fuse fact_only
python rag_obqa.py --method c2c_rag  --fact retrieved --fuse all
```

## Scoring

Two independent judge families, both scored on the **raw unmodified generations** and blind to
which system produced each answer, plus lexical metrics.

```bash
python judge_trivia.py --data squad --model haiku text_rag_retrieved c2c_retrieved_doc_only_cached
python judge_open.py   --data squad --model microsoft/phi-4 --short phi-4 \
                       text_rag_retrieved c2c_retrieved_doc_only_cached

python report_table.py --data squad --baseline text_rag_retrieved \
  --cells receiver_none text_rag_retrieved text_rag_retrieved_distill_strong \
          recvkv_retrieved_doc_only c2c_retrieved_doc_only_cached sharer_retrieved
```

`report_table.py` prints per-cell metrics, paired exact McNemar, question-level bootstrap CIs and
inter-judge Cohen's κ. All metrics use the dataset's full alias list.

## How doc-only fusion works

`RosettaModel` applies fusion **per section** of `kv_cache_index`, keyed on each section's first
token. Doc-only fusion therefore uses three sections: `[doc → fuse, question → no-fuse,
generation]`. Because attention is causal, fusing the document in a single forward pass with the
question appended un-fused is **mathematically identical** to loading an offline-built,
question-agnostic fused cache — so for top-1 retrieval with a fixed preamble there is no RoPE
mismatch and no position handling is required.

`rag_trivia_cached.py` implements that as a real two-phase pipeline: phase 1 prefills
`[preamble+doc]` with fusion and persists the receiver-space KV, keyed by a content hash; phase 2
loads it into the **base** receiver and decodes the question with no sharer and no projector on
the query path. The two paths agree to 91–94% token-exact generations, the residual being
floating-point nondeterminism under greedy decoding.

Cost: offline build 0.05–0.19 s and ~18 MB (fp16) per ~130-token passage, once per passage;
online 0.58 s per query, receiver-only.

## Limitations

* **Storage.** ~18 MB per passage, ~2.6×10⁴ times the passage text. The cheap text alternatives
  are *worse*, not merely cheaper, so the trade is real — but it is a trade, and KV quantization
  is not yet measured.
* **Scale and scope.** 0.6B receiver, top-1 retrieval, n≤800 per cell, greedy decoding, clean
  topically-focused retrieval pools. The Llama-1B←8B pair suggests gains shrink as the receiver
  gets stronger.
* **Multi-chunk.** Fused keys are not RoPE-re-rotatable, so top-k composition needs a cache per
  (chunk, slot) pair. Multi-hop is a separate failure: fusion supplies per-passage knowledge, not
  cross-passage reasoning.
* **Effect sizes are sample-dependent on TriviaQA.** The same 500 questions re-run in a separate
  process moved the oracle delta by 2.6 points; report intervals, not point estimates.

## Repo layout

```
scripts/
  common.py             # models, fuser resolution, per-dataset paths
  prep.py               # eval-set builder: squad | covidqa | adversarialqa | qasper
  trivia_prep.py        # TriviaQA passage corpus + embeddings
  rag_trivia.py         # core runner (--data, text_rag/c2c/receiver/sharer, doc_only/doc_q, --terse)
  rag_trivia_cached.py  # persisted two-phase pipeline + the receiver-own-KV control
  kvcache.py            # KV primitives: build / save / load / decode
  trivia_distill.py     # offline sharer summaries (cost-matched text channel)
  rag_obqa.py           # OpenBookQA (multiple-choice; --fuse fact_only | all)
  judge_trivia.py       # judge A: Claude Haiku via the `claude` CLI
  judge_open.py         # judge B: a local open-weights model (default microsoft/phi-4)
  report_table.py       # metrics + paired McNemar + bootstrap CIs + inter-judge kappa
  eval_llama.sh         # evaluate a locally trained fuser (Llama pair) via env vars
recipes/                # fuser training recipes for Llama-3.2-1B <- Llama-3.1-8B
```

Generated data, caches and results are gitignored; regenerate by re-running.
`docs/NOTES.md` is the working log.
