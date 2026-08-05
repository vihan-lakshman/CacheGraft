# Project notes / session handoff

Context to resume this work on a fresh machine. (The README has the polished reproduction
instructions; this file is the "what we did, why, and what's next" log.)

## Goal

Take Cache-to-Cache (C2C) "beyond the paper" into **retrieval-augmented generation**: fuse a
large sharer model's KV-cache *representation* of retrieved context into a small receiver model,
instead of feeding the receiver the retrieved *text*. Receiver = Qwen3-0.6B, Sharer =
Qwen3-4B-Base (frozen), Fuser = released `nics-efc/C2C_Fuser :: qwen3_0.6b+qwen3_4b_base_Fuser`.

## Distilled-text baseline CLOSED + paper draft started (2026-08-04)  <- RESUME HERE

**Session goal:** close the one hole that could have sunk the systems claim — the *cost-matched
offline-sharer text baseline* — and start the workshop paper.

### Naming
The system is now **CacheGraft** (`turborag-fusion` was a placeholder; TurboRAG-style position
rearrangement is not used on the headline path, so the old name was misleading). Repo/README
rename NOT yet done — only the paper draft uses the new name (via a `\sys` macro, so it is a
one-line change if you want a different one).

### Paper draft
- `paper/main.tex` — full ~8-page workshop draft (abstract, method incl. the query-independence
  proposition, TikZ two-phase figure, results, scope/screen section, limitations).
- ⚠️ **`main.tex` NUMBERS ARE STALE.** It was written *before* this session's run and still
  carries the old README figures (+7.1/+12.9) with `\todo{}` placeholders where the distilled
  rows go. **Do the update pass before reading it as truth** — the correct numbers are below.
- ⚠️ **`paper/refs.bib` does not exist yet**, so the draft does not compile. Citation keys used
  by `main.tex`: c2c2025, lu2025turborag, yao2025cacheblend, gim2024promptcache, lewis2020rag,
  jiang2023llmlingua, rajpurkar2016squad, joshi2017triviaqa, mihaylov2018obqa, yang2018hotpotqa,
  cobbe2021gsm8k, kocisky2018narrativeqa, jin2019pubmedqa, clark2019boolq, xiao2023cpack, qwen3,
  llama3, zheng2023judge, liu2024kivi, hinton2015distill, huh2024platonic.

### Fresh-machine environment (this worked end-to-end; A100-40GB)
System python was unusable (transformers 5.x, jinja2 3.0.3, numpy 1.21). Use a venv:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/thu-nics/C2C && .venv/bin/pip install -e C2C --no-deps
cd scripts && USE_TF=0 ../.venv/bin/python <script>          # run everything this way
```
Resulting versions: py 3.10.12, torch 2.13.0+cu130, **transformers 4.52.4** (the pin that
matters), sentence-transformers 5.6.1, datasets 5.0.1, accelerate 1.14.0, **jinja2 3.1.6**,
numpy 2.2.6. `pip install -e C2C --no-deps` succeeded (no setuptools problem this time).

### Harness changes (committed)
- `trivia_distill.py` — now **dataset-generic** (`--data`), and distills only the passages the
  eval actually reads (`--subset used`, default) instead of the whole corpus: 54 passages for
  SQuAD instead of 2067, 23 s instead of hours. Also `--style {facts,detailed}`, `--tag`
  (writes `<data>_distilled<tag>.json`), and a **sentence-boundary trim** in `clean()`.
  *Why the trim matters:* at the old `max_new=96`, **35 % of summaries ended mid-clause**, which
  would have handicapped the baseline for a decoding-budget reason rather than a channel reason.
  With `--max_new 200` + trim it is 0 %. Re-check this if you change the distiller.
- `rag_trivia.py` — `--distill` now reads `<data>_distilled<--distill_tag>.json` (was hardcoded
  to `trivia_distilled.json`) and **errors loudly** if a needed passage was not distilled.
- `judge_stats.py` (NEW) — paired exact McNemar **on judge verdicts** (`rescore_trivia.py` only
  does lexical), plus a **passage-level cluster bootstrap**. Needed because SQuAD's 500 questions
  read only **54 distinct paragraphs** (~9 Qs/passage), so question-level p-values are
  anti-conservative. NB `judge_trivia.py` divides its printed accuracy by `len(samples)` (500)
  while the verdict dict is keyed by question string (490 unique) → its printout under-reports by
  ~2 %. **`judge_stats.py` numbers are the correct ones** and are what the README matches.

### RESULT — the distilled baseline does NOT threaten the claim; it supports it

SQuAD, Qwen3-0.6B <- Qwen3-4B-Base, released fuser, top-1 (hit 60.8 %), original (non-terse)
prompts. **LLM judge (Haiku 4.5), n=490** — the primary metric:

| channel | stored/passage | retrieved | oracle |
|---|---|--:|--:|
| **c2c doc_only (fusion)** | ~18 MB fp16 KV | **58.4** | **84.1** |
| text_rag raw passage | ~0.7 kB | 50.4 | 70.4 |
| text_rag distilled-**strong** (1.2x) | ~0.6 kB | 48.2 | 66.9 |
| text_rag distilled (1.7x) | ~0.4 kB | 40.4 | 53.3 |

Paired significance (`judge_stats.py --cluster`; cluster CI resamples the **53 passages**, not
the questions):

| A vs B | Δ | win/lose | McNemar p | passage-cluster 95 % CI | p_clu |
|---|--:|--:|--:|---|--:|
| c2c vs raw, retrieved | +8.0 | 63/24 | <0.0001 | [+2.7, +14.1] | 0.001 |
| c2c vs distill-strong, retrieved | +10.2 | 74/24 | <0.0001 | [+3.7, +17.8] | 0.001 |
| c2c vs distill, retrieved | +18.0 | 106/18 | <0.0001 | [+9.9, +27.5] | <0.001 |
| c2c vs raw, oracle | +13.7 | 94/27 | <0.0001 | [+7.5, +19.9] | <0.001 |
| c2c vs distill-strong, oracle | +17.1 | 115/31 | <0.0001 | [+10.3, +24.8] | <0.001 |
| c2c vs distill, oracle | +30.8 | 177/26 | <0.0001 | [+23.8, +38.5] | <0.001 |

Lexical (n=500, uniform copula extraction) — the SQuAD verbosity signature reproduces:

| cell | EM | F1 | contains | words |
|---|--:|--:|--:|--:|
| c2c_retrieved_doc_only | 21.0 | 35.4 | **55.6** | 7.3 |
| text_rag_retrieved | **30.8** | **41.7** | 50.8 | 4.4 |
| text_rag_retrieved_distill_strong | 29.4 | 39.3 | 47.6 | 4.5 |
| text_rag_retrieved_distill | 23.4 | 33.1 | 40.2 | 4.6 |
| c2c_oracle_doc_only | 30.8 | 49.8 | **78.4** | 6.7 |
| text_rag_oracle | **43.2** | **56.7** | 69.0 | 3.9 |
| text_rag_oracle_distill_strong | 42.4 | 54.1 | 64.6 | 4.0 |
| text_rag_oracle_distill | 31.8 | 43.1 | 51.6 | 4.2 |

**Reads:**
1. **The text rungs order MONOTONICALLY with compression** (raw 1.0x > strong 1.2x > plain 1.7x)
   on *every* metric in *both* retrieval settings. That is a dose-response, not one baseline row
   — the information-bottleneck claim measured directly. A query-agnostic summary must serve
   ~9 different questions per paragraph, so compression provably drops answer-bearing spans.
2. **Offline sharer preprocessing into TEXT is counterproductive; into KV it is worth +8/+13.7.**
   Stronger and cleaner than "we also beat summaries." Use this framing in the paper.
3. **Raw text remains the binding baseline** (the distilled rungs sit below it), so the headline
   comparison is unchanged.
4. **The clustering worry does not bite** — resampling 53 passages still excludes 0 everywhere.
5. **The win does not depend on the judge, only its size does.** On `contains` (no judge at all)
   c2c leads +6.2 retrieved / +11.2 oracle. EM is the *only* metric where c2c loses (verbosity,
   7.3 vs 4.4 words).
6. **Judge-bias bound (useful for the limitations section):** the judge's net correction over
   `contains` is +1.8 for c2c vs +0.2 for raw text (retrieved), +3.4 vs +1.2 (oracle). Attributing
   the **entire** differential to length bias still leaves **+6.4 / +11.5**. Cost $3.43 for 8 cells.
7. **Reproduction check vs the README:** c2c_retrieved 58.4 (exact), text 50.4 vs 51.2, oracle
   c2c 84.1 vs 83.7, text 70.4 vs 70.8, retr hit 0.608 vs 0.61. The prior record is trustworthy.
8. Distiller stats: passage 153 tok / 114 words; plain summary 92 tok / 68 words (1.67x); strong
   95 words (1.2x) — i.e. the text channel only approaches raw text by *not compressing*.

### ⚠️ PROVENANCE GAP — what is NOT on disk anywhere
Results are gitignored and this was a fresh machine, so **only SQuAD text/c2c/distill exists as
artifacts.** Everything else (TriviaQA, OBQA, HotpotQA, GSM8K, the `recv_kv` control, the cached
pipeline + timings, the Llama pair, the amenability screen) exists **only as numbers recorded in
this file and the README.** Before submitting, re-run so every paper number comes from one
environment. Priority order: `recv_kv` control (the key ablation) > TriviaQA distilled ladder
(the `(pending)` row) > cached-pipeline equivalence + timing > OBQA.

### Punch list to a submittable workshop paper (~3 days, ~1 day unattended GPU)
1. `paper/refs.bib` + make `main.tex` compile; fold in the numbers above. *(blocking)*
2. Rename repo/README `turborag-fusion` -> CacheGraft. *(blocking)*
3. TriviaQA distilled ladder + judge, mirroring SQuAD. *(blocking — second complete dataset)*
4. Re-run `recv_kv` control, cached equivalence + timing, OBQA. *(blocking — provenance)*
5. Judge validation: second judge family (~$5) and/or ~150 human labels. *(strongly advised)*
6. KV quantization (int8/int4) — turns the storage limitation into a measured tradeoff. *(high value)*
Deliberately OUT of scope for the workshop version: 7B receiver, distractor robustness, linear
answer-probe, RAG-aware fuser, position-decoupled projector. Those are the conference items.

### Standing risk to state in the paper ourselves
All numbers use 0.6B/1B receivers and small, topically-clean retrieval pools. Our own Llama-pair
finding (a competent 1B-Instruct reader shows a smaller gain) predicts the effect **shrinks as
receivers get stronger**. Say it before a reviewer does.

## Train-our-own fuser: Llama-1B-Instruct <- Llama-8B-Instruct (2026-06-19)

First fuser **we trained** (not a released checkpoint), on a **new, larger-gap, different-family
pair** — tests both "can we train fusers" and "does the RAG-fusion finding generalize."

**Setup.** base/receiver = `meta-llama/Llama-3.2-1B-Instruct` (16 layers, head_dim 64, 8 KV heads),
teacher/sharer = `meta-llama/Llama-3.1-8B-Instruct` (32 layers, head_dim 128, 8 KV heads). Same
128256 Llama-3 vocab on both -> `is_do_alignment: false` (no token aligner; the easy case).
`last_aligned` mapping (target t -> source 16+t), 16 C2CProjectors, 268M trainable (2.82%).
Recipe: `recipes/llama_1b8b_c2c.json`. Trained on raw OpenHermes-2.5 chat (general, NOT RAG),
50k samples / 1 epoch on one A100-40GB. Final eval loss 0.879 (0.912 -> 0.879, plateaued early).

**Harness adaptation (minimal):** `common.py` now reads `C2C_RECEIVER` / `C2C_SHARER` /
`C2C_FUSER_DIR` env vars (Qwen defaults unchanged). The Llama tokenizer accepts the Qwen-only
`enable_thinking=False` kwarg harmlessly, so NO other code change was needed. Eval runner:
`scripts/eval_llama.sh {obqa,squad}`.

**Training gotchas (cost time):** `wandb` is an unlisted import in C2C's `SFT_train.py` (pip
install it; run with `WANDB_MODE=offline`). Single-A100 memory: per_device_batch_size=4 OOMs
mid-run (~step 68, peak ~39.7GB) even with `expandable_segments`; **batch_size=3 (grad_accum 11,
effective 33) is the safe config** (~38GB peak, ~10s/it, ~4.3h for 50k). The trainer has NO
resume-from-checkpoint logic, so a late OOM loses everything back to the last `save_steps`.

**Results (n=500).** OBQA = full-prompt fusion; SQuAD = doc_only (cacheable) fusion + Haiku judge.

| OpenBookQA | acc | | SQuAD (judge) | text | c2c doc_only | delta |
|---|--:|---|---|--:|--:|--:|
| receiver/none (1B) | 27.6 | | retrieved | 57.6 | 56.5 | -1.0 (ns) |
| **c2c/none (fusion)** | **39.8** (+12.2) | | oracle | 82.0 | 79.8 | -2.2 (ns) |
| sharer/none (8B) | 78.6 | | | | | |
| text_rag/retrieved | 41.2 | | | | | |
| **c2c/retrieved** | **45.4** (+4.2) | | | | | |

**Read:** OBQA **reproduces the C2C win** on a self-trained pair (+12.2 pure fusion, +4.2 over
text) -> training works AND generalizes. SQuAD is **flat/slightly negative** (doc_only, ns) —
*not* a contradiction: Llama-1B-**Instruct** is already a strong extractive reader (text_rag judge
57.6 vs Qwen-0.6B's 51.2), so on extractive QA there's little encoding deficit for fusion to fill;
C2C-RAG helps when **knowledge-bottlenecked** (OBQA), per the original thesis. The lexical/judge
split also flips here (c2c wins `contains` 55.4>53.0 but loses judge) — wordier, less-precise
doc_only output, a sign the **feasibility fuser is undertrained for the harder doc_only projection**
(consistent with the early eval-loss plateau). Absolute gains < released Qwen fuser (+13.8/+16),
expected at 50k/1-GPU vs 500k/8-GPU.

**Scaling: 50k -> 150k OpenHermes (2026-06-19).** Same pair/recipe, 3x data
(`recipes/llama_1b8b_c2c_150k.json`, batch 3, ~12.6h, eval loss 0.879 -> **0.717**). The big loss
drop -> the 50k plateau was **undertraining, not a capacity ceiling**. Re-eval (50k vs 150k, n=500):

| OpenBookQA (acc) | 50k | 150k | | SQuAD judge | text | c2c 50k | c2c 150k |
|---|--:|--:|---|---|--:|--:|--:|
| c2c/none vs recv 27.6 | 39.8 (+12.2) | 40.8 (**+13.2**) | | retrieved | 57.3 | 56.5 (-1.0 ns) | **61.2 (+3.9, p=.042)** |
| c2c/retr vs text 41.2 | 45.4 (+4.2) | 47.8 (**+6.6**) | | oracle | 82.7 | 79.8 (-2.2 ns) | 83.9 (+1.2 ns) |

**Headline: SQuAD doc_only FLIPPED from a loss to a significant win** (retrieved -1.0 -> +3.9,
p=0.042; oracle sign flip -2.2 -> +1.2). So the 50k SQuAD flatness was mostly **undertraining of
the harder doc_only (cacheable) projection**, exactly what the eval-loss drop predicted. OBQA
fusion advantage also widened. **The verbosity-robust judge was essential**: the 150k fuser's
token-EM *dropped* (wordier) while `contains` AND judge-correctness *rose* — the extra words carry
correct content; EM/F1 would have hidden the win. Scaling is sub-linear vs loss (loss fell a lot,
OBQA moved +1-2.4) but **broad, and the deployable doc_only setting benefited most**. Moral:
these projectors are data-hungry — let them train. 50k eval JSONs backed up to
`scripts/{obqa,squad}_results_llama50k/`.

**Third benchmark: TriviaQA-RAG (150k fuser, 2026-06-19).** Built the corpus
(`trivia_prep.py --target 500 --scan 2000` -> 500 Qs, 21,812 ~100-word Wikipedia chunks),
doc_only + Haiku judge (n=500; judge is SLOW here — TriviaQA's long gold-alias lists make big
prompts, ~12 min/cell). Results backed up to `scripts/trivia_results_llama150k/`.

| TriviaQA judge | text | c2c doc_only | delta |
|---|--:|--:|--:|
| retrieved | 52.0 | 50.2 | -1.8 (53/62, p=0.46, ns) |
| oracle | 64.8 | 63.6 | -1.2 (46/52, p=0.61, ns) |

**Clear non-win** (loses `contains` too, unlike SQuAD). Completes a clean 3-point picture for the
**Llama-1B-Instruct <- 8B-Instruct** pair — the fusion advantage tracks the receiver's knowledge
bottleneck almost monotonically:

| benchmark | 1B floor | retr hit | C2C vs text (150k, judge) |
|---|--:|--:|---|
| OpenBookQA (MC, knowledge-bottlenecked) | 27.6 | 48% | **+6.6** retr / **+13.2** no-retr (win) |
| SQuAD (extractive) | ~0-17 | 61% | **+3.9** p=0.042 (win) |
| TriviaQA (famous facts) | 44.2 | 53% | **-1.8** ns (no win) |

Throughline: fusing a passage's KV helps where the receiver is **weak on the task** (OBQA) or
**genuinely needs the retrieved info** (SQuAD), but **not on trivia a competent 1B-Instruct already
knows** (highest floor, noisier retrieval). Consistent with the original GSM8K/HotpotQA boundary:
C2C-RAG is knowledge-injection, and only helps when there's a knowledge deficit to inject.

**Next:** RAG-aware training (see "RAG-specialized fuser" directions) — higher signal-per-sample,
targets the doc_only/cacheable projection; and/or go bigger (we're at 150k/1-GPU vs the released
Qwen fuser's 500k/8-GPU, likely not the ceiling).

## Why TriviaQA failed + a dataset SCREEN for fusion-amenability (2026-06-20)

Dug into the TriviaQA negative (8B sharer clearly > 1B, so why no fusion win?). Measured the 8B
ceiling we'd never run: **8B parametric (none) contains 73.6 ~= 8B reading the passage (72.0)** ->
the 8B's TriviaQA edge is **parametric, not document-derived** (the noisy 100-word chunk doesn't
help it). And **fusing the question (no retrieval, doc_q) HURT the 1B: 44.2 -> 36.2 contains**
(vs OBQA where the same move *lifted* 27.6 -> 39.8). Conclusion: **C2C fuses context-token KV
(encode-side); an open-domain answer materializes at DECODE from the sharer's weights, so KV
fusion structurally can't transfer it** — same encode/decode boundary as the GSM8K/HotpotQA
negatives. Headroom existed but was the wrong *kind*. (Where the answer IS in the doc — SQuAD
oracle — fusion is flat-not-negative and `doc_q`>`doc_only`, i.e. the trainable regime.)

**=> A cheap pre-screen to pick fusion-amenable datasets BEFORE training** (oracle context, 4
batched cells: 1B-none, 8B-none, 1B-text, 8B-text; no fuser needed). Two preconditions:
- **encoding headroom** = `8B-text - 1B-text` (does the small model under-encode the context?)
- **answer-in-doc** = `8B-text - 8B-none` (is the answer in the passage vs the sharer's weights?)
Fusion needs BOTH large. Scripts: `pubmedqa_prep.py`, `boolq_prep.py`, `narrativeqa_prep.py`
(NarrativeQA uses the story SUMMARY as context). Screen (F1, oracle, n=500):

| dataset | 1B-none | 8B-none | 1B-text | 8B-text | enc gap | answer-in-doc |
|---|--:|--:|--:|--:|--:|--:|
| **NarrativeQA** | .05 | .07 | .53 | .69 | +0.15 | **+0.62** |
| PubMedQA | .26 | .55 | .44 | .66 | +0.22 | +0.11 |
| BoolQ | .22 | .66 | .35 | .81 | +0.46* | +0.15 |
| TriviaQA (ref, lost) | .39 | .72 | .63 | .81 | +0.18 | +0.09 |

**NarrativeQA = the prime target**: answer ~entirely in-doc (8B-none ~= 0; the precondition
TriviaQA flunked) AND a real encoding gap -> the ideal doc_only/cacheable-fusion profile
(non-extractive, encode-bottlenecked). PubMedQA solid (dense technical encoding). BoolQ gap is
real but *inflated by yes/no format + label-prior (the 1B barely emits clean yes/no); treat
cautiously. NB NarrativeQA collapsed to 17 unique summaries in 500 Qs -> clean for the ORACLE
encoding test, but needs more documents for a real retrieval eval. Screen only measures
PRECONDITIONS; whether the fuser captures the headroom still needs a fusion run / RAG-aware
training (NarrativeQA the natural target).

**Off-the-shelf 150k fuser on NarrativeQA (oracle, F1):** the current fuser captures NONE of the
headroom -- it's net-NEGATIVE.

| method | F1 | vs 1B-text |
|---|--:|--:|
| text_rag (1B reads summary) = bar | 0.534 | — |
| c2c doc_only (150k fuser) | 0.437 | -0.097 |
| c2c doc_q (150k fuser) | 0.484 | -0.050 |
| sharer (8B reads summary) = ceiling | 0.688 | +0.154 |

Fusing the summary KV makes the 1B *worse* than reading it (doc_q>doc_only again -> doc_only is
the weak link). BUT unlike TriviaQA the headroom is **reachable in principle** (answer is in the
doc), so this is the RAG-aware-training opportunity, not a structural dead-end. **Falsifiable
prediction for the RAG-aware run: it should FLIP NarrativeQA to a win (reachable) but NOT fix
TriviaQA (structural / decode-time).** That contrast is the cleanest test of the encode/decode
thesis. NarrativeQA datasets/results live under `scripts/narrativeqa_*` + `narrativeqa_results/`.
(Free-form F1 is rough here; an LLM-judge pass would refine the absolute numbers.)

## HF Hub upload validated + target-application framing (2026-06-20)

The 150k fuser is published at **`vihan29/c2c_llama8b_llama1b_150k`** (16 `projector_*.pt` +
`projector_*.json` + `projector_config.json` at repo root; no `config.json`/recipe — keep the
recipe `recipes/llama_1b8b_c2c.json` for provenance). Round-trip validated: `snapshot_download`
the repo, point `C2C_FUSER_DIR` at the snapshot dir, run `eval_llama.sh squad` -> reproduces the
local model. SQuAD judge via the HF fuser: retrieved text 57.1 / **c2c 60.8 (+3.7, p=0.047)**,
oracle 83.1 / 83.3 (ns) — matches local (+3.9 / +1.2) within judge+fp noise. Generations near
token-identical to the local checkpoint.

**Strategic throughline (where this method should live).** Public RAG benchmarks (TriviaQA/NQ/
SQuAD) mostly test Wikipedia facts the models already hold parametrically — the *worst* case for
fusion (no knowledge deficit). The real RAG use case is **non-parametric knowledge** (private/
enterprise corpora, post-cutoff content, niche domains, fiction/long-tail). Our two preconditions
ARE that target, operationalized: **answer-in-doc (`8B-none ≈ 0`) = "non-parametric"** (and the
`8B-none` cell doubles as a pretraining-contamination detector), **plus encoding headroom
(`8B-text ≫ 1B-text`) = "the doc is hard to encode."** Necessary together — a plainly-stated fact
in a short non-parametric doc still won't beat 1B-text. Crucially, the **cacheable doc_only
architecture and this accuracy precondition converge on the same customer: enterprise/private-
corpus RAG** (non-parametric + high reuse + offline 8B precompute). Reasoning-RAG (GSM8K) is NOT
the target — reasoning is decode-time like parametric recall; predict it fails the screen
(`8B-none ≈ 8B-text`). Next-session targets: RAG-aware training on NarrativeQA (reachable headroom,
verified) + screen 1-2 genuinely non-parametric sets (post-cutoff / synthetic-private-doc QA).

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

## RAG-specialized fuser — research directions (2026-06-18)

We've so far used fusers trained on **general OpenHermes chat** (monolithic full-prompt fusion)
and applied them **zero-shot** to a structured RAG setup (`[doc=fuse, question=no-fuse, answer]`).
That train/test mismatch is the opening: a fuser *tailored* to retrieval may beat the general one
on the same eval. Clean scientific framing for any idea below — train a RAG-specialized fuser and
**ablate it vs the general fuser on the same OBQA/SQuAD harness, same pair** (delta = "did
specialization help"). All are recipe/projector changes, not rewrites. Ranked by leverage:

**Objective (cheapest, highest near-term quality)**
1. **Distill the sharer-reading-text behaviour (top pick).** Replace plain next-token LM loss with
   a KL distillation: teacher = the *8B reading the passage as text in-context*; student =
   `(1B + fused-doc-KV)`; supervise on the answer span. Trains the fuser to literally make the
   small model act like the big model on that passage — directly attacks our **encode/decode split**
   (fusion lifts `contains` ≫ EM) by supervising the *decode* distribution, not just what's readable.
2. **Answer-span-weighted loss + brevity.** Short answers dominate our tasks; weight loss toward the
   answer tokens (and/or a small format objective) so the fuser closes the **EM gap natively**
   instead of via the terse-instruction + copula-extraction scoring hack.

**Architecture (highest impact; ties to a wall we already hit)**
3. **Query-agnostic fused doc KV.** The deployable path precomputes doc KV offline, then a
   *different* question attends to it. Train with that structure — augment each doc with several
   different questions so the fused doc KV is regularized to be query-general (not co-adapted to one
   prompt). Serves cacheability directly.
4. **Re-rotatable / position-decoupled fusion (the big architectural bet).** Top-k offline
   composition stalled because **fused keys aren't RoPE-re-rotatable** (see Multi-doc / HotpotQA).
   A projector that emits position-canonical keys (re-rotatable like vanilla KV) would unlock
   multi-chunk cached fusion — a concrete contribution motivated by a documented limitation, not a
   hypothetical. Opens the whole multi-doc cacheable story.

**Data/recipe**
5. Mix QA-in-RAG-format (SQuAD / TriviaQA / NQ train splits) into the OpenHermes pretraining so the
   fuser sees retrieval-passage lengths and the 3-section structure; hold out QA for eval.

Suggested ordering: **#1** for the quickest win, **#4** for the highest ceiling. Both are
projector/recipe edits in the C2C `SFT_train.py` + recipe JSON pipeline (see the Llama-pair
training run below / `recipes/llama_1b8b_c2c.json`).

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
