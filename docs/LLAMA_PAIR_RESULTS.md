# Second fuser pair: Llama-3.2-1B-Instruct <- Llama-3.1-8B-Instruct (2026-09-06)

Full paper matrix re-run on a second receiver/sharer pair, to test whether the CacheGraft result
generalizes beyond Qwen3-0.6B <- Qwen3-4B-Base. Same six eval sets, same cells, same two local
judges, one A100-40GB, 3h12m end to end. Fuser = the published `vihan29/c2c_llama8b_llama1b_150k`
(150k OpenHermes, general chat — zero-shot w.r.t. retrieval, same as the Qwen fuser).

Raw table: `llama_pair_full_report.txt`. Results dirs: `scripts/<data>_results_llama/`
(via the new `C2C_RUN_TAG` env suffix in `common.paths`, so both pairs coexist).

## Headline: the effect does NOT generalize

CacheGraft - receiver text-RAG, primary judge **phi-4** (`*` = p<0.05, exact McNemar):

| dataset | setting | Qwen 0.6B<-4B | **Llama 1B<-8B** | gemma2 | contains |
|---|---|---:|---:|---:|---:|
| SQuAD | retrieved | +7.9* | **+4.1*** | +0.1 | +4.5* |
| SQuAD | oracle | +6.4* | **+5.2*** | +0.8 | +5.8* |
| AdversarialQA | retrieved | +6.0* | +1.0 | -0.7 | +2.4 |
| AdversarialQA | oracle | +7.2* | -0.5 | -3.8 | -2.4 |
| CovidQA | retrieved | +3.8* | -0.5 | -1.8 | +2.1 |
| CovidQA | oracle | +5.4* | -2.0 | -3.7* | +2.1 |
| QASPER | retrieved | +0.4 | +1.6 | -2.9 | +2.3* |
| QASPER | oracle | +2.9 | **+4.7*** | -2.2 | +5.2* |
| TriviaQA | retrieved | +3.5* | -2.2 | -4.5* | -2.5 |
| TriviaQA | oracle | +3.2 | -0.9 | -3.5* | -0.6 |
| TriviaQA-500 | retrieved | — | -3.2 | -5.8* | -3.6 |
| TriviaQA-500 | oracle | — | -0.8 | -4.4* | -1.4 |

**Qwen pair: 21/22 judge x dataset x setting deltas positive, 11 significant.
Llama pair: 7/24 positive, 3 significantly positive (SQuAD x2, QASPER oracle),
and 5 significantly NEGATIVE (all Gemma: CovidQA oracle + all four TriviaQA cells).**

Gemma finds no significant gain anywhere for this pair, including on SQuAD where phi-4 does.
Inter-judge kappa is 0.48-0.78, comparable to the paper's 0.47-0.79 — the disagreement is about
this pair's effect, not about judge quality.

## What DOES replicate (and is worth keeping in the paper)

1. **The receiver-own-KV control is null in all 12 cells, on every metric.** phi-4 deltas
   -0.4..+0.5, all n.s.; contains -0.4..+0.4. The isolation argument — the gain is *whose* KV
   fills the doc section, not the cache machinery — now holds on a second model family. This is
   the single most transferable result of the run.
2. **The persisted two-phase cache reproduces the inline path.** phi-4 +4.4/+5.2 (cached) vs
   +4.1/+5.2 (inline); **92.8% / 93.8% token-exact** generations, matching the paper's 93-94%.
3. **Offline sharer text is still worse than raw text.** phi-4 -1.8 (retrieved) / -1.9 (oracle),
   contains -2.2* / -3.0*. Directionally replicates, but far weaker than the Qwen pair's -14.0 /
   -20.6, because Llama-8B-Instruct writes much better query-agnostic summaries than Qwen-4B-Base.
   The paper's "offline text is counterproductive" framing is Qwen-specific in magnitude.

## What does NOT replicate

4. **"Verbosity is not the explanation" fails here.** Under the matched short-answer protocol
   (`--terse`, applied to every method), CacheGraft wins `contains` (+7.8 retrieved / +11.1
   oracle) but **loses EM** (27.6 vs 30.0; 39.8 vs 42.1). The paper claims an EM win (28.1 vs
   23.1) for the Qwen pair. That paragraph must be scoped to the Qwen pair.
5. **The un-terse EM penalty is much larger**: -14.8 / -20.4 points (Qwen: ~-9). This pair is
   1.8-2.9x wordier than its text baseline.

## The screen: one precondition works, the other anti-predicts

Screen terms recomputed for this pair (contains, oracle context):

| dataset | encoding headroom | answer-in-doc | phi-4 gain (oracle) | gap recovery |
|---|---:|---:|---:|---:|
| AdversarialQA | **+24.0** | +41.6 | -0.5 | -2% |
| SQuAD | +22.5 | +70.0 | +5.2* | 24% |
| CovidQA | +16.3 | +39.0 | -2.0 | -9% |
| TriviaQA-500 | +16.2 | **+10.2** | -0.8 | -6% |
| TriviaQA | +14.9 | **+6.6** | -0.9 | -7% |
| QASPER | **+9.9** | +33.9 | +4.7* | 37% |

* **answer-in-doc works, and works better than for Qwen.** TriviaQA's term collapses to +6.6/+10.2
  for this pair (Qwen: +24.7) because Llama-8B-Instruct holds the trivia parametrically — and
  TriviaQA is indeed the most negative dataset, in all four cells and both judges. The screen
  rules it out in advance, as designed.
* **encoding headroom ANTI-predicts.** AdversarialQA has the largest headroom of any dataset
  (+24.0) and delivers nothing; QASPER has the smallest (+9.9) and delivers the most (37%
  recovery). The paper's ordering claim does not survive a change of pair.
* **The screen does not transfer across pairs.** It ranks datasets given a fixed pair; it does not
  predict that this pair captures less than the Qwen pair despite having *more* headroom on SQuAD
  (+22.5 vs +11.3).
* **Missing term.** The receiver's own parametric knowledge is uncontrolled. Llama-1B scores
  40.6% phi-4 with no passage on TriviaQA vs Qwen-0.6B's 15.0%. A `receiver_none` term belongs in
  the screen: fusion cannot help where the receiver already knows the answer.

## Incidental win: the storage limitation is 5x smaller here

| | Qwen 0.6B receiver | Llama 1B receiver |
|---|---:|---:|
| KV per ~130-token passage | ~18 MB | **~3.6 MB** |
| offline build | 0.05-0.19 s | 0.11 s |
| online query (receiver-only) | 0.58 s | **0.242-0.251 s** |

28 layers x head_dim 128 vs 16 x 64. The paper's storage limitation is a property of the receiver
architecture, not of the method — worth saying, since it cuts the headline cost 5x.

## Suggested framing for the paper

Do not present this as a generalization table. The honest claims are:

1. The **mechanism** is pair-invariant (null control, cache equivalence, offline-text negative).
2. The **magnitude** is strongly pair-dependent: significant on 2 of 6 datasets under the primary
   judge and on none under the second judge.
3. The **screen's second precondition** predicts the failures correctly and is the part that
   generalizes; the first precondition does not, and the screen needs a receiver-parametric term.
4. The existing limitation ("the benefit shrinks as receivers get stronger") is now **measured**,
   not preliminary — and it is sharper than stated: with a stronger receiver the effect can go
   negative under a second judge, not merely shrink.
