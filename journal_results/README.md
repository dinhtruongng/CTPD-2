# CTPD Journal Extension — Experiment Results

This folder holds **real, reproducible** results for the Neurocomputing journal
extension of the CTPD AAAI 2026 paper. It lives inside the reference code so the
scripts run against the exact same aligned-span algorithm and data pipeline.

Nothing here overwrites the paper. The LaTeX draft
(`NeuComp_Template/.../main.tex`) currently contains **placeholder/illustrative
numbers** in its results tables; the values below are what the code actually
produces and should replace them.

## Layout

- `scripts/mismatch_analysis.py` — tokenizer-mismatch stats (no GPU). Reuses the
  repo's `find_parent_token` so spans match training exactly.
- `scripts/weight_lite.py` — **CTPD-Lite** single-teacher cached span-weight
  generation (new contribution; replaces the positive/negative DPO teacher pair
  of `src/prefkd/weight.py`).
- `scripts/CTPD_Lite.sh` — launch CTPD-Lite training (same training stage as
  `script/training/CTPD.sh`, only the weighted dataset differs).
- `mismatch/mismatch_stats.json` — raw per-pair mismatch statistics.

## Mismatch analysis (DONE, CPU-only)

Computed over 2,000 UltraFeedback examples (chosen+rejected) per pair with each
pair's real teacher/student tokenizers. `find_parent_token` resolves shared
character boundaries into aligned spans, then spans are classified 1:1 / 1:n /
n:1 / n:m.

### Key finding — the draft's mismatch numbers are wrong

The draft claims Qwen→Llama is heavily fragmented (~60% 1:1, ~11% n:m). In
reality Qwen-2.5 and Llama-3 are **both byte-level BPE with 64.3% vocabulary
overlap**, so they agree on almost every boundary: **98.6% of spans are 1:1**
and only 0.6% are n:m. The genuinely mismatched pairs are the ones involving
**Gemma** (SentencePiece, 256K vocab, ~7% overlap) and **Mistral→Llama**
(Mistral's coarse 32K vocab → 14.3% n:1 spans).

This *strengthens* the journal story but changes which pairs illustrate it:
mismatch is real and graded, but it tracks *tokenizer family/algorithm*, not the
Qwen↔Llama axis the conference paper happened to use.

### Real per-pair statistics

| Teacher → Student | Vocab overlap (%) | Tok/span (T) | Tok/span (S) | T:S ratio | 1:1 (%) | 1:n (%) | n:1 (%) | n:m (%) |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B → Llama-3.2-1B (control) | 100.0 | 1.00 | 1.00 | 1.00 | 99.99 | 0.00 | 0.00 | 0.01 |
| Qwen-2.5-14B → Llama-3.1-8B | 64.3 | 1.005 | 1.00 | 1.011 | 98.60 | 0.00 | 0.80 | 0.60 |
| Qwen-2.5-7B → Llama-3.2-1B | 64.3 | 1.005 | 1.00 | 1.011 | 98.60 | 0.00 | 0.80 | 0.60 |
| Llama-3.1-8B → Qwen-2.5-1.5B | 64.3 | 0.997 | 1.011 | 0.989 | 98.87 | 0.79 | 0.01 | 0.34 |
| Qwen-2.5-7B → Gemma-2-2B | 6.6 | 1.043 | 1.090 | 0.960 | 89.17 | 6.33 | 1.80 | 2.71 |
| Gemma-2-9B → Llama-3.2-1B | 7.1 | 1.086 | 1.047 | 1.053 | 87.19 | 1.78 | 7.13 | 3.90 |
| Mistral-7B → Llama-3.2-1B | 7.1 | 1.165 | 1.006 | 1.172 | 83.93 | 0.07 | 14.25 | 1.76 |

(Rows for the two Qwen→Llama pairs are identical because the statistic depends
only on the tokenizer pair, which is shared within a family. Vocab overlap is
Jaccard over token strings; Gemma's low overlap is partly a vocab-size artifact,
which is why the span-type distribution is the more reliable mismatch index.)

### Mismatch index for the scatter plot

Recommended scalar mismatch index = `1:1 deficit` = `100 − pct_1to1`
(equivalently `pct_1ton + pct_nto1 + pct_ntom`):

| Pair | 1:1 deficit (%) |
|---|---|
| Llama→Llama (control) | 0.01 |
| Qwen-2.5-14B→Llama-3.1-8B | 1.40 |
| Llama-3.1-8B→Qwen-2.5-1.5B | 1.13 |
| Qwen-2.5-7B→Gemma-2-2B | 10.83 |
| Gemma-2-9B→Llama-3.2-1B | 12.81 |
| Mistral-7B→Llama-3.2-1B | 16.07 |

The CTPD-gain side of the scatter requires the GPU training runs (not yet run).

## GPU experiments — NOT YET RUN

The cross-family, efficiency, and robustness tables in the draft are
placeholders. They require training (SFT → teacher DPO pos/neg → weight gen →
CTPD), which had not been run on this server (no checkpoints, datasets, or base
models were present). Awaiting greenlight on which pairs/sweeps to run.
