# p53 dataset v1 — replicated peaks (GSE132246, Nutlin-3a)

Rebuild with `python build_dataset_v1.py --fasta /path/to/hg38.fa` (needs `pandas numpy pyfaidx pyliftover`).
Seed 42; output is deterministic.

## Source
- **GEO GSE132246**, OCI-AML3 (AML line, TP53 wild-type), p53 ChIP-seq, 4 treatments × 4 replicates.
- **Positives come from Nutlin-3a 2.5 µM, Rep1–4 (GSM3854900–GSM3854903).**
- Depositor's peaks: Bowtie2 → MACS 1.4.2, `--pvalue 1e-5`, hg19. **No input control is deposited**, which
  likely explains why most single-replicate peaks are noise (see below).
- Metadata conflict, unresolved: the series summary says 24 h treatment, every sample record says 6 h.
- DMSO / CPI-203 / Combo peaks are used only to keep negatives away from any called peak.

## Why ≥3 of 4 replicates
Merged Nutlin peak regions, 500 bp at the region midpoint, p53 motif rate vs number of supporting replicates:

| support | regions | full p53 site | half-sites / window |
|---|---|---|---|
| 1/4 | 13,145 | 0.6% | 0.21 |
| 2/4 | 1,122 | 3.4% | 0.51 |
| 3/4 | 745 | 8.9% | 0.76 |
| 4/4 | 1,892 | 14.7% | 0.93 |
| random genome | — | 0.1% | 0.14 |

1/4 peaks are close to random genome; even the stronger-scoring half of them is (0.7%, 0.23).
Reproducibility tracks peak score: Nutlin Rep1 peaks in score deciles 1–6 reappear in another replicate
6–20% of the time, deciles 9–10 98–100%. The top 1,000 peaks reproduce 82–87% between any two Nutlin
replicates; DMSO and CPI-203 reach only 12–58%, so those conditions are not used for positives.
2/4 regions are written to `ambiguous_2of4.csv` and are not trained on.

## Construction (same filters for both classes)
- Canonical chromosomes chr1–22, X; hg38 ENCODE blacklist v2; no N; exactly 500 bp.
- Positive window: 500 bp centred on the midpoint of the highest-scoring replicate peak in the region
  (MACS 1.4 BEDs have no summit), lifted hg19 → hg38 with UCSC `hg19ToHg38.over.chain`.
- Negatives: random windows **on the same chromosome as each positive**, outside the blacklist and
  outside every peak from all 16 samples ± 1 kb, GC-matched 1:1 in 0.02 bins.
- `fold`: chromosomes split into 5 folds balanced by positive count. Use grouped CV over `fold`;
  never split windows from one chromosome across train and test.

## Contents
| file | rows | |
|---|---|---|
| `dataset_v1.csv` | 2,608 pos / 2,606 neg | `loc, chr, start, end, seq, gc, label, support, peak_score, fold, src` (hg38) |
| `ambiguous_2of4.csv` | 1,068 | 2/4-support windows, for a separate analysis only |
| `raw_peaks/` | 16 | depositor BEDs, as downloaded from GEO |
| `ref/` | | hg38 blacklist v2, hg19→hg38 chain |

## Baselines on this file (grouped 5-fold, chromosome folds, logistic regression)
| model | AUROC |
|---|---|
| GC + CpG | 0.530 ± 0.010 |
| p53 half-site count (no training) | 0.767 |
| 4-mer | 0.795 ± 0.014 |
| 5-mer | 0.828 ± 0.006 |
| 4-mer + regex motif counts | 0.872 ± 0.011 |
| **JASPAR MA0106.3 PWM, max score per window (no training)** | **0.966** — the real bar |

### p53 motif (added 2026-10-25)
- Consensus (el-Deiry 1992): two half-sites `RRRCWWGYYY` separated by 0–13 bp. An exact regex for it hits
  only 13.1% of positives (0% of negatives), because real sites carry mismatches.
- PWM: `ref/MA0106.3.jaspar` (JASPAR TP53, HT-SELEX, 18 bp, contiguous half-sites). It was derived in
  vitro, not from this ChIP data, so scoring with it involves no leakage. Relative score ≥ 0.85 hits
  82.7% of positives, 55.5% of 2/4 peaks, 1.1% of negatives.
- Spacers: 75% of positives score best at spacer 0; negatives are flat across 0–13. Allowing spacers
  rescues 149/171 low-scoring positives but also 69% of low-scoring negatives, so it adds little.
- Motif-free positives (full-site and half-site scores both ≤ the negatives' 90th percentile): 69 of
  2,608. They are reproducible (36 at 4/4) but are 51% within 1 kb of a GENCODE v49 TSS (all positives 18%,
  negatives 12%) with 3.4× the CpG density, i.e. mostly CpG-island promoters. Promoters often show up in
  ChIP-seq regardless of the factor, so a model that is confident about motif-free sequences may simply be
  detecting CpG islands. Any motif-free prediction needs a CpG/promoter-matched control before it is believed.
- Consequence: random-genome negatives are easy. The informative test is **motif-matched negatives**
  (PWM ≥ 0.85, no peak in any of the 16 samples): can a model tell bound sites from unbound ones?

## Soft-label test (2026-10-25, same baseline and folds)
Training targets per tier; evaluation always on the hard 3+/4-vs-background set.

| training scheme | primary AUROC | change vs hard |
|---|---|---|
| hard: 3+/4 = 1, background = 0 | 0.8721 | — |
| + 2/4 = 0.75 | 0.8730 | +0.001 (noise) |
| + 2/4 = 0.75, 1/4 = 0.5 | 0.8700 | −0.002 |
| 4/4 = 1, 3/4 = 0.7, 2/4 = 0.35 | 0.8682 | −0.004 |
| … + 1/4 = 0.05 | 0.8573 | −0.015 (every fold) |
| 1/4 as hard negatives | 0.8474 | −0.025 (every fold) |

1/4 windows are not pure noise (the hard model ranks them above background at 0.61 AUROC), so
labelling them negative or 0.5 hurts. Keep 1/4 out of training. 2/4 at any soft label is neutral
for this linear baseline; re-test with the CNN / Evo 2 (secondary experiment).
1/4 windows are not saved here; regenerate them with the same filters as `build_dataset_v1.py`.

## Context-length variants (`make_length_variants.py`)
`dataset_v1_tiers_w{500,1000,2000}.csv`: the same 9,843 rows at three window lengths. Every row keeps
its 500 bp centre, tier and annotation columns (`gc`, `cpg`, `pwm`, `pwm_offset`, `tss_dist` are
computed on the central 500 bp, so evaluation strata are identical across lengths); only `seq`,
`start`, `end` change. A row was dropped from all three if its 2 kb window left the chromosome or hit
an N, the blacklist, or (negatives) any peak: 3 rows (2 random_neg, 1 motif_neg). `w500` is
therefore `dataset_v1_tiers.csv` minus those 3 rows. Flank GC (outside the central 500 bp) is
balanced across classes: at 1 kb, positive 0.436, random_neg 0.438, motif_neg 0.439
(promoter_neg 0.500, as expected near TSSs).

Not comparable to v0 numbers (different study, cell line, genome build, and split).
