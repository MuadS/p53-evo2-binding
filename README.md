# Predicting p53 binding sites with Evo 2 and LoRA

Course project for COSC 4337 (Data Science II), University of Houston, Fall 2026.
Group: Muad Shaikh, Kenny Ngo and Kevin Chau. Muad Shaikh built and trained the CNN and Evo 2 models.

## Question

Can a genomic foundation model (Evo 2, 7B parameters), adapted with LoRA, predict where the tumor suppressor p53 binds from DNA sequence alone, and does it learn anything beyond the known p53 motif?

## Data

- Positives: p53 ChIP-seq peaks from GEO [GSE132246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132246) (OCI-AML3 cells, Nutlin-3a), kept when reproduced in at least 3 of 4 replicates (n = 2,608), as 500 bp windows on hg38.
- Negatives in three tiers: GC-matched random genome windows, CpG/GC-matched promoter windows, and windows that contain the p53 motif but are unbound.
- References: JASPAR MA0106.3 (p53 motif), the ENCODE hg38 blacklist and the UCSC hg19-to-hg38 liftOver chain, in `p53_dataset_v1/ref/`.

`p53_dataset_v1/dataset_v1_tiers.csv` is the training table. The scripts in `p53_dataset_v1/` rebuild it from `raw_peaks/`.

## Models

| model | what it is |
|---|---|
| PWM | scores the published p53 motif; no training |
| k-mer | logistic regression on short DNA words |
| 1D CNN | 1.1M-parameter convolutional network trained from scratch |
| Evo 2, frozen | fixed backbone with a trained classification head |
| Evo 2 + LoRA | low-rank adapters in the first five blocks plus a classification head |

## Evaluation

Grouped 5-fold cross-validation by chromosome, so no chromosome appears in both training and testing. AUROC is reported as the mean and standard deviation across folds, and each model is compared with the PWM baseline using paired-bootstrap 95% confidence intervals. All numbers are in `results/eval_summary.csv`; figures are in `results/figures/`.

Selected results (AUROC, mean over 5 folds):

| comparison | PWM | best k-mer | CNN | Evo 2 + LoRA |
|---|---|---|---|---|
| bound vs. random windows | 0.966 | 0.957 | 0.978 | 0.985 |
| bound vs. promoters | 0.970 | 0.940 | 0.972 | 0.975 |
| motif-free bound sites (n = 182) vs. promoters | 0.623 | 0.717 | 0.850 | 0.891 |
| bound vs. motif-present unbound* | 0.532 | 0.723 | 0.898 | 0.919 |

\* Provisional. The motif-containing negatives were selected by a scan that required both halves of the p53 motif to match exactly, so they look more canonical than real binding sites. That set is being rebuilt.

## Limitations

One random seed per model; LoRA adapters in 5 of 32 blocks; LoRA training capped at 6 epochs while validation AUROC was still rising. Predictions indicate binding potential, not proof of binding.

## Running it

See `model_training/README.md`. Evo 2 runs need a GPU and the `evo2_7b_base` weights from Hugging Face (arcinstitute/evo2_7b_base); scripts write to `p53_evo2/` by default.
