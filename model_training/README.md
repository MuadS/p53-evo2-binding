# p53 binding — model training (Evo 2 7B + LoRA vs 1D CNN)

Every script uses the same rows, chromosome folds and evaluation (`p53_common.py`):
grouped 5-fold CV over `fold`; for test fold *k* the validation fold is *(k+1) % 5*; training
uses positives + all three negative tiers; evaluation always uses hard labels and reports each
tier separately. Data: `../p53_dataset_v1/dataset_v1_tiers.csv` (see that folder's README).

| script | GPU | what it does |
|---|---|---|
| `train_cnn.py` | 1, short | 1.1M-param CNN from scratch, all 5 folds in one run |
| `extract_evo2_embeddings.py` | 1, one pass | frozen Evo 2, mean-pooled features for 9 layers, both strands |
| `probe_evo2.py` | CPU | frozen arm #1: logistic probe per layer, layer chosen on the validation fold |
| `train_evo2_lora.py --no-lora` | 1 per fold | frozen arm #2: same head as LoRA, trained by SGD |
| `train_evo2_lora.py` | 1 per fold, long | LoRA arm |
| `eval_models.py` | CPU | PWM / k-mer baselines + every model, per tier, paired-bootstrap gaps |

## Bar to beat (baselines, grouped 5-fold AUROC, computed by `eval_models.py`)

| comparison | PWM (no training) | 5-mer | 4-mer + PWM |
|---|---|---|---|
| positive vs random_neg | **0.966** | 0.774 | 0.957 |
| positive vs motif_neg (motif present, unbound) | 0.532 | **0.723** | 0.717 |
| positive vs promoter_neg | **0.970** | 0.747 | 0.940 |
| TSS-proximal positive vs promoter_neg | **0.944** | 0.710 | 0.920 |
| motif-free positive vs promoter_neg (n = 182) | 0.623 | **0.717** | 0.647 |

The informative comparison is **positive vs motif_neg**: the PWM is at chance there by design,
so a model that wins must read something besides the motif.

## Working directory layout on the GPU machine

```
p53_evo2/
├── code/         this folder
├── data/         dataset_v1_tiers.csv
├── weights/evo2_7b_base/evo2_7b_base.pt     13.0 GB, from huggingface.co/arcinstitute/evo2_7b_base
├── embeddings/evo2_7b_base/layer*.npy       ~2 GB (9 layers x 13,822 rows x 4096 float32)
└── models/       logs, checkpoints (LoRA + head only), *_preds.csv, eval_summary.csv
```

## Environment (conda env `evo2`)

Evo 2 needs Python 3.11/3.12, CUDA ≥ 12.1, torch 2.6/2.7 and flash-attn. The 7B checkpoints run
in bf16 **without** Transformer Engine (FP8 projections fall back to bf16 automatically), so the
"light install" is enough. Needs an Ampere-or-newer GPU (bf16). If `nvidia-smi` reports CUDA < 12.6,
swap `cu126` for the matching wheel index.

## Run order

`run_overnight.sh <gpu>` runs steps 1 and 3–5 unattended on one GPU (downloading the weights first if
missing), logs to `models/overnight.log`, stops at the first error, and skips finished steps on rerun.
The layer for steps 4–5 is the one the probe chose most often. The CNN (step 2) and evaluation
(step 6) run on the Mac.


1. **Smoke tests** (minutes): `train_cnn.py --smoke`, `extract_evo2_embeddings.py --smoke`,
   `train_evo2_lora.py --smoke`. The LoRA smoke prints LoRA tensors with gradient (expect half
   nonzero: B matrices) and peak GPU memory. If it runs out of memory: `--batch-size 4 --grad-accum 8`.
2. **CNN**, all folds: `train_cnn.py`.
3. **Embeddings** once, then **probe** on CPU. The probe prints the layer chosen per fold:
   use the most common one as `--rep-layer` for steps 4–5 (default 26 until you know).
4. **Frozen arm (same head)**: `train_evo2_lora.py --no-lora --fold k` for k = 0..4.
5. **LoRA arm**: `train_evo2_lora.py --fold k` for k = 0..4. Folds are independent: spread them
   over your GPUs, one process per GPU.
6. **Evaluate**: `eval_models.py --preds <models>/*_preds.csv`.

Secondary (after the primary comparison is done): `--train-frac 0.1/0.25/0.5` on the CNN and LoRA
(data efficiency); `--soft-2of4 0.75` (soft-label test).

## Cost estimates (unmeasured — read the real numbers from the smoke tests and first epoch)

A 7B forward over 500 bp is roughly 7 TFLOPs. On an A100 that is about 0.1 s per sequence
per strand, so: embeddings ≈ 45–60 min; frozen arm ≈ 30 min per fold; LoRA ≈ 30 min per
epoch per fold, up to 6 epochs → ~3 h per fold. CNN: minutes per fold.

## What was verified before handing over

- CNN: full smoke run on CPU (train → early stop → reload → fwd/rc predictions → evaluation).
- Evo 2 scripts: run end to end against a tiny CPU mock with Evo 2's module layout
  (embedding_layer, blocks returning `(x, None)`, Wqkv/out_proj/out_filter_dense nn.Linear,
  TELinear projections). LoRA wrapped 7 modules, skipped TELinear, gradients reached LoRA,
  checkpoint reload and predictions worked; extraction → probe → eval chain worked.
- **Not verified: the real Evo 2.** Backward through vortex's Hyena/FFT layers, flash-attn with
  batch > 1, and memory are first exercised by the LoRA smoke test on the workstation. Evo 2's
  config has `max_batch_size: 1` (believed to affect only generation caching); if batched
  forward fails, retry with `--batch-size 1 --grad-accum 32`.

Built against evo2 0.6.0 / vtx 1.1.0 source (read 2026-10-25).
