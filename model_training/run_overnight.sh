#!/usr/bin/env bash
# run_overnight.sh — unattended Evo 2 pipeline on one GPU. Stops at the first failure.
#
#   download weights (if missing) -> smoke tests -> frozen embeddings -> linear probe
#   -> frozen-head arm, folds 0-4 (layer = the one the probe chose most often)
#   -> LoRA arm, folds 0-4 (slowest; each fold writes its own predictions, so stopping
#      part-way still leaves usable folds)
#
# Run inside tmux with the evo2 env active:
#   bash run_overnight.sh 3          # 3 = GPU id
# Progress: tail -f p53_evo2/models/overnight.log
# Each step is skipped if its output already exists, so rerunning resumes where it stopped.
set -euo pipefail
G=${1:?usage: bash run_overnight.sh <gpu id>}
R=p53_evo2
D=$R/data/dataset_v1_tiers.csv
W=$R/weights/evo2_7b_base/evo2_7b_base.pt
M=$R/models
LOG=$M/overnight.log
cd "$(dirname "$0")"
mkdir -p "$M"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date '+%F %T') | $*"; }

step "start | GPU $G | $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
python -c "import evo2, flash_attn, sklearn, pandas" || { echo "evo2 env incomplete — conda activate evo2 / finish the install"; exit 1; }
nvidia-smi -i "$G" --query-gpu=name,memory.used,utilization.gpu --format=csv

if [ ! -s "$W" ]; then
  step "download weights"
  hf download arcinstitute/evo2_7b_base evo2_7b_base.pt --local-dir "$(dirname "$W")" \
    || huggingface-cli download arcinstitute/evo2_7b_base evo2_7b_base.pt --local-dir "$(dirname "$W")"
fi
ls -lh "$W"

if [ ! -f "$M/.smoke_ok" ]; then
  step "smoke: extraction";  python extract_evo2_embeddings.py --csv "$D" --gpu "$G" --weights "$W" --smoke
  step "smoke: LoRA";        python train_evo2_lora.py --csv "$D" --gpu "$G" --weights "$W" --smoke --out-dir "$M/smoke"
  touch "$M/.smoke_ok"
fi

if [ ! -f "$R/embeddings/evo2_7b_base/rows.csv" ]; then
  step "frozen embeddings"; python extract_evo2_embeddings.py --csv "$D" --gpu "$G" --weights "$W"
fi

if [ ! -f "$M/evo2_probe_preds.csv" ]; then
  step "linear probe (CPU)"; python probe_evo2.py --csv "$D" --emb-dir "$R/embeddings/evo2_7b_base" --out-dir "$M"
fi
L=$(python -c "import pandas as pd; print(int(pd.read_csv('$M/evo2_probe_preds.csv').groupby('fold').layer.first().mode()[0]))")
step "rep-layer = $L (most-chosen probe layer)"

for k in 0 1 2 3 4; do
  [ -f "$M/evo2_frozen_fold${k}_preds.csv" ] && continue
  step "frozen head, fold $k"; python train_evo2_lora.py --csv "$D" --gpu "$G" --weights "$W" --fold "$k" --rep-layer "$L" --no-lora --out-dir "$M"
done

for k in 0 1 2 3 4; do
  [ -f "$M/evo2_lora_fold${k}_preds.csv" ] && continue
  step "LoRA, fold $k"; python train_evo2_lora.py --csv "$D" --gpu "$G" --weights "$W" --fold "$k" --rep-layer "$L" --out-dir "$M"
done

step "all done"
