#!/usr/bin/env python3
"""
extract_evo2_embeddings.py — one frozen Evo 2 forward pass over every row, cached to disk.

For each requested block, the hidden state is mean-pooled over the 500 positions, separately
for the forward strand and the reverse complement, and the two are averaged (strand-symmetric
features; Evo 2 is causal, so each strand only sees left context). Output: one float32 .npy
per layer, [N, 4096], row order = the CSV's row order, plus rows.csv for alignment.

This is the "frozen Evo 2" arm. probe_evo2.py fits the probes on CPU from these files, so
choosing a layer costs one GPU pass instead of one training run per layer.

    python extract_evo2_embeddings.py --csv .../dataset_v1_tiers.csv --gpu 0 \
        --weights p53_evo2/weights/evo2_7b_base/evo2_7b_base.pt --smoke
    python extract_evo2_embeddings.py --csv .../dataset_v1_tiers.csv --gpu 0 --weights ...

Memory: blocks after the deepest requested layer are dropped before moving on.
"""
import argparse, os, sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--weights', default=None, help='local evo2_7b_base.pt (default: HuggingFace cache)')
    ap.add_argument('--model-name', default='evo2_7b_base')
    ap.add_argument('--out-dir', default='p53_evo2/embeddings')
    ap.add_argument('--gpu', required=True, help='ONE GPU id; vortex spreads blocks over every visible GPU')
    ap.add_argument('--layers', default='4,8,12,16,20,24,26,28,31', help='block indices to pool')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--smoke', action='store_true', help='first 64 rows only')
    return ap.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from evo2_utils import load_evo2, tokenize, forward_blocks, gpu_gb
from p53_common import rc


def main():
    layers = [int(x) for x in args.layers.split(',')]
    df = pd.read_csv(args.csv)
    if args.smoke:
        df = df.groupby('tier', group_keys=False).head(16).reset_index(drop=True)
    assert df.seq.str.len().nunique() == 1, 'windows must share one length (no padding path)'
    print(f'{len(df)} rows | layers {layers} | GPU {args.gpu}')

    t0 = time.time()
    _, m, tok = load_evo2(args.model_name, args.weights, keep_blocks=max(layers) + 1)
    dev = next(m.parameters()).device
    print(f'loaded in {time.time()-t0:.0f}s on {dev} | {gpu_gb():.1f} GB allocated')

    feats = {l: np.zeros((len(df), m.config.hidden_size), np.float32) for l in layers}
    seqs = df.seq.str.upper().tolist()
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), args.batch_size)):
            chunk = seqs[i:i + args.batch_size]
            acc = None
            for strand in (chunk, [rc(s) for s in chunk]):
                hs = forward_blocks(m, tokenize(tok, strand, dev), layers)
                pooled = {l: hs[l].float().mean(1) for l in layers}
                acc = pooled if acc is None else {l: (acc[l] + pooled[l]) / 2 for l in layers}
            for l in layers:
                feats[l][i:i + len(chunk)] = acc[l].cpu().numpy()
    peak = gpu_gb(peak=True)

    out = os.path.join(args.out_dir, 'smoke' if args.smoke else args.model_name)
    os.makedirs(out, exist_ok=True)
    for l in layers:
        bad = ~np.isfinite(feats[l]).all(1)
        if bad.any():
            print(f'  [warn] layer {l}: {bad.sum()} rows non-finite (activation overflow?)')
        np.save(os.path.join(out, f'layer{l}.npy'), feats[l])
    df.drop(columns='seq').to_csv(os.path.join(out, 'rows.csv'), index=False)
    print(f'done in {(time.time()-t0)/60:.1f} min | peak GPU {peak:.1f} GB -> {out}')


if __name__ == '__main__':
    main()
