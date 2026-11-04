#!/usr/bin/env python3
"""
probe_evo2.py — frozen Evo 2 arm: logistic probes on cached embeddings (CPU only).

For each test fold k: fit a standardized logistic regression for every (layer, C) on the
three training folds, pick the pair with the best AUROC on validation fold (k+1)%5, and
score fold k with it. The layer is therefore chosen without looking at test data.
Also prints every layer's test AUROC (exploratory: it is chosen-on-test if you read it that
way) and writes the chosen layer per fold, which is the --rep-layer to give train_evo2_lora.py.

    python probe_evo2.py --csv .../dataset_v1_tiers.csv --emb-dir p53_evo2/embeddings/evo2_7b_base
"""
import argparse, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from p53_common import load_data, split, evaluate, save_predictions

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--csv', required=True)
ap.add_argument('--emb-dir', required=True)
ap.add_argument('--out-dir', default='p53_evo2/models')
ap.add_argument('--tag', default='evo2_probe')
ap.add_argument('--C', default='0.001,0.01,0.1')
ap.add_argument('--train-frac', type=float, default=1.0)
ap.add_argument('--n-jobs', type=int, default=8)
args = ap.parse_args()

df = load_data(args.csv, train_frac=args.train_frac)
rows = pd.read_csv(os.path.join(args.emb_dir, 'rows.csv'))
assert len(rows) == len(pd.read_csv(args.csv, usecols=['loc'])), 'embeddings were extracted from a different CSV'
E = {int(re.search(r'layer(\d+)', f).group(1)): np.load(f, mmap_mode='r')
     for f in glob.glob(os.path.join(args.emb_dir, 'layer*.npy'))}
layers, Cs = sorted(E), [float(c) for c in args.C.split(',')]
print(f'layers {layers} | C {Cs} | rows {len(df)}')

preds, per_layer = [], []
for k in range(5):
    tr, va, te = split(df, k)
    best = (-1, None, None)
    for l in layers:
        X = E[l]
        for C in Cs:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, n_jobs=args.n_jobs))
            m.fit(np.asarray(X[tr.row.values]), tr.target.values > 0.5)
            v = roc_auc_score(va.label, m.predict_proba(np.asarray(X[va.row.values]))[:, 1])
            t = roc_auc_score(te.label, m.predict_proba(np.asarray(X[te.row.values]))[:, 1])
            per_layer.append(dict(fold=k, layer=l, C=C, val_auroc=v, test_auroc=t))
            if v > best[0]:
                best = (v, (l, C), m)
    (l, C), m = best[1], best[2]
    print(f'fold {k}: chose layer {l}, C={C} (val AUROC {best[0]:.4f})')
    preds.append(save_predictions(te, m.predict_proba(np.asarray(E[l][te.row.values]))[:, 1],
                                  os.path.join(args.out_dir, f'{args.tag}_fold{k}_preds.csv'),
                                  {'model': args.tag, 'layer': l, 'C': C}))

pl = pd.DataFrame(per_layer)
pl.to_csv(os.path.join(args.out_dir, f'{args.tag}_per_layer.csv'), index=False)
print('\nmean over folds, best C per layer (exploratory):')
print(pl.loc[pl.groupby(['fold', 'layer']).val_auroc.idxmax()].groupby('layer')[['val_auroc', 'test_auroc']]
      .mean().round(4).to_string())
preds = pd.concat(preds, ignore_index=True)
path = os.path.join(args.out_dir, f'{args.tag}_preds.csv')
preds.to_csv(path, index=False)
print('\n' + evaluate(preds).round(4).to_string(index=False))
print('chosen layers per fold:', preds.groupby('fold').layer.first().to_dict(), '\npredictions ->', path)
