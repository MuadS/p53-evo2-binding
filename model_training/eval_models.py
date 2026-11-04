#!/usr/bin/env python3
"""
eval_models.py — score every model on the same folds and tiers (CPU only).

Computes the non-learned / linear baselines itself (identical folds, train folds only), then
reads any number of *_preds.csv written by train_cnn.py / probe_evo2.py / train_evo2_lora.py.

For each model and comparison (positives vs each negative tier, plus the diagnostic strata):
  * AUROC per test fold -> mean +- sd over folds  (primary: fold models are never pooled)
  * paired bootstrap of (model - reference) AUROC on the pooled rows, 95% CI
    (reference = the PWM scanner by default; same rows for both, so the CI is on the gap)

    python eval_models.py --csv .../dataset_v1_tiers.csv \
        --preds p53_evo2/models/{cnn,evo2_probe,evo2_frozen_fold*,evo2_lora_fold*}_preds.csv
"""
import argparse, glob, itertools, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from p53_common import load_data, split, comparisons, save_predictions, N_FOLDS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--csv', required=True)
ap.add_argument('--preds', nargs='*', default=[], help='prediction CSVs (globs ok)')
ap.add_argument('--reference', default='pwm')
ap.add_argument('--n-boot', type=int, default=1000)
ap.add_argument('--out', default='eval_summary.csv')
args = ap.parse_args()


def kmer(seqs, k):
    idx = {''.join(t): i for i, t in enumerate(itertools.product('ACGT', repeat=k))}
    X = np.zeros((len(seqs), 4 ** k), np.float32)
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None: X[r, j] += 1
    return X


def baselines(df):
    """pwm (no training), 5-mer logistic, 4-mer + pwm logistic; per-fold predictions."""
    seqs = df.seq.str.upper().tolist()
    feats = {'kmer5': kmer(seqs, 5), 'kmer4_pwm': np.c_[kmer(seqs, 4), df[['pwm']].values]}
    C = {'kmer5': 0.05, 'kmer4_pwm': 0.1}
    out = []
    for k in range(N_FOLDS):
        tr, _, te = split(df, k)
        out.append(save_predictions(te, te.pwm.values, None, {'model': 'pwm'}))
        for name, X in feats.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C[name], max_iter=4000))
            m.fit(X[tr.index], tr.target.values > 0.5)
            out.append(save_predictions(te, m.predict_proba(X[te.index])[:, 1], None, {'model': name}))
    return pd.concat(out, ignore_index=True)


df = load_data(args.csv)
P = [baselines(df)]
for pat in args.preds:
    for f in sorted(glob.glob(pat)):
        p = pd.read_csv(f)
        p['model'] = p.get('model', re.sub(r'(_fold\d)?_preds\.csv$', '', os.path.basename(f)))
        P.append(p)
P = pd.concat(P, ignore_index=True).drop_duplicates(['model', 'loc'])

rng = np.random.default_rng(0)
ref = P[P.model == args.reference].set_index('loc').score
rows = []
for model, pm in P.groupby('model', sort=False):
    for name, mask in comparisons(pm).items():
        d = pm[mask]
        per_fold = [roc_auc_score(g.label, g.score) for _, g in d.groupby('fold') if g.label.nunique() == 2]
        r = dict(model=model, comparison=name, n_pos=int(d.label.sum()), n_neg=int((d.label == 0).sum()),
                 auroc_mean=np.mean(per_fold) if per_fold else np.nan,
                 auroc_sd=np.std(per_fold) if per_fold else np.nan, n_folds=len(per_fold))
        if model != args.reference and d.label.nunique() == 2:
            y, s, s0 = d.label.values, d.score.values, ref.reindex(d['loc']).values
            ok = ~np.isnan(s0)
            y, s, s0 = y[ok], s[ok], s0[ok]
            pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            diffs = []
            for _ in range(args.n_boot):
                b = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
                diffs.append(roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], s0[b]))
            r.update(delta_vs_ref=roc_auc_score(y, s) - roc_auc_score(y, s0),
                     delta_lo=np.percentile(diffs, 2.5), delta_hi=np.percentile(diffs, 97.5))
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
pd.set_option('display.width', 200)
tab = res.pivot(index='comparison', columns='model', values='auroc_mean').round(3)
print('AUROC, mean over test folds\n' + tab.to_string())
if 'delta_vs_ref' in res:
    d = res.dropna(subset=['delta_vs_ref'])
    d = d.assign(gap=d.apply(lambda r: f'{r.delta_vs_ref:+.3f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}]', axis=1))
    print(f'\nAUROC minus {args.reference} (paired bootstrap 95% CI)\n'
          + d.pivot(index='comparison', columns='model', values='gap').to_string())
print('\nn per comparison:', res[res.model == args.reference].set_index('comparison')[['n_pos', 'n_neg']].to_dict('index'))
print('->', args.out)
