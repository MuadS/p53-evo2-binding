"""p53_common.py — data, folds, tiers and metrics shared by every p53 model script.

Every model (CNN, frozen Evo 2 probe, Evo 2 + LoRA, k-mer / PWM baselines) must see the
same rows, the same chromosome folds and the same evaluation. Keep that logic here only.

Protocol (pre-specified, see coscproject/p53_dataset_v1/README.md):
  * grouped 5-fold CV over the `fold` column (chromosomes). For test fold k the validation
    fold is (k+1) % 5 (early stopping / model selection) and the other three train.
  * training rows: positive + random_neg + motif_neg + promoter_neg (hard 0/1 labels).
    pos_2of4 rows are only used when --soft-2of4 is given (secondary experiment).
  * evaluation always uses hard labels and reports each negative tier separately.
"""
import csv, os
import numpy as np, pandas as pd

NEG_TIERS = ['random_neg', 'motif_neg', 'promoter_neg']
TRAIN_TIERS = ['positive'] + NEG_TIERS
N_FOLDS = 5
COMP = str.maketrans('ACGT', 'TGCA')


def rc(s):
    return s.translate(COMP)[::-1]


def load_data(path, soft_2of4=None, train_frac=1.0, seed=42):
    """Rows used by a model, with a float `target` column for training.
    train_frac < 1 subsamples TRAINING rows later (see split); evaluation rows are never cut."""
    df = pd.read_csv(path)
    df['row'] = np.arange(len(df))            # position in the CSV = row in cached embeddings
    keep = df.tier.isin(TRAIN_TIERS) | ((df.tier == 'pos_2of4') & (soft_2of4 is not None))
    df = df[keep].reset_index(drop=True)
    df['target'] = df.label.astype(float)
    if soft_2of4 is not None:
        df.loc[df.tier == 'pos_2of4', 'target'] = float(soft_2of4)
    df.attrs['train_frac'], df.attrs['seed'] = train_frac, seed
    return df


def split(df, test_fold):
    """(train, val, test) for one CV fold. pos_2of4 never enters val or test.
    If train_frac < 1, the training part is subsampled within each tier (same seed for every model)."""
    val_fold = (test_fold + 1) % N_FOLDS
    evalable = df.tier != 'pos_2of4'
    test = df[(df.fold == test_fold) & evalable]
    val = df[(df.fold == val_fold) & evalable]
    train = df[~df.fold.isin([test_fold, val_fold])]
    frac = df.attrs.get('train_frac', 1.0)
    if frac < 1.0:
        train = train.groupby('tier', group_keys=False).sample(frac=frac, random_state=df.attrs.get('seed', 42))
    return train, val, test


def one_hot(seqs):
    """list of equal-length ACGT strings -> float32 [N, 4, L] (N = all zeros)."""
    lut = np.full(256, 4, np.uint8)
    for i, ch in enumerate(b'ACGT'):
        lut[ch] = i
    x = lut[np.frombuffer(''.join(seqs).encode(), np.uint8)].reshape(len(seqs), -1)
    out = np.zeros((len(seqs), 5, x.shape[1]), np.float32)
    np.put_along_axis(out, x[:, None, :], 1.0, axis=1)
    return out[:, :4]


# ----------------------------------------------------------------------------- metrics
def _auc(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    return roc_auc_score(y, s), average_precision_score(y, s)


def comparisons(df):
    """Named evaluation sets: positives vs each negative tier, plus the two diagnostic strata.
    Returns {name: boolean mask over df}. All sets use hard labels."""
    pos = df.tier == 'positive'
    out = {f'pos_vs_{t}': pos | (df.tier == t) for t in NEG_TIERS}
    out['pos_vs_all_neg'] = df.tier.isin(TRAIN_TIERS)
    # TSS-proximal positives vs promoter negatives: is the model reading more than "promoter"?
    out['tss_pos_vs_promoter_neg'] = (pos & (df.tss_dist <= 1000)) | (df.tier == 'promoter_neg')
    # motif-free positives (PWM no better than 90% of random windows) vs random and promoter negatives
    thr = df.loc[df.tier == 'random_neg', 'pwm'].quantile(0.90)
    mf = pos & (df.pwm <= thr)
    out['motiffree_pos_vs_random_neg'] = mf | (df.tier == 'random_neg')
    out['motiffree_pos_vs_promoter_neg'] = mf | (df.tier == 'promoter_neg')
    return out


def evaluate(df, score_col='score'):
    """Per-comparison AUROC / AUPRC / n_pos / n_neg on rows that have a score."""
    df = df[df[score_col].notna()]
    rows = []
    for name, m in comparisons(df).items():
        d = df[m]
        auroc, auprc = _auc(d.label.values.astype(int), d[score_col].values)
        rows.append(dict(comparison=name, auroc=auroc, auprc=auprc,
                         n_pos=int((d.label == 1).sum()), n_neg=int((d.label == 0).sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- logging
class EpochLog:
    """Append-only CSV, fsync'd every row so a disconnect never loses an epoch."""

    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row):
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow({k: row.get(k, '') for k in self.fields})
            f.flush(); os.fsync(f.fileno())


def save_predictions(test, scores, path, extra=None):
    """Test-fold predictions in the one format eval_models.py reads."""
    out = test[['loc', 'chr', 'tier', 'label', 'fold', 'support', 'gc', 'cpg', 'pwm', 'tss_dist']].copy()
    out['score'] = scores
    for k, v in (extra or {}).items():
        out[k] = v
    if path is not None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        out.to_csv(path, index=False)
    return out
