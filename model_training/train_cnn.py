#!/usr/bin/env python3
"""
train_cnn.py — baseline 1D CNN for p53 binding, trained from scratch, grouped 5-fold CV.

Same rows, folds and evaluation as the Evo 2 scripts (p53_common.py). For each test fold k:
train on 3 folds, early-stop on fold (k+1)%5, predict fold k. Predictions from all folds are
concatenated into one CSV that eval_models.py scores.

    CUDA_VISIBLE_DEVICES is set from --gpu. Check nvidia-smi first.
    python train_cnn.py --csv p53_evo2/data/dataset_v1_tiers.csv --gpu 0 --smoke
    python train_cnn.py --csv p53_evo2/data/dataset_v1_tiers.csv --gpu 0
    python train_cnn.py ... --train-frac 0.25 --tag cnn_frac025          # data-efficiency curve
    python train_cnn.py ... --soft-2of4 0.75 --tag cnn_soft075          # soft-label experiment

Architecture (~1.1M params): 3 conv blocks (k=19/9/7, 256 ch, BN, ReLU, max-pool) ->
global max + mean pool -> dropout -> linear. Reverse-complement augmentation in training;
test score = mean of forward and reverse-complement logits (ChIP peaks have no strand).
"""
import argparse, os, sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tag', default='cnn')
    ap.add_argument('--out-dir', default='p53_evo2/models')
    ap.add_argument('--gpu', required=True, help='CUDA_VISIBLE_DEVICES; set before torch import. Check nvidia-smi.')
    ap.add_argument('--folds', default='0,1,2,3,4', help='test folds to run (comma list)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--min-delta', type=float, default=0.001)
    ap.add_argument('--train-frac', type=float, default=1.0, help='subsample training rows (learning curve)')
    ap.add_argument('--soft-2of4', type=float, default=None, help='include 2/4 peaks with this soft target')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='fold 0, 2 epochs, 300 rows/tier')
    return ap.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from p53_common import load_data, split, one_hot, evaluate, EpochLog, save_predictions


class CNN(nn.Module):
    def __init__(self, ch=256, dropout=0.3):
        super().__init__()
        def block(cin, cout, k, pool):
            layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU()]
            return nn.Sequential(*layers, nn.MaxPool1d(pool)) if pool > 1 else nn.Sequential(*layers)
        self.body = nn.Sequential(block(4, ch, 19, 4), block(ch, ch, 9, 4), block(ch, ch, 7, 1))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * ch, 1)

    def forward(self, x):                      # x: [B, 4, L]
        h = self.body(x)
        h = torch.cat([h.amax(-1), h.mean(-1)], dim=1)
        return self.out(self.drop(h)).squeeze(-1)


def revcomp_onehot(x):                         # A<->T, C<->G is channel reversal for ACGT order
    return x.flip(1).flip(2)


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        out.append(((model(xb) + model(revcomp_onehot(xb))) / 2).float().cpu())
    return torch.cat(out).numpy()


def run_fold(df, k, device):
    torch.manual_seed(args.seed + k); np.random.seed(args.seed + k)
    tr, va, te = split(df, k)
    if args.smoke:
        tr = tr.groupby('tier', group_keys=False).head(300)
        va, te = va.groupby('tier', group_keys=False).head(100), te.groupby('tier', group_keys=False).head(100)
    print(f'\n[fold {k}] train {len(tr)} | val {len(va)} | test {len(te)}  {dict(tr.tier.value_counts())}')
    Xtr, Xva, Xte = (torch.from_numpy(one_hot(d.seq.tolist())) for d in (tr, va, te))
    ytr = torch.tensor(tr.target.values, dtype=torch.float32)
    yva = va.label.values.astype(int)

    model = CNN(args.channels, args.dropout).to(device)
    if k == 0:
        print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    log = EpochLog(os.path.join(args.out_dir, f'{args.tag}_fold{k}_log.csv'),
                   ['epoch', 'train_loss', 'val_auroc', 'lr'])
    ckpt = os.path.join(args.out_dir, f'{args.tag}_fold{k}.pt')
    best, bad = -1.0, 0
    for ep in range(1, (2 if args.smoke else args.epochs) + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            flip = torch.rand(len(idx), device=device) < 0.5          # reverse-complement augmentation
            xb = torch.where(flip[:, None, None], revcomp_onehot(xb), xb)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        auc = roc_auc_score(yva, predict(model, Xva, device))
        sched.step(auc)
        log.write({'epoch': ep, 'train_loss': tot / len(perm), 'val_auroc': auc, 'lr': opt.param_groups[0]['lr']})
        print(f'  ep{ep:02d} train {tot/len(perm):.4f} | val AUROC {auc:.4f}')
        if auc > best + args.min_delta:
            best, bad = auc, 0; torch.save({'state_dict': model.state_dict(), 'epoch': ep, 'val_auroc': auc,
                                            'args': vars(args)}, ckpt)
        else:
            bad += 1
            if bad >= args.patience: print('  early stopping'); break
    model.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
    return save_predictions(te, predict(model, Xte, device),
                            os.path.join(args.out_dir, f'{args.tag}_fold{k}_preds.csv'), {'model': args.tag})


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
    df = load_data(args.csv, soft_2of4=args.soft_2of4, train_frac=args.train_frac, seed=args.seed)
    print('Loaded', df.shape, dict(df.tier.value_counts()))
    folds = [0] if args.smoke else [int(x) for x in args.folds.split(',')]
    preds = pd.concat([run_fold(df, k, device) for k in folds], ignore_index=True)
    path = os.path.join(args.out_dir, f'{args.tag}_preds.csv')
    preds.to_csv(path, index=False)
    print('\n' + evaluate(preds).round(4).to_string(index=False))
    print('predictions ->', path)


if __name__ == '__main__':
    main()
