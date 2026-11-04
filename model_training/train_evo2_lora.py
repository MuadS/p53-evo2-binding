#!/usr/bin/env python3
"""
train_evo2_lora.py — Evo 2 7B + LoRA + pooled linear head, one CV fold per run.

    # 1. smoke test (checks forward, backward, LoRA gradients, peak memory)
    python train_evo2_lora.py --csv .../dataset_v1_tiers.csv --gpu 0 --weights .../evo2_7b_base.pt --smoke
    # 2. frozen arm, SAME head (fast: backbone runs once, head trains on cached pooled features)
    python train_evo2_lora.py --csv ... --gpu 0 --weights ... --fold 0 --no-lora --rep-layer 26
    # 3. LoRA arm
    python train_evo2_lora.py --csv ... --gpu 0 --weights ... --fold 0 --rep-layer 26

Run folds 0-4 (one per GPU in parallel tmux sessions, or sequentially), then eval_models.py.
Pick --rep-layer from probe_evo2.py (most-chosen layer). Blocks after it are deleted at load:
they cannot affect the head, so running them would only cost memory and time.

Head (proposal design): mean-pool over the 500 positions -> LayerNorm -> dropout -> linear.
LayerNorm is added because block outputs are read before Evo 2's final norm and are large.
Reverse-complement augmentation in training; scores average forward and reverse complement.
Selection: best validation AUROC on fold (k+1)%5; test fold k is touched once, at the end.
Only the LoRA A/B matrices and the head are saved (MBs, not the 13 GB backbone).
"""
import argparse, os, sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--weights', default=None, help='local evo2_7b_base.pt (default: HuggingFace cache)')
    ap.add_argument('--model-name', default='evo2_7b_base')
    ap.add_argument('--tag', default=None, help='default: evo2_lora / evo2_frozen')
    ap.add_argument('--out-dir', default='p53_evo2/models')
    ap.add_argument('--gpu', required=True, help='ONE GPU id; vortex spreads blocks over every visible GPU')
    ap.add_argument('--fold', type=int, default=0, help='test fold (0-4)')
    ap.add_argument('--rep-layer', type=int, default=26, help='block whose output the head reads')
    ap.add_argument('--no-lora', action='store_true', help='frozen backbone, same head (frozen arm)')
    ap.add_argument('--lora-targets', default='Wqkv,out_proj,out_filter_dense',
                    help='leaf names of nn.Linear to adapt. Add l1,l2,l3 for MLP LoRA (more params)')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--head-epochs', type=int, default=60, help='--no-lora only: epochs over cached features')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=4, help='effective batch = batch-size * grad-accum')
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--warmup-steps', type=int, default=50, help='optimizer steps of linear warmup')
    ap.add_argument('--head-dropout', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--min-delta', type=float, default=0.001)
    ap.add_argument('--no-grad-ckpt', action='store_true', help='faster, much more memory')
    ap.add_argument('--train-frac', type=float, default=1.0)
    ap.add_argument('--soft-2of4', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='tiny data, 1 epoch, 4 steps, gradient check')
    return ap.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TAG = args.tag or ('evo2_frozen' if args.no_lora else 'evo2_lora')

import time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from evo2_utils import load_evo2, tokenize, forward_blocks, inject_lora, lora_state_dict, gpu_gb, make_autograd_safe
from p53_common import load_data, split, rc, evaluate, EpochLog, save_predictions, NEG_TIERS


class PoolHead(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.norm, self.drop, self.out = nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 1)
        nn.init.normal_(self.out.weight, std=0.01); nn.init.zeros_(self.out.bias)

    def forward(self, pooled):                  # [B, D] float32
        return self.out(self.drop(self.norm(pooled))).squeeze(-1)


def pooled_rep(m, tok, seqs, device, grad_ckpt):
    h = forward_blocks(m, tokenize(tok, seqs, device), [args.rep_layer], grad_ckpt=grad_ckpt)[args.rep_layer]
    return h.float().mean(1)


@torch.no_grad()
def score(m, tok, head, d, device, bs):
    """fwd/rc-averaged logits for every row of d."""
    head.eval(); out = []
    seqs = d.seq.str.upper().tolist()
    for i in tqdm(range(0, len(seqs), bs), desc='score', leave=False):
        c = seqs[i:i + bs]
        lf = head(pooled_rep(m, tok, c, device, False))
        lr = head(pooled_rep(m, tok, [rc(s) for s in c], device, False))
        out.append(((lf + lr) / 2).cpu())
    return torch.cat(out).numpy()


def tier_aucs(d, s):
    out = {}
    for t in NEG_TIERS:
        mk = d.tier.isin(['positive', t]).values
        out[f'val_auroc_{t}'] = roc_auc_score(d.label.values[mk], s[mk]) if mk.any() else float('nan')
    return out


def main():
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_data(args.csv, soft_2of4=args.soft_2of4, train_frac=args.train_frac, seed=args.seed)
    tr, va, te = split(df, args.fold)
    if args.smoke:
        tr = tr.groupby('tier', group_keys=False).head(32)
        va, te = va.groupby('tier', group_keys=False).head(16), te.groupby('tier', group_keys=False).head(16)
    print(f'{TAG} fold {args.fold} | train {len(tr)} val {len(va)} test {len(te)} | rep-layer {args.rep_layer}')

    t0 = time.time()
    _, m, tok = load_evo2(args.model_name, args.weights, keep_blocks=args.rep_layer + 1)
    print(f'loaded {len(m.blocks)} blocks in {time.time()-t0:.0f}s | {gpu_gb():.1f} GB')
    head = PoolHead(m.config.hidden_size, args.head_dropout).to(device)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f'{TAG}_fold{args.fold}')
    fields = ['epoch', 'train_loss', 'val_auroc'] + [f'val_auroc_{t}' for t in NEG_TIERS]
    log = EpochLog(stem + '_log.csv', fields)

    if args.no_lora:
        run_frozen(m, tok, head, tr, va, te, device, log, stem)
    else:
        run_lora(m, tok, head, tr, va, te, device, log, stem)


def run_frozen(m, tok, head, tr, va, te, device, log, stem):
    """Backbone runs once per row and strand; the head trains on cached pooled features."""
    def feats(d):
        f, r = [], []
        seqs = d.seq.str.upper().tolist()
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), args.batch_size * 2), desc='features', leave=False):
                c = seqs[i:i + args.batch_size * 2]
                f.append(pooled_rep(m, tok, c, device, False)); r.append(pooled_rep(m, tok, [rc(s) for s in c], device, False))
        return torch.cat(f), torch.cat(r)
    (Ftr, Rtr), (Fva, Rva), (Fte, Rte) = feats(tr), feats(va), feats(te)
    y = torch.tensor(tr.target.values, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=0.01)
    best, bad = -1.0, 0
    for ep in range(1, (2 if args.smoke else args.head_epochs) + 1):
        head.train(); perm = torch.randperm(len(y), device=device); tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            x = torch.where((torch.rand(len(idx), device=device) < 0.5)[:, None], Rtr[idx], Ftr[idx])
            loss = F.binary_cross_entropy_with_logits(head(x), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            s = ((head(Fva) + head(Rva)) / 2).cpu().numpy()
        auc = roc_auc_score(va.label, s)
        log.write({'epoch': ep, 'train_loss': tot / len(y), 'val_auroc': auc, **tier_aucs(va, s)})
        if auc > best + args.min_delta:
            best, bad = auc, 0; torch.save({'head': head.state_dict(), 'epoch': ep, 'val_auroc': auc, 'args': vars(args)}, stem + '.pt')
        else:
            bad += 1
            if bad >= 8: break
    print(f'best val AUROC {best:.4f}')
    head.load_state_dict(torch.load(stem + '.pt', map_location=device)['head']); head.eval()
    with torch.no_grad():
        s = ((head(Fte) + head(Rte)) / 2).cpu().numpy()
    finish(te, s, stem)


def run_lora(m, tok, head, tr, va, te, device, log, stem):
    print(f'converted {make_autograd_safe(m)} inference-mode tensors to normal tensors (needed for backward)')
    wrapped = inject_lora(m, args.lora_targets.split(','), args.rep_layer, args.lora_r, args.lora_alpha, args.lora_dropout)
    lora_params = [p for n, p in m.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    for p in lora_params: p.requires_grad = True
    print(f'LoRA on {len(wrapped)} modules (e.g. {wrapped[:3]}) | {sum(p.numel() for p in lora_params)/1e6:.2f}M LoRA params')
    if not wrapped:
        raise SystemExit('no modules matched --lora-targets')
    opt = torch.optim.AdamW([{'params': lora_params, 'lr': args.lr_lora, 'weight_decay': 0.0},
                             {'params': head.parameters(), 'lr': args.lr_head, 'weight_decay': 0.01}])
    base_lrs = [g['lr'] for g in opt.param_groups]
    grad_ckpt = not args.no_grad_ckpt
    seqs, y = tr.seq.str.upper().tolist(), tr.target.values
    best, bad, gstep = -1.0, 0, 0
    for ep in range(1, (1 if args.smoke else args.epochs) + 1):
        m.train(); head.train()
        perm = np.random.permutation(len(seqs)); tot, n = 0.0, 0
        steps = range(0, len(perm), args.batch_size)
        for si, i in enumerate(tqdm(steps, desc=f'ep{ep}', leave=False)):
            if args.smoke and si >= 4: break
            idx = perm[i:i + args.batch_size]
            batch = [seqs[j] for j in idx]
            if np.random.rand() < 0.5: batch = [rc(s) for s in batch]
            yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(head(pooled_rep(m, tok, batch, device, grad_ckpt)), yb)
            (loss / args.grad_accum).backward()
            if args.smoke and si == 0:
                g = [p.grad.abs().sum().item() for p in lora_params if p.grad is not None]
                print(f'  smoke: LoRA tensors with grad {len(g)}/{len(lora_params)}, nonzero {sum(x > 0 for x in g)} '
                      f'(expect half: B is nonzero, A is exactly 0 on step 1 because B starts at 0) | '
                      f'loss {loss.item():.4f} | peak {gpu_gb(peak=True):.1f} GB')
            if (si + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                gstep += 1
                f = min(1.0, gstep / max(args.warmup_steps, 1))
                for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * f
                opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(idx); n += len(idx)
        m.eval()
        s = score(m, tok, head, va, device, args.batch_size * 2)
        auc = roc_auc_score(va.label, s)
        log.write({'epoch': ep, 'train_loss': tot / n, 'val_auroc': auc, **tier_aucs(va, s)})
        print(f'ep{ep} train {tot/n:.4f} | val AUROC {auc:.4f} | peak {gpu_gb(peak=True):.1f} GB')
        if auc > best + args.min_delta:
            best, bad = auc, 0
            torch.save({'lora': lora_state_dict(m), 'head': head.state_dict(), 'epoch': ep, 'val_auroc': auc,
                        'wrapped': wrapped, 'args': vars(args)}, stem + '.pt')
        else:
            bad += 1
            if bad >= args.patience: print('early stopping'); break
    ck = torch.load(stem + '.pt', map_location=device)
    m.load_state_dict(ck['lora'], strict=False); head.load_state_dict(ck['head'])
    m.eval()
    finish(te, score(m, tok, head, te, device, args.batch_size * 2), stem)


def finish(te, s, stem):
    preds = save_predictions(te, s, stem + '_preds.csv', {'model': TAG})
    print('\n' + evaluate(preds).round(4).to_string(index=False))
    print('predictions ->', stem + '_preds.csv')


if __name__ == '__main__':
    main()
