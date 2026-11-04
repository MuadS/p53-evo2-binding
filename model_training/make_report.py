"""Build the shareable results workbook + figures from ~/coscproject/results."""
import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, os.path.expanduser('~/coscproject/model_training'))
from p53_common import comparisons
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule

R = os.path.expanduser('~/coscproject/results')
FIG = f'{R}/figures'
os.makedirs(FIG, exist_ok=True)
OUT = f'{R}/p53_results_2026-10-26.xlsx'

# palette (validated: first three categorical slots, all-pairs, light surface)
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BASE = '#a9a8a1'
COL = {'cnn': '#2a78d6', 'evo2_probe': '#eb6834', 'evo2_frozen': '#eb6834', 'evo2_lora': '#1baf7a'}
NAME = {'pwm': 'PWM motif score', 'kmer5': '5-mer logistic', 'kmer4_pwm': '4-mer + PWM logistic',
        'cnn': '1D CNN', 'evo2_probe': 'Evo 2 frozen + linear probe', 'evo2_frozen': 'Evo 2 frozen + trained head',
        'evo2_lora': 'Evo 2 + LoRA (blocks 0-4)'}
COMP = [('pos_vs_random_neg', 'vs random genome'),
        ('pos_vs_promoter_neg', 'vs promoters'),
        ('tss_pos_vs_promoter_neg', 'promoter-near sites vs promoters'),
        ('pos_vs_motif_neg', 'vs motif-containing unbound*'),
        ('pos_vs_all_neg', 'vs all negatives'),
        ('motiffree_pos_vs_random_neg', 'motif-free sites vs random (n=182)'),
        ('motiffree_pos_vs_promoter_neg', 'motif-free sites vs promoters (n=182)')]
CLAB = dict(COMP)
MODELS = ['pwm', 'kmer5', 'kmer4_pwm', 'cnn', 'evo2_probe', 'evo2_frozen', 'evo2_lora']

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': AXIS,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})

ev = pd.read_csv(f'{R}/eval_summary.csv')

# ---------- LoRA vs CNN paired bootstrap (same rows) ----------
def load_preds(pat):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
cnn = load_preds(f'{R}/cnn_mac/cnn_fold*_preds.csv').set_index('loc')
lora = load_preds(f'{R}/evo2/evo2_lora_fold*_preds.csv').set_index('loc')
both = lora[['tier', 'label', 'fold', 'pwm', 'tss_dist', 'score']].copy()
both = both.join(cnn[['score']], rsuffix='_cnn', how='inner')
masks = comparisons(both)
rng = np.random.default_rng(0)


def fold_gap(d, a, b, n_boot=1000):
    """Mean over test folds of AUROC(a) - AUROC(b) on the same rows; bootstrap resamples within each fold."""
    groups = [g for _, g in d.groupby('fold') if g.label.nunique() == 2]
    gaps = [roc_auc_score(g.label, g[a]) - roc_auc_score(g.label, g[b]) for g in groups]
    boots = []
    for _ in range(n_boot):
        gs = []
        for g in groups:
            y = g.label.values
            p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            i = np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))]
            gs.append(roc_auc_score(y[i], g[a].values[i]) - roc_auc_score(y[i], g[b].values[i]))
        boots.append(np.mean(gs))
    return np.mean(gaps), np.percentile(boots, 2.5), np.percentile(boots, 97.5), f'{sum(x > 0 for x in gaps)}/{len(gaps)}'


gap_rows = []
for c, _ in COMP:
    g, lo, hi, nb = fold_gap(both[masks[c]], 'score', 'score_cnn')
    gap_rows.append(dict(comparison=CLAB[c], lora_minus_cnn=g, ci_low=lo, ci_high=hi, folds_lora_better=nb))
gaps_lc = pd.DataFrame(gap_rows)

# every learned model vs the PWM score (the pwm column is in every preds file)
pv = {}
for m, pat in [('cnn', f'{R}/cnn_mac/cnn_fold*_preds.csv'), ('evo2_probe', f'{R}/evo2/evo2_probe_fold*_preds.csv'),
               ('evo2_frozen', f'{R}/evo2/evo2_frozen_fold*_preds.csv'), ('evo2_lora', f'{R}/evo2/evo2_lora_fold*_preds.csv')]:
    p = load_preds(pat)
    mk = comparisons(p)
    for c, _ in COMP:
        g, lo, hi, nb = fold_gap(p[mk[c]], 'score', 'pwm', 500)
        pv[(CLAB[c], NAME[m])] = f'{g:+.3f} [{lo:+.3f}, {hi:+.3f}]  better in {nb} folds'
gap_pwm = pd.Series(pv).unstack().reindex(index=[l for _, l in COMP], columns=[NAME[m] for m in MODELS[3:]])

# ---------- tables ----------
def fmt_table(col):
    t = ev.pivot(index='comparison', columns='model', values=col).reindex(index=[c for c, _ in COMP], columns=MODELS)
    t.index = [CLAB[c] for c in t.index]
    t.columns = [NAME[m] for m in t.columns]
    return t
auroc = fmt_table('auroc_mean').round(3)
auroc_sd = fmt_table('auroc_sd').round(3)
n = ev[ev.model == 'pwm'].set_index('comparison').reindex([c for c, _ in COMP])[['n_pos', 'n_neg']]
n.index = [CLAB[c] for c in n.index]

# per-fold test AUROC (recomputed from preds for every learned model + baselines from eval are folds-mean only)
pf_rows = []
for m, pat in [('cnn', f'{R}/cnn_mac/cnn_fold*_preds.csv'), ('evo2_probe', f'{R}/evo2/evo2_probe_fold*_preds.csv'),
               ('evo2_frozen', f'{R}/evo2/evo2_frozen_fold*_preds.csv'), ('evo2_lora', f'{R}/evo2/evo2_lora_fold*_preds.csv')]:
    p = load_preds(pat)
    for c in ['pos_vs_random_neg', 'pos_vs_promoter_neg', 'pos_vs_motif_neg']:
        neg = c.replace('pos_vs_', '')
        for k, g in p[p.tier.isin(['positive', neg])].groupby('fold'):
            pf_rows.append(dict(model=NAME[m], comparison=CLAB[c], test_fold=k, auroc=roc_auc_score(g.label, g.score)))
per_fold = pd.DataFrame(pf_rows).pivot_table(index=['model', 'comparison'], columns='test_fold', values='auroc').round(3)
per_fold.columns = [f'fold {k}' for k in per_fold.columns]

# epoch logs (long format)
ep = []
for m, pat in [('cnn', f'{R}/cnn_mac/cnn_fold*_log.csv'), ('evo2_frozen', f'{R}/evo2/evo2_frozen_fold*_log.csv'),
               ('evo2_lora', f'{R}/evo2/evo2_lora_fold*_log.csv')]:
    for f in sorted(glob.glob(pat)):
        d = pd.read_csv(f)
        d.insert(0, 'fold', int(re.search(r'fold(\d)', f).group(1)))
        d.insert(0, 'model', NAME[m])
        d['best_epoch'] = d.epoch == d.loc[d.val_auroc.idxmax(), 'epoch']
        ep.append(d)
epochs = pd.concat(ep, ignore_index=True).drop(columns=['lr'], errors='ignore')
epochs = epochs.rename(columns={'val_auroc': 'val_auroc_all_neg', 'val_auroc_random_neg': 'val_auroc_vs_random',
                                'val_auroc_motif_neg': 'val_auroc_vs_motif', 'val_auroc_promoter_neg': 'val_auroc_vs_promoter'})
epochs = epochs.round(4)

probe = pd.read_csv(f'{R}/evo2/evo2_probe_per_layer.csv')
probe_l = probe.loc[probe.groupby(['fold', 'layer']).val_auroc.idxmax()]
probe_tab = probe_l.pivot(index='layer', columns='fold', values='test_auroc').round(3)
probe_tab.columns = [f'fold {k} test AUROC' for k in probe_tab.columns]
probe_tab['mean'] = probe_tab.mean(axis=1).round(3)

# ---------- figures ----------
# 1. headline bars: best baseline vs CNN vs Evo 2 probe vs LoRA
show = ['pos_vs_random_neg', 'pos_vs_promoter_neg', 'tss_pos_vs_promoter_neg', 'pos_vs_motif_neg',
        'motiffree_pos_vs_promoter_neg']
A = ev.pivot(index='comparison', columns='model', values='auroc_mean')
S = ev.pivot(index='comparison', columns='model', values='auroc_sd')
best = A[['pwm', 'kmer5', 'kmer4_pwm']].max(axis=1)
best_sd = pd.Series({c: S.loc[c, A.loc[c, ['pwm', 'kmer5', 'kmer4_pwm']].astype(float).idxmax()] for c in A.index})
series = [('Best baseline (PWM or k-mer)', best, best_sd, BASE), ('1D CNN', A.cnn, S.cnn, COL['cnn']),
          ('Evo 2 frozen + probe', A.evo2_probe, S.evo2_probe, COL['evo2_probe']),
          ('Evo 2 + LoRA', A.evo2_lora, S.evo2_lora, COL['evo2_lora'])]
fig, ax = plt.subplots(figsize=(10, 5.6))
h, gap = 0.19, 0.02
y0 = np.arange(len(show))[::-1]
for i, (lab, v, sd, c) in enumerate(series):
    yy = y0 + (1.5 - i) * (h + gap)
    vals = v.reindex(show).values
    ax.barh(yy, vals - 0.5, left=0.5, height=h, color=c, label=lab, zorder=2)
    ax.errorbar(vals, yy, xerr=sd.reindex(show).values, fmt='none', ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
    for y, x, e in zip(yy, vals, sd.reindex(show).values):
        ax.text(x + e + 0.006, y, f'{x:.3f}', va='center', fontsize=8, color=INK2)
ax.set_yticks(y0, [CLAB[c] for c in show])
ax.set_xlim(0.5, 1.05)
ax.set_xlabel('Test AUROC, mean of 5 held-out chromosome folds (0.5 = chance)')
ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.tick_params(axis='y', length=0, labelcolor=INK)
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False, fontsize=9)
ax.set_title('p53 binding: test AUROC by comparison', loc='left', fontsize=12, color=INK, pad=28)
fig.text(0.01, 0.01, '* motif-containing negatives have a known construction flaw that makes them easier for learned '
         'models; treat that row as provisional. Error bars = SD over folds.', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{FIG}/1_test_auroc.png', dpi=160)
plt.close(fig)

# 2. training curves: small multiples, rows = train loss / val AUROC, cols = model
mods = [('1D CNN', COL['cnn']), ('Evo 2 frozen + trained head', COL['evo2_frozen']), ('Evo 2 + LoRA (blocks 0-4)', COL['evo2_lora'])]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey='row')
for j, (m, c) in enumerate(mods):
    d = epochs[epochs.model == m]
    for k, g in d.groupby('fold'):
        for i, col in enumerate(['train_loss', 'val_auroc_all_neg']):
            axes[i, j].plot(g.epoch, g[col], color=c, lw=1.4, alpha=0.75)
        b = g[g.best_epoch]
        axes[1, j].plot(b.epoch, b.val_auroc_all_neg, 'o', ms=6, color=c, mec=SURF, mew=1.5, zorder=3)
    axes[0, j].set_title(m, loc='left', fontsize=10.5, color=INK)
    axes[1, j].set_xlabel('epoch')
    for i in range(2):
        axes[i, j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for i in range(2):
        axes[i, j].yaxis.grid(True, color=GRID, lw=0.6)
axes[0, 0].set_ylabel('training loss')
axes[1, 0].set_ylabel('validation AUROC (all negatives)')
fig.suptitle('Training curves, one line per fold (dot = epoch kept by early stopping)', x=0.01, ha='left',
             fontsize=12, color=INK)
fig.text(0.01, 0.005, 'LoRA was capped at 6 epochs and was still improving in most folds. '
         'The frozen head trains for many cheap epochs on cached features.', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f'{FIG}/2_training_curves.png', dpi=160)
plt.close(fig)

# 3. probe by layer
fig, ax = plt.subplots(figsize=(7.5, 4.2))
mean = probe_tab['mean']
for k in range(5):
    ax.plot(probe_tab.index, probe_tab[f'fold {k} test AUROC'], color=COL['evo2_probe'], lw=0.8, alpha=0.35)
ax.plot(mean.index, mean.values, color=COL['evo2_probe'], lw=2, marker='o', ms=6, mec=SURF, mew=1.5)
for l, off, ha in [(4, (0, 9), 'center'), (31, (-10, -4), 'right')]:
    ax.annotate(f'{mean[l]:.3f}', (l, mean[l]), textcoords='offset points', xytext=off, ha=ha, fontsize=8.5, color=INK2)
ax.axhline(A.loc['pos_vs_all_neg', 'pwm'], color=BASE, lw=1, ls='--')
ax.text(31, A.loc['pos_vs_all_neg', 'pwm'] + 0.006, 'PWM motif score', ha='right', fontsize=8, color=MUTED)
ax.set_xticks(probe_tab.index)
ax.set_xlabel('Evo 2 layer (of 32) used as frozen features')
ax.set_ylabel('test AUROC, all negatives')
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_title('Frozen Evo 2: earlier layers are better for this task', loc='left', fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(f'{FIG}/3_probe_by_layer.png', dpi=160)
plt.close(fig)

# ---------- workbook ----------
readme = pd.DataFrame({'': [
    'COSC 4337 Group 11: p53 binding-site prediction, Evo 2 + LoRA vs 1D CNN. Results from the first full run (2026-10-26).',
    '',
    'DATA: GSE132246 p53 ChIP-seq in OCI-AML3 cells treated with Nutlin-3a. Positives = peaks found in at least 3 of 4 replicates '
    '(2,608), 500 bp windows, hg38. Negatives, three kinds: random genome GC-matched (2,606); windows that contain a p53-like motif '
    'but are not bound (2,608); promoters matched for CpG/GC (956).',
    'EVALUATION: 5-fold cross-validation with whole chromosomes held out, so no test window shares a chromosome with training. '
    'Numbers are test AUROC averaged over the 5 folds (0.5 = chance, 1.0 = perfect).',
    'MODELS: PWM = JASPAR p53 motif score, no training. k-mer = logistic regression on k-mer counts. 1D CNN = 1.1M-parameter CNN trained '
    'from scratch (Mac CPU). Evo 2 = 7B DNA language model: frozen features + linear probe, frozen features + trained head, and LoRA fine-tuning '
    '(workstation, RTX 5000 Ada).',
    '',
    'HEADLINE: Evo 2 + LoRA is the best model on every comparison. Its clearest win is on binding sites with no recognisable p53 motif '
    '(0.891 vs promoters; CNN 0.850, best baseline 0.717). On the easy comparisons (vs random genome / promoters) the motif score alone '
    'is already ~0.97, so there is little room left: LoRA and the CNN both edge past it by about 0.01, and LoRA beats the CNN by 0.004-0.012.',
    '',
    'CAVEATS (read before quoting numbers):',
    '0. The trained Evo 2 head and linear probe (frozen Evo 2) are clearly WORSE than the motif score on the easy comparisons: mean-pooled frozen features blur a 20 bp motif across 500 bp.',
    '1. Motif-containing negatives (rows marked *) have a construction flaw: their motifs are "too perfect" (both core halves intact in 91% '
    'vs 66% of real sites). Learned models can exploit this, so that row overstates them. It is being rebuilt.',
    '2. "Evo 2 + LoRA" used only the first 5 of Evo 2\'s 32 blocks (about 1/6 of the model). The linear probe picked layer 4 as best, and '
    'deeper layers were worse when mean-pooled (sheet Probe_by_layer). So this is not yet full-depth Evo 2 fine-tuning.',
    '3. LoRA was capped at 6 epochs and validation AUROC was still rising in most folds, so it is probably undertrained.',
    '4. Motif-free rows have only 182 positives: real effect sizes but wide uncertainty. These sites are DEFINED by a low PWM score, so '
    'their gap vs PWM is large by construction; compare them with the 5-mer baseline (0.717 / 0.741) instead.',
    '5. One training run per model (one seed). Gap sheets: mean over folds of the AUROC difference on identical test rows, with a 95% bootstrap CI resampled within each fold.',
    '',
    'SHEETS: AUROC (main table) | AUROC_SD (spread over folds) | Gap_vs_PWM | LoRA_vs_CNN | Per_fold | Epochs (every training epoch) | '
    'Probe_by_layer | Figures',
]})

with pd.ExcelWriter(OUT, engine='openpyxl') as xw:
    readme.to_excel(xw, sheet_name='README', index=False, header=False)
    auroc.join(n).to_excel(xw, sheet_name='AUROC')
    auroc_sd.to_excel(xw, sheet_name='AUROC_SD')
    gap_pwm.to_excel(xw, sheet_name='Gap_vs_PWM')
    gaps_lc.round(3).to_excel(xw, sheet_name='LoRA_vs_CNN', index=False)
    per_fold.to_excel(xw, sheet_name='Per_fold')
    epochs.to_excel(xw, sheet_name='Epochs', index=False)
    probe_tab.to_excel(xw, sheet_name='Probe_by_layer')

wb = load_workbook(OUT)
hdr = PatternFill('solid', fgColor='EEEDE8')
for ws in wb.worksheets:
    if ws.title != 'README':
        for c in ws[1]:
            c.font = Font(bold=True); c.fill = hdr
            c.alignment = Alignment(wrap_text=True, vertical='top')
        ws.freeze_panes = 'B2'
        ws.column_dimensions['A'].width = 40 if ws.title != 'Epochs' else 30
        for col in ws.iter_cols(min_col=2, max_col=ws.max_column):
            ws.column_dimensions[col[0].column_letter].width = 16
        ws.row_dimensions[1].height = 45
ws = wb['README']
ws.column_dimensions['A'].width = 160
for r in ws.iter_rows():
    r[0].alignment = Alignment(wrap_text=True, vertical='top')
ws['A1'].font = Font(bold=True, size=13)
for rng_ in ['B2:H8']:
    wb['AUROC'].conditional_formatting.add(rng_, ColorScaleRule(start_type='num', start_value=0.5, start_color='FFFFFF',
                                                                 end_type='num', end_value=1.0, end_color='7FC8A9'))
ws = wb['Epochs']
ws.auto_filter.ref = ws.dimensions
f = wb.create_sheet('Figures')
row = 1
for png, w in [('1_test_auroc.png', 1000), ('2_training_curves.png', 1100), ('3_probe_by_layer.png', 750)]:
    img = XLImage(f'{FIG}/{png}')
    scale = w / img.width
    img.width, img.height = int(img.width * scale), int(img.height * scale)
    f.add_image(img, f'A{row}')
    row += int(img.height / 20) + 2
wb.save(OUT)
print(auroc.to_string()); print(); print(gaps_lc.round(3).to_string()); print('->', OUT)
