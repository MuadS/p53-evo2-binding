import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '../model_training')
from sklearn.metrics import roc_auc_score as auc
R='../results'
full = pd.read_csv('../p53_dataset_v1/dataset_v1_tiers.csv')
lora = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/evo2/evo2_lora_fold*_preds.csv'))]).set_index('loc')
cnn = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/cnn_mac/cnn_fold*_preds.csv'))]).set_index('loc')
d = full.set_index('loc').join(lora[['score']].rename(columns={'score':'lora'})).join(cnn[['score']].rename(columns={'score':'cnn'}))
d = d[d.lora.notna()]
thr = d.loc[d.tier=='random_neg','pwm'].quantile(.9)
pos = d.tier=='positive'
d['grp'] = np.where(pos & (d.pwm<=thr), 'motif-free pos', np.where(pos, 'motif pos', d.tier))
d['tss1k'] = d.tss_dist.abs()<=1000
# per-fold rank-normalise scores so folds are comparable when pooled
for m in ['lora','cnn']: d[m+'_r'] = d.groupby('fold')[m].rank(pct=True)
d['seqU']=d.seq.str.upper()
# CpG obs/exp
def oe(s):
    c,g,n=s.count('C'),s.count('G'),len(s); return s.count('CG')*n/(c*g) if c*g else 0
d['cpg_oe']=d.seqU.map(oe)
d['cgi_like']=(d.gc>=0.5)&(d.cpg_oe>=0.6)
print('PWM threshold (90th pct random):', round(thr,3))
g = d.groupby('grp').agg(n=('lora','size'), tss_within_1kb=('tss1k','mean'), cgi_like=('cgi_like','mean'),
    gc=('gc','median'), cpg=('cpg','median'), pwm=('pwm','median'), support4of4=('support', lambda s:(s==4).mean()),
    peak_score=('peak_score','median'), lora_rank=('lora_r','median'), cnn_rank=('cnn_r','median'))
print(g.round(3).to_string())
for neg in ['promoter_neg','random_neg']:
    s = d[d.grp.isin(['motif-free pos',neg])]; y=(s.grp=='motif-free pos').astype(int)
    print(f'\nmotif-free pos vs {neg}: n_pos={y.sum()} n_neg={(1-y).sum()}')
    for f in ['gc','cpg','cpg_oe','pwm','lora_r','cnn_r']:
        print(f'  AUROC of {f:7s} alone: {auc(y,s[f]):.3f}')
    # LoRA within CpG strata (quintiles of cpg defined on the pooled set)
    s=s.assign(q=pd.qcut(s.cpg,5,labels=False,duplicates='drop'))
    rows=[]
    for q,gq in s.groupby('q'):
        yy=(gq.grp=='motif-free pos').astype(int)
        if yy.nunique()==2 and yy.sum()>=5: rows.append((q,yy.sum(),(1-yy).sum(),auc(yy,gq.lora_r)))
    print('  LoRA AUROC within CpG quintiles (q, n_pos, n_neg, auroc):', [(a,b,c,round(e,3)) for a,b,c,e in rows])
    w=[r[1] for r in rows]; print('  weighted mean within-stratum LoRA AUROC:', round(np.average([r[3] for r in rows],weights=w),3))
# within motif-free positives: is LoRA score just CpG/TSS?
mf=d[d.grp=='motif-free pos']
print('\nwithin motif-free positives: spearman(lora_rank, cpg)=%.2f  (lora_rank, gc)=%.2f  (lora_rank, pwm)=%.2f' %
      (mf.lora_r.corr(mf.cpg,method='spearman'), mf.lora_r.corr(mf.gc,method='spearman'), mf.lora_r.corr(mf.pwm,method='spearman')))
print('motif-free pos: TSS-near vs distal LoRA rank median:', mf.groupby('tss1k').lora_r.median().round(3).to_dict(), mf.tss1k.value_counts().to_dict())
pr=d[d.tier=='promoter_neg']
print('promoter_neg LoRA rank median: %.3f ; random_neg: %.3f' % (pr.lora_r.median(), d[d.tier=='random_neg'].lora_r.median()))
# half-site check: one perfect RRRCWWGYYY half-site?
import re
hs=re.compile(r'(?=([AG]{3}C[AT][AT]G[CT]{3}))')
d['n_half']=d.seqU.map(lambda s: len(hs.findall(s)))
print('\nmedian / mean perfect half-sites per window:', d.groupby('grp').n_half.agg(['median','mean']).round(2).to_dict('index'))

print('\n================ half-site breakdown')
def spaced_pair(s, max_sp=13):
    st=[m.start() for m in hs.finditer(s)]
    return any(0 < b-a-10 <= max_sp for a in st for b in st if b>a)  # two half-sites with a 1-13 bp spacer
def adjacent_pair(s):
    st=[m.start() for m in hs.finditer(s)]
    return any(b-a==10 for a in st for b in st)
d['spaced']=d.seqU.map(spaced_pair); d['adjacent']=d.seqU.map(adjacent_pair)
print(d.groupby('grp').agg(any_half=('n_half',lambda x:(x>=1).mean()), two_plus=('n_half',lambda x:(x>=2).mean()),
      spaced_pair=('spaced','mean'), adjacent_pair=('adjacent','mean')).round(3).to_string())
mf=d[d.grp=='motif-free pos'].copy()
mf['hcls']=np.where(mf.n_half==0,'0 half-sites','>=1 half-site')
print('\nmotif-free positives split:', mf.hcls.value_counts().to_dict())
print(mf.groupby('hcls').agg(lora_rank=('lora_r','median'), tss_1kb=('tss1k','mean'), cgi_like=('cgi_like','mean'),
      support4=('support',lambda s:(s==4).mean()), peak=('peak_score','median')).round(3).to_string())
pr=d[d.tier=='promoter_neg']
for c,gq in mf.groupby('hcls'):
    s=pd.concat([gq,pr]); y=(s.tier=='positive').astype(int)
    print(f'  {c} vs promoter_neg: LoRA AUROC {auc(y,s.lora_r):.3f}  CNN {auc(y,s.cnn_r):.3f}  (n_pos {y.sum()})')

print('\n================ half-site count as a baseline')
for neg in ['promoter_neg','random_neg']:
    s=d[d.grp.isin(['motif-free pos',neg])]; y=(s.grp=='motif-free pos').astype(int)
    print(f'motif-free vs {neg}: AUROC n_half {auc(y,s.n_half):.3f} | n_half+spaced {auc(y,s.n_half+2*s.spaced):.3f}')
