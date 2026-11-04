import sys, glob, gzip, re, numpy as np, pandas as pd
from pyliftover import LiftOver
V='../p53_dataset_v1'
lo = LiftOver(f'{V}/ref/hg19ToHg38.over.chain.gz')
d = pd.read_csv(f'{V}/dataset_v1_tiers.csv')
d = d[d.tier=='positive'].copy()
thr = pd.read_csv(f'{V}/dataset_v1_tiers.csv').query("tier=='random_neg'").pwm.quantile(.9)
hs = re.compile(r'(?=([AG]{3}C[AT][AT]G[CT]{3}))')
d['n_half'] = d.seq.str.upper().map(lambda s: len(hs.findall(s)))
d['grp'] = np.where(d.pwm > thr, '1 full motif', np.where(d.n_half > 0, '2 no full motif, half-site', '3 no half-site'))
d['mid'] = (d.start + d.end)//2

def load(cond):
    out = []
    for f in sorted(glob.glob(f'{V}/raw_peaks/*_{cond}_Rep*_peaks.bed.gz')):
        rep = re.search(r'Rep(\d)', f).group(1)
        p = pd.read_csv(f, sep='\t', header=None, comment='t').iloc[:, :5]
        p.columns = ['chr','s','e','name','score']
        L = [lo.convert_coordinate(c, int(m)) for c, m in zip(p.chr, (p.s+p.e)//2)]
        p['chr38'] = [x[0][0] if x else None for x in L]; p['m38'] = [x[0][1] if x else None for x in L]
        p['rep'] = rep
        out.append(p.dropna(subset=['chr38']))
    return pd.concat(out)

def hit(peaks, win=250):
    """per positive: number of replicates with a peak summit-ish midpoint within +-win of the window centre, and max score"""
    nrep = np.zeros(len(d), int); best = np.zeros(len(d))
    by = {c: g for c, g in peaks.groupby('chr38')}
    for i, (c, m) in enumerate(zip(d.chr, d.mid)):
        g = by.get(c)
        if g is None: continue
        h = g[(g.m38 - m).abs() <= win]
        if len(h): nrep[i] = h.rep.nunique(); best[i] = h.score.max()
    return nrep, best

for cond in ['DMSO', 'Nutlin']:
    n, b = hit(load(cond)); d[f'{cond}_reps'] = n; d[f'{cond}_score'] = b
d['dmso_any2'] = d.DMSO_reps >= 2
d['nut_over_dmso'] = d.Nutlin_score / d.DMSO_score.replace(0, np.nan)
print(d.groupby('grp').agg(n=('loc','size'), in_DMSO_2plus_reps=('dmso_any2','mean'), in_DMSO_any=('DMSO_reps', lambda x: (x>0).mean()),
      median_Nutlin_score=('Nutlin_score','median'), median_DMSO_score_when_present=('DMSO_score', lambda x: x[x>0].median()),
      median_Nutlin_over_DMSO=('nut_over_dmso','median'), tss_1kb=('tss_dist', lambda x:(x.abs()<=1000).mean())).round(3).to_string())
d[['loc','grp','n_half','pwm','tss_dist','DMSO_reps','DMSO_score','Nutlin_reps','Nutlin_score']].to_csv(f'{V}/positives_dmso_check.csv', index=False)
