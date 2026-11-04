"""Build p53 dataset v1 from GSE132246 (OCI-AML3, Nutlin-3a, 4 replicates).

Source: GEO GSE132246, GSM3854900-GSM3854903 (Nutlin-3a 2.5uM Rep1-4). Peaks are the
depositors' MACS 1.4.2 calls (p < 1e-5, hg19, no input control deposited). All 16 sample
BEDs are in raw_peaks/; DMSO/CPI/Combo peaks are used only to exclude regions from negatives.

Positives: merged Nutlin peak regions supported by >= MIN_SUPPORT of 4 replicates, W bp centred
on the highest-scoring replicate peak in the region, lifted hg19 -> hg38. 2/4-support regions go
to ambiguous_2of4.csv (not trained on); 1/4 regions are dropped (motif rate ~ background).
Negatives: same chromosome as each positive, GC-matched (0.02 bins), outside the hg38 blacklist
and outside every peak from all 16 samples +- 1 kb. Same filters for both classes: canonical
chroms (chr1-22, X), blacklist, no N, full length.
Folds: chromosomes assigned to 5 folds, balanced by positive count. Use grouped CV over `fold`.

Requires: pip install pandas numpy pyfaidx pyliftover ; hg38.fa (+ .fai)
Usage:   python build_dataset_v1.py --fasta /path/to/hg38.fa
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, pyfaidx
from pyliftover import LiftOver

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--min-support", type=int, default=3)
ap.add_argument("--window", type=int, default=500)
ap.add_argument("--neg-pool", type=int, default=10, help="random candidates per positive before GC matching")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
W, rng = a.window, np.random.default_rng(a.seed)
lo = LiftOver(os.path.join(HERE, "ref/hg19ToHg38.over.chain.gz"))
fa = pyfaidx.Fasta(a.fasta)
L = {c: len(fa[c]) for c in CH}

def lift(c, x):
    r = lo.convert_coordinate(c, int(x))
    return r[0][1] if r and r[0][0] == c else None

def merge(iv):  # iv: DataFrame c,s,e -> dict chrom -> (starts, ends) merged & sorted
    out = {}
    for c, g in iv.sort_values(["c", "s"]).groupby("c"):
        ss, ee = [], []
        for s, e in zip(g.s, g.e):
            if ss and s <= ee[-1]: ee[-1] = max(ee[-1], e)
            else: ss.append(s); ee.append(e)
        out[c] = (np.array(ss), np.array(ee))
    return out

def overlaps(m, c, s, e):
    if c not in m: return False
    ss, ee = m[c]; i = np.searchsorted(ss, e) - 1
    return i >= 0 and ee[i] > s

def window(c, mid):
    s = int(mid) - W // 2
    if s < 0 or s + W > L[c]: return None
    q = str(fa[c][s:s + W]).upper()
    return (s, q) if "N" not in q else None

def gc(q): return (q.count("G") + q.count("C")) / len(q)

# ---- load all 16 samples (hg19) ----
P = []
for f in sorted(glob.glob(os.path.join(HERE, "raw_peaks/*_peaks.bed.gz"))):
    cond, rep = re.search(r"_(\w+)_Rep(\d)", f).groups()
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 4], names=["c", "s", "e", "score"])
    P.append(d[d.c.isin(CH)].assign(cond=cond, rep=int(rep)))
P = pd.concat(P, ignore_index=True)

# ---- exclusion set (hg38): blacklist + every peak from every sample, +-1 kb ----
bl = pd.read_csv(os.path.join(HERE, "ref/hg38-blacklist.v2.bed.gz"), sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
bl = bl[bl.c.isin(CH)]
ls_, le_ = P.apply(lambda r: lift(r.c, r.s), axis=1), P.apply(lambda r: lift(r.c, r.e), axis=1)
ok = ls_.notna() & le_.notna() & (le_ > ls_)
allpk = pd.DataFrame({"c": P.c[ok], "s": ls_[ok].astype(int) - 1000, "e": le_[ok].astype(int) + 1000})
BL, EXCL = merge(bl), merge(pd.concat([bl, allpk]))

# ---- Nutlin consensus regions (hg19), support = # replicates ----
nut = P[P.cond == "Nutlin"].sort_values(["c", "s"]).reset_index(drop=True)
g, pc, pe, grp = -1, None, -1, []
for c, s, e in zip(nut.c, nut.s, nut.e):
    if c != pc or s > pe: g += 1; pc, pe = c, e
    else: pe = max(pe, e)
    grp.append(g)
nut["g"] = grp
best = nut.loc[nut.groupby("g").score.idxmax()].set_index("g")
reg = nut.groupby("g").agg(c=("c", "first"), s19=("s", "min"), e19=("e", "max"), support=("rep", "nunique"))
reg["score"], reg["mid19"] = best.score, (best.s + best.e) // 2
reg = reg[reg.support >= 2]

rows = []
for r in reg.itertuples():
    m = lift(r.c, r.mid19)
    if m is None: continue
    w = window(r.c, m)
    if w is None or overlaps(BL, r.c, w[0], w[0] + W): continue
    rows.append(dict(chr=r.c, start=w[0], end=w[0] + W, seq=w[1], support=r.support, peak_score=r.score,
                     src=f"{r.c}:{r.s19}-{r.e19}(hg19)"))
pos_all = pd.DataFrame(rows)
amb = pos_all[pos_all.support < a.min_support]
pos = pos_all[pos_all.support >= a.min_support].copy()

# ---- negatives: same chromosome, GC-matched ----
pos["gc"] = pos.seq.map(gc)
negs = []
for c, gp in pos.groupby("chr"):
    cand = []
    while len(cand) < a.neg_pool * len(gp):
        s = int(rng.integers(0, L[c] - W))
        if overlaps(EXCL, c, s, s + W): continue
        w = window(c, s + W // 2)
        if w: cand.append((w[0], w[1], gc(w[1])))
    cand = pd.DataFrame(cand, columns=["start", "seq", "gc"])
    pool = {k: list(rng.permutation(v.index)) for k, v in cand.groupby((cand.gc / 0.02).round().astype(int))}
    for gb in (gp.gc / 0.02).round().astype(int):
        for d in (0, 1, -1, 2, -2, 3, -3):
            if pool.get(gb + d): negs.append(cand.loc[pool[gb + d].pop()].to_dict() | {"chr": c}); break
neg = pd.DataFrame(negs).assign(label=0, support=0, peak_score=np.nan, src="random")
neg["end"] = neg.start + W

df = pd.concat([pos.assign(label=1), neg], ignore_index=True).drop_duplicates("seq")
df["gc"] = df.seq.map(gc)

# ---- chromosome folds, balanced by positive count ----
npos = df[df.label == 1].chr.value_counts()
load, fold = [0] * a.folds, {}
for c, n in npos.items():
    k = int(np.argmin(load)); fold[c] = k; load[k] += n
df["fold"] = df.chr.map(fold)
df["loc"] = df.chr + ":" + df.start.astype(str) + "-" + df.end.astype(str)

cols = ["loc", "chr", "start", "end", "seq", "gc", "label", "support", "peak_score", "fold", "src"]
df[cols].to_csv(os.path.join(HERE, "dataset_v1.csv"), index=False)
amb.assign(gc=amb.seq.map(gc), fold=amb.chr.map(fold)).to_csv(os.path.join(HERE, "ambiguous_2of4.csv"), index=False)

print(f"positives (>= {a.min_support}/4): {(df.label == 1).sum()}  | ambiguous 2/4: {len(amb)}  | negatives: {(df.label == 0).sum()}")
print(df.groupby(["fold", "label"]).size().unstack().assign(chroms=pd.Series({k: ",".join(sorted([c for c in fold if fold[c] == k], key=lambda x: CH.index(x))) for k in range(a.folds)})))
print("mean GC by label:", df.groupby("label").gc.mean().round(4).to_dict())
