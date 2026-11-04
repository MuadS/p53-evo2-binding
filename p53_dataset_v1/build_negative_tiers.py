"""Add hard negative tiers to dataset_v1.csv -> dataset_v1_tiers.csv (one file, every model uses it).

Tiers (column `tier`):
  positive      Nutlin peaks in >= 3/4 replicates (from dataset_v1.csv)
  random_neg    random same-chromosome, GC-matched windows (from dataset_v1.csv)
  motif_neg     unbound windows containing a strong p53 motif, matched to positives on
                window max PWM score x GC, within each fold. Asks: motif present, bound or not?
  promoter_neg  unbound windows near a GENCODE TSS, matched on CpG density x GC to the
                TSS-proximal positives (<= 1 kb), 2x their count, within each fold.
                Asks: is a model just detecting CpG-island promoters?
  pos_2of4      2/4-replicate peaks (from ambiguous_2of4.csv). NOT used for training or
                evaluation unless a script is told to (soft-label experiment).

Unbound = outside the hg38 blacklist and outside every peak from all 16 GSE132246 samples
+- 1 kb. Negative windows place their motif / TSS at offsets drawn from the positives' own
offset distribution, so "motif exactly at the centre" cannot separate the classes.

Motif scan: JASPAR MA0106.3 (ref/MA0106.3.jaspar). Genome candidates are pre-filtered on the
two C-W-W-G cores at the matrix's fixed spacing (10 bp, the 0-spacer full site) and then
scored with the full matrix; every window's final `pwm` is recomputed with the same scanner
used for positives.

Requires: pandas numpy pyfaidx pyliftover; hg38.fa; GENCODE GTF (hg38).
Usage:    python build_negative_tiers.py --fasta hg38.fa --gtf gencode.v49.annotation.gtf.gz
"""
import argparse, glob, gzip, os, re
import numpy as np, pandas as pd, pyfaidx
from pyliftover import LiftOver

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--gtf", required=True)
ap.add_argument("--min-pwm", type=float, default=0.70, help="lowest window PWM score kept in the motif_neg pool")
ap.add_argument("--promoter-mult", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
W, rng = 500, np.random.default_rng(a.seed)
fa = pyfaidx.Fasta(a.fasta)
L = {c: len(fa[c]) for c in CH}

# ---------------------------------------------------------------- motif scoring
rows = [l for l in open(os.path.join(HERE, "ref/MA0106.3.jaspar")) if not l.startswith(">")]
M = np.array([[float(x) for x in re.findall(r"[\d.]+", l)] for l in rows])
LO = np.log2(((M + 0.8) / (M + 0.8).sum(0)) / 0.25)          # 4 x 18 log-odds, ACGT
K = LO.shape[1]
LOMIN, LOMAX = LO.min(0).sum(), LO.max(0).sum()
LUT = np.full(256, 4, np.uint8)
for i, ch in enumerate(b"ACGT"): LUT[ch] = i
LO5 = np.vstack([LO, np.full((1, K), -50.0)])                 # row 4 = N/other, never a hit

def encode(s): return LUT[np.frombuffer(s.upper().encode(), np.uint8)]

def pwm_scan(s):
    """(relative max score over both strands, offset of best hit centre from window centre)"""
    x = encode(s); rc = np.where(x < 4, 3 - x, 4)[::-1]
    best, off = -1e9, 0
    for arr, flip in ((x, False), (rc, True)):
        w = np.lib.stride_tricks.sliding_window_view(arr, K)
        sc = LO5[w, np.arange(K)].sum(1); i = int(sc.argmax())
        if sc[i] > best:
            c = i + K // 2
            best, off = sc[i], (len(s) - 1 - c if flip else c) - len(s) // 2
    return (best - LOMIN) / (LOMAX - LOMIN), off

def gc(q): return (q.count("G") + q.count("C")) / len(q)
def cpg(q): return q.count("CG") / len(q)

# ---------------------------------------------------------------- exclusion (same as v1)
def merge(iv):
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

lo = LiftOver(os.path.join(HERE, "ref/hg19ToHg38.over.chain.gz"))
def lift(c, x):
    r = lo.convert_coordinate(c, int(x)); return r[0][1] if r and r[0][0] == c else None
P = []
for f in sorted(glob.glob(os.path.join(HERE, "raw_peaks/*_peaks.bed.gz"))):
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); P.append(d[d.c.isin(CH)])
P = pd.concat(P, ignore_index=True)
ls_, le_ = P.apply(lambda r: lift(r.c, r.s), axis=1), P.apply(lambda r: lift(r.c, r.e), axis=1)
ok = ls_.notna() & le_.notna() & (le_ > ls_)
bl = pd.read_csv(os.path.join(HERE, "ref/hg38-blacklist.v2.bed.gz"), sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
EXCL = merge(pd.concat([bl[bl.c.isin(CH)],
                        pd.DataFrame({"c": P.c[ok], "s": ls_[ok].astype(int) - 1000, "e": le_[ok].astype(int) + 1000})]))

def window(c, mid):
    s = int(mid) - W // 2
    if s < 0 or s + W > L[c] or overlaps(EXCL, c, s, s + W): return None
    q = str(fa[c][s:s + W]).upper()
    return (s, q) if "N" not in q else None

# ---------------------------------------------------------------- TSS
tss = {}
with gzip.open(a.gtf, "rt") as f:
    for l in f:
        if l[0] == "#": continue
        x = l.split("\t", 8)
        if x[2] == "transcript" and x[0] in L:
            tss.setdefault(x[0], []).append(int(x[3]) - 1 if x[6] == "+" else int(x[4]) - 1)
TSS = {c: np.unique(v) for c, v in tss.items()}
def tss_dist(c, s):
    v, m = TSS[c], s + W // 2; i = np.searchsorted(v, m)
    return int(min(abs(v[max(i - 1, 0)] - m), abs(v[min(i, len(v) - 1)] - m)))

# ---------------------------------------------------------------- base rows
base = pd.read_csv(os.path.join(HERE, "dataset_v1.csv"))
base["tier"] = np.where(base.label == 1, "positive", "random_neg")
amb = pd.read_csv(os.path.join(HERE, "ambiguous_2of4.csv")).assign(tier="pos_2of4", label=1)
amb["loc"] = amb.chr + ":" + amb.start.astype(str) + "-" + amb.end.astype(str)
FOLD = base.groupby("chr").fold.first().to_dict()
amb["fold"] = amb.chr.map(FOLD)
base = pd.concat([base, amb[base.columns]], ignore_index=True)

def annotate(df):
    r = df.seq.map(pwm_scan)
    df["pwm"], df["pwm_offset"] = r.str[0], r.str[1]
    df["cpg"] = df.seq.map(cpg)
    df["tss_dist"] = [tss_dist(c, s) for c, s in zip(df.chr, df.start)]
    return df
base = annotate(base)
pos = base[base.tier == "positive"]
OFFS = pos.pwm_offset.values                                    # empirical motif offsets in positives
print(f"positives: {len(pos)} | motif offset from centre: median |off| {np.median(np.abs(OFFS)):.0f} bp")

def match(pool, target, keys, widths, n_mult=1):
    """For each target row pick an unused pool row from the same fold and the same
    (key/width) bin, widening one bin at a time when a bin runs out."""
    pool, picks = pool.copy(), []
    for k, wd in zip(keys, widths):
        pool[k + "_b"] = (pool[k] / wd).round().astype(int)
        target = target.assign(**{k + "_b": (target[k] / wd).round().astype(int)})
    bins = {}
    for key, g in pool.groupby(["fold"] + [k + "_b" for k in keys]):
        bins[key] = list(rng.permutation(g.index))
    for _ in range(n_mult):
        for t in target.itertuples():
            b0 = tuple(getattr(t, k + "_b") for k in keys)
            done = False
            for r in range(0, 6):                                   # widen search radius
                for d in np.ndindex(*(2 * r + 1,) * len(keys)):
                    key = (t.fold,) + tuple(x + dx - r for x, dx in zip(b0, d))
                    if bins.get(key):
                        picks.append(bins[key].pop()); done = True; break
                if done: break
    return pool.loc[picks].drop(columns=[k + "_b" for k in keys])

# ---------------------------------------------------------------- motif_neg pool: genome scan
cand = []
for c in CH:
    x = encode(str(fa[c][:]))
    n = len(x) - K
    m = np.ones(n, bool)
    for j, allowed in ((2, (1,)), (3, (0, 3)), (4, (0, 3)), (5, (2,)), (12, (1,)), (13, (0, 3)), (14, (0, 3)), (15, (2,))):
        col = x[j:j + n]
        m &= np.isin(col, allowed) if len(allowed) > 1 else (col == allowed[0])
    idx = np.flatnonzero(m)
    w = x[idx[:, None] + np.arange(K)]
    sf = LO5[w, np.arange(K)].sum(1)
    sr = LO5[np.where(w < 4, 3 - w, 4)[:, ::-1], np.arange(K)].sum(1)
    rel = (np.maximum(sf, sr) - LOMIN) / (LOMAX - LOMIN)
    keep = idx[rel >= a.min_pwm] + K // 2
    rows = []
    for site in keep:
        wv = window(c, site - rng.choice(OFFS))
        if wv: rows.append((c, wv[0], wv[1]))
    cand += rows
    print(f"  {c}: {len(idx):7d} core pairs, {len(keep):6d} >= {a.min_pwm}, {len(rows):6d} usable windows")
mpool = pd.DataFrame(cand, columns=["chr", "start", "seq"]).drop_duplicates("start")
mpool = mpool[~mpool.seq.isin(set(base.seq))]
mpool["fold"], mpool["gc"] = mpool.chr.map(FOLD), mpool.seq.map(gc)
mpool = mpool[mpool.fold.notna()]
mpool["pwm"] = mpool.seq.map(lambda s: pwm_scan(s)[0])
motif_neg = match(mpool, pos, ["pwm", "gc"], [0.02, 0.04])

# ---------------------------------------------------------------- promoter_neg pool
pos_tss = pos[pos.tss_dist <= 1000]
dist = pos_tss.tss_dist.values
prow = []
for c in CH:
    for t in rng.permutation(TSS[c])[:6000]:
        wv = window(c, t + rng.choice([-1, 1]) * rng.choice(dist))
        if wv: prow.append((c, wv[0], wv[1]))
ppool = pd.DataFrame(prow, columns=["chr", "start", "seq"]).drop_duplicates("start")
ppool = ppool[~ppool.seq.isin(set(base.seq) | set(motif_neg.seq))]
ppool["fold"], ppool["gc"], ppool["cpg"] = ppool.chr.map(FOLD), ppool.seq.map(gc), ppool.seq.map(cpg)
ppool = ppool[ppool.fold.notna()]
promoter_neg = match(ppool, pos_tss, ["cpg", "gc"], [0.004, 0.04], n_mult=a.promoter_mult)

# ---------------------------------------------------------------- assemble
new = pd.concat([motif_neg.assign(tier="motif_neg"), promoter_neg.assign(tier="promoter_neg")], ignore_index=True)
new = new.drop_duplicates("seq").assign(label=0, support=0, peak_score=np.nan, src="random")
new["end"] = new.start + W
new["loc"] = new.chr + ":" + new.start.astype(str) + "-" + new.end.astype(str)
new["fold"] = new.fold.astype(int)
new["gc"] = new.seq.map(gc)
new = annotate(new)
out = pd.concat([base, new[base.columns]], ignore_index=True)
cols = ["loc", "chr", "start", "end", "seq", "tier", "label", "fold", "support", "peak_score",
        "gc", "cpg", "pwm", "pwm_offset", "tss_dist", "src"]
out[cols].to_csv(os.path.join(HERE, "dataset_v1_tiers.csv"), index=False)

print("\nrows per tier x fold:")
print(out.groupby(["tier", "fold"]).size().unstack())
summ = out.groupby("tier").agg(n=("seq", "size"), gc=("gc", "mean"), cpg=("cpg", "mean"), pwm=("pwm", "mean"),
                               pwm_ge85=("pwm", lambda v: (v >= 0.85).mean()),
                               tss_1kb=("tss_dist", lambda v: (v <= 1000).mean()),
                               abs_off=("pwm_offset", lambda v: v.abs().median()))
print(summ.round(4))
print("TSS-proximal positives:", len(pos_tss), "| CpG", round(pos_tss.cpg.mean(), 4), "GC", round(pos_tss.gc.mean(), 4))
