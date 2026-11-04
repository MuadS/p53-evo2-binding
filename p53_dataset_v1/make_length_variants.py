"""Window-length variants of dataset_v1_tiers.csv for the context-length experiment.

Each row keeps its 500 bp centre and tier; only the window around it widens. A row is kept
only if its window is valid at EVERY requested length (in bounds, no N, no blacklist, and for
negatives no overlap with any GSE132246 peak), so all output files contain the identical rows
and the experiment varies context alone. Tier definitions and the annotation columns
(gc, cpg, pwm, pwm_offset, tss_dist) stay as computed on the central 500 bp, so evaluation
strata are identical across lengths. Added per length: gc_flank (GC outside the central 500 bp),
to check whether flanks reintroduce a composition difference between classes.

Output: dataset_v1_tiers_w{L}.csv for 500 and every --lengths value.
Usage:  python make_length_variants.py --fasta hg38.fa --lengths 1000,2000
"""
import argparse, glob, os
import numpy as np, pandas as pd, pyfaidx
from pyliftover import LiftOver

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--lengths", default="1000,2000")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
fa = pyfaidx.Fasta(a.fasta)
LENS = sorted({500, *[int(x) for x in a.lengths.split(",")]})


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

# peaks (no +-1 kb here: the 500 bp negatives already sit >= 1 kb from peaks; a wider negative
# window may approach a peak but must not contain one) and blacklist, hg38
lo = LiftOver(os.path.join(HERE, "ref/hg19ToHg38.over.chain.gz"))
def lift(c, x):
    r = lo.convert_coordinate(c, int(x)); return r[0][1] if r and r[0][0] == c else None
P = pd.concat([pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
               for f in glob.glob(os.path.join(HERE, "raw_peaks/*_peaks.bed.gz"))])
P = P[P.c.isin(CH)]
ls_, le_ = P.apply(lambda r: lift(r.c, r.s), axis=1), P.apply(lambda r: lift(r.c, r.e), axis=1)
ok = ls_.notna() & le_.notna() & (le_ > ls_)
PEAKS = merge(pd.DataFrame({"c": P.c[ok], "s": ls_[ok].astype(int), "e": le_[ok].astype(int)}))
bl = pd.read_csv(os.path.join(HERE, "ref/hg38-blacklist.v2.bed.gz"), sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
BL = merge(bl[bl.c.isin(CH)])

df = pd.read_csv(os.path.join(HERE, "dataset_v1_tiers.csv"))
df["centre"] = df.start + 250
seqs = {L: [] for L in LENS}
keep = np.ones(len(df), bool)
def window(r, L):
    s, e = r.centre - L // 2, r.centre - L // 2 + L
    if s < 0 or e > len(fa[r.chr]) or overlaps(BL, r.chr, s, e) or (r.label == 0 and overlaps(PEAKS, r.chr, s, e)):
        return None
    q = str(fa[r.chr][s:e]).upper()
    return None if "N" in q or (L == 500 and q != r.seq) else q

for i, r in enumerate(df.itertuples()):
    w = {L: window(r, L) for L in LENS}
    if any(q is None for q in w.values()):
        keep[i] = False
        continue
    for L in LENS:
        seqs[L].append(w[L])

print(f"rows kept at all lengths: {keep.sum()} / {len(df)}")
print("dropped by tier:", df[~keep].tier.value_counts().to_dict())
base = df[keep].drop(columns="centre").reset_index(drop=True)
for L in LENS:
    out = base.copy()
    out["seq"], out["start"] = seqs[L], base.start + 250 - L // 2
    out["end"] = out.start + L
    out["gc_flank"] = [((q[:(L - 500) // 2] + q[(L + 500) // 2:]).count("G") + (q[:(L - 500) // 2] + q[(L + 500) // 2:]).count("C"))
                       / max(L - 500, 1) if L > 500 else np.nan for q in out.seq]
    out.to_csv(os.path.join(HERE, f"dataset_v1_tiers_w{L}.csv"), index=False)
    if L > 500:
        print(f"w{L}: flank GC by tier", out.groupby("tier").gc_flank.mean().round(4).to_dict())
print("written:", [f"dataset_v1_tiers_w{L}.csv" for L in LENS])
