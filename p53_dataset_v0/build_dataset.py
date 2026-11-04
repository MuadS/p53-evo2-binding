import pandas as pd, numpy as np
rng=np.random.default_rng(42)
TEST={"chr8","chr9"}; VAL={"chr7"}
def load(f,label):
    d=pd.read_csv(f,sep="\t",header=None,names=["loc","seq"])
    d["seq"]=d.seq.str.upper(); d=d[~d.seq.str.contains("N")].copy()
    d["chr"]=d["loc"].str.split(":").str[0]
    d["gc"]=d.seq.apply(lambda s:(s.count("G")+s.count("C"))/len(s))
    d["gcb"]=(d.gc/0.02).round().astype(int); d["label"]=label
    return d.reset_index(drop=True)
pos=load("pos.tsv",1); neg=load("neg.tsv",0)
pool={k:list(rng.permutation(v.index)) for k,v in neg.groupby("gcb")}
pick=[]
for gb in pos.gcb:
    for d in (0,1,-1,2,-2,3,-3):
        l=pool.get(gb+d)
        if l: pick.append(l.pop()); break
df=pd.concat([pos,neg.loc[pick]]).drop_duplicates("seq").reset_index(drop=True)
df["split"]=df.chr.map(lambda c:"test" if c in TEST else "val" if c in VAL else "train")
df[["loc","chr","seq","gc","label","split"]].to_csv("dataset_v0.csv",index=False)
print(df.groupby(["split","label"]).size().unstack())
print("mean GC pos/neg:", df.groupby("label").gc.mean().round(3).to_dict())
