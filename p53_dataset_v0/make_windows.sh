set -e
python3 - <<'EOF'
import pandas as pd
W=500
CH=[f"chr{i}" for i in range(1,23)]+["chrX"]
sizes=dict(l.split()[:2] for l in open("hg19.chrom.sizes"))
p=pd.read_csv("peaks.bed",sep="\t",header=None,comment="#",usecols=[0,1,2])
p=p[p[0].isin(CH)]
c=(p[1]+p[2])//2
p["s"]=c-W//2; p["e"]=p.s+W
p=p[(p.s>=0)&(p.e<=p[0].map(lambda x:int(sizes[x])))]
p[[0,"s","e"]].drop_duplicates().to_csv("pos.bed",sep="\t",header=False,index=False)
EOF
bedtools slop -i pos.bed -g hg19.chrom.sizes -b 1000 | cat - hg19-blacklist.v2.bed | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge > excl.bed
for i in $(seq 10); do cat pos.bed; done > pos_x10.bed
bedtools shuffle -i pos_x10.bed -g hg19.chrom.sizes -excl excl.bed -noOverlapping -seed 42 > neg_raw.bed
bedtools getfasta -fi hg19.fa -bed pos.bed -tab -fo pos.tsv
bedtools getfasta -fi hg19.fa -bed neg_raw.bed -tab -fo neg.tsv
