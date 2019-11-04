import numpy as np
import os

file_dir = "rawdata/dsQTL.eval.txt"
loci = []
with open(file_dir) as f:
	heading = f.readline().split()
	for line in f:
		row = line.split()
		SNP = {}
		for k, v in zip(heading, row):
			SNP[k] = v
		loci.append(SNP)

print(loci[0]['label'])
print(loci[0]['gkm_SVM'])
print(len(loci))

num_dsQTL = sum([int(x['label']) == 1 for x in loci])
print(num_dsQTL)

chromosomes = sorted(list(set([x['chr'] for x in loci])), key=lambda x: int(x.split('chr')[1]))
print(chromosomes)
