import numpy as np
import os

load data
X = np.loadtxt('data/LCL_X.out', dtype=int)
Y = np.loadtxt('data/LCL_Y.out', dtype=int)

regions = {}
with open('data/regions.txt') as f:
	for line in f:
		row = line.split()
		chrm = row[0]
		region = [row[1], row[2]]
		if chrm not in regions.keys():
			regions[chrm] = []
		regions[chrm].append(region)

motifnames = []
with open('data/motif_names.txt') as f:
	for line in f:
		row = line.split()
		motifnames.append(row[0])

print('loaded data')

file_dir = "rawdata/dsQTL.eval.txt"
loci = [] # a list of dictionaries
with open(file_dir) as f:
	heading = f.readline().split()
	for line in f:
		row = line.split()
		SNP = {}
		for k, v in zip(heading, row):
			SNP[k] = v
		loci.append(SNP)

num_dsQTL = sum([int(x['label']) == 1 for x in loci])
print(num_dsQTL)

# for every dictionary in the list loci
# find the region instance
# in X for that region, there should be motifs labelled 1 if they are present
# create a new instance of all 0
# turn on and off motifs that overlap with the Denger dsQTL SNP