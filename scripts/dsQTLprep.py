import numpy as np
import os

# load data
# print("loading X")
# X = np.loadtxt('data/LCL_X.out', dtype=int)

print('loading regions...')
regions = {}
with open('data/regions.txt') as f:
	for line in f:
		row = line.split()
		chrm = row[0]
		region = [int(row[1]), int(row[2])]
		if chrm not in regions.keys():
			regions[chrm] = []
		regions[chrm].append(region)

print('loading motif names')
motifnames = []
with open('data/motif_names.txt') as f:
	for line in f:
		row = line.split()
		motifnames.append(row[0])

print('loading X')
i = 0
CENTIPEDEinstances = {}
with open('data/LCL_X.out') as f:
	for chrm in regions.keys():
		# for region in regions[chrm][0:10]:
		for region in regions[chrm]:
			instance_index = str(chrm + "-" + str(region[0]) + "-" + str(region[1]))
			instance = [int(x) for x in f.readline().split()]
			CENTIPEDEinstances[instance_index] = instance
			print(i)
			i += 1

print(len(CENTIPEDEinstances.values()))
print()
print(len(CENTIPEDEinstances[instance_index]))
print(len(motifnames))

print("data loaded")


# X = np.array([[1,2], [3,4], [5,6]], dtype=int)
# indexes = []
# for chrm in ['chr1']:
# 	for region in regions[chrm][0:3]:
# 		instance_index = str(chrm + "-" + str(region[0]) + "-" + str(region[1]))
# 		indexes.append(instance_index)

# for k,v in zip(indexes, X):
# 	print(k,":",v)



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

# num_dsQTL = sum([int(x['label']) == 1 for x in loci])
# print(num_dsQTL)

# for every dictionary in the list loci
# find the region instance
# in X for that region, there should be motifs labelled 1 if they are present
# create a new instance of all 0
# turn on and off motifs that overlap with the Denger dsQTL SNP

dsQTL_instances = []
for locus in loci[0:10]: # change later
	chrm = locus['chr']
	SNPpos = int(locus['pos0'])
	isdsQTL = int(locus['label'])

	dsQTLinstance = {}
	for region in regions[chrm]:
		if region[0] <= SNPpos <= region[1]:
			dsQTLinstance['chr'] = chrm
			dsQTLinstance['region'] = [region[0], region[1]]
			dsQTLinstance['SNP_pos'] = SNPpos
			
			index = str(chrm + "-" + str(region[0]) + "-" + str(region[1]))
			dsQTLinstance['instance'] = CENTIPEDEinstances[index]

			dsQTLinstance['isdsQTL'] = isdsQTL
			
			dsQTL_instances.append(dsQTLinstance)
			
			break

# print(dsQTL_instances)