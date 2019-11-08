import numpy as np
import os

# load data
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

print()
print(sum([len(x) for x in regions.values()]))
print(len(CENTIPEDEinstances.values()))
print()
print(len(CENTIPEDEinstances[instance_index]))
print(len(motifnames))

print("data loaded")

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

dsQTL_instances = []
for locus in loci: # change later
	chrm = locus['chr']
	SNPpos = int(locus['pos0'])
	isdsQTL = int(locus['label']) # 1 = dsQTL, -1 = not dsQTL

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

with open('data/dsQTLinstances.txt', 'w') as f:
	f.write("chr\tregion_start\tregion_end\tSNP_position\tdsQTL_label\tinstance\n")
	for inst in dsQTL_instances:
		chrm = inst['chr']
		region_start = inst['region'][0]
		region_end = inst['region'][1]
		SNPpos = inst['SNP_pos']
		instance = inst['instance']
		isdsQTL = inst['isdsQTL']
		
		f.write("%s\t%i\t%i\t%i\t%i\t" % (chrm, region_start, region_end, SNPpos, isdsQTL))
		f.writelines("%s\n" % instance)