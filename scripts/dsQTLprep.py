import numpy as np
import os

# load data
print('loading regions...')
regions = {}
with open('data/regions.txt', 'r') as f:
	for line in f:
		row = line.split()
		chrm = row[0]
		region = [int(row[1]), int(row[2])]
		if chrm not in regions.keys():
			regions[chrm] = []
		regions[chrm].append(region)

print('loading motif names')
motifnames = {}
with open('data/motif_names.txt', 'r') as f:
	idx = 0
	for line in f:
		row = line.split()
		motifnames[idx] = row[0]
		idx += 1

print('loading X')
CENTIPEDEinstances = {}
with open('data/LCL_X.out') as f:
	for chrm in regions.keys():
		# for region in regions[chrm][0:10]:
		for region in regions[chrm]:
			instance_index = str(chrm + "-" + str(region[0]) + "-" + str(region[1]))
			instance = [int(x) for x in f.readline().split()]
			CENTIPEDEinstances[instance_index] = instance

print()
print(sum([len(x) for x in regions.values()]))
print(len(CENTIPEDEinstances.values()))
print()
print(len(CENTIPEDEinstances[instance_index]))
print(len(motifnames))

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

print("data loaded")

print("making new instances")

dsQTL_instances = []
for locus in loci:
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

for i in range(0, len(dsQTL_instances)):
	region_motifs = [(motifidx, motifnames[motifidx]) for motifidx in np.nonzero(dsQTL_instances[i]['instance'])[0]]

	# look up the region motifs in the CENTIPEDE data file
	# see which motifs overlap with the SNP
	path = "rawdata/CENTIPEDEdata/motif.combo"
	motif_ovlp_SNP = []
	for motifidx, motiffile in region_motifs:
		with open(os.path.join(path, motiffile)) as f:
			for line in f:
				motifinfo = line.split()
	
				if chrm == motifinfo[0]:
					motifstart = int(motifinfo[1])
					motifend = int(motifinfo[2])
	
					if motifstart <= SNPpos <= motifend:
						motif_ovlp_SNP.append((motifidx, motiffile, motifstart, motifend))

	# Vf: 1=motif footprint in region overlaps with SNP, 0 otherwise
	# Vf is created using motifs overlapping with SNP
	Vf = [0]*len(dsQTL_instances[i]['instance'])
	for motifidx, motiffile, motifstart, motifend in motif_ovlp_SNP:
		Vf[motifidx] = 1
	dsQTL_instances[i]['Vf'] = Vf

	# Vref: 1=ref allele increases binding or no effect, 0=alt allele increases binding
	# Vref is created by copying Vf and then applying the effect to the correct index
	# do similar with Valt
	Vref = Vf
	Valt = Vf
	# among those motifs check centiSNP files to see what affect the alleles have
	path = "rawdata/centiSNPdata/motif"
	for motifidx, motiffile, motifstart, motifend in motif_ovlp_SNP:
		motiffile = motiffile.replace(".combo", "")
		with open(os.path.join(path, motiffile)) as f:
			for line in f:
				centiSNPinfo = line.split()
	
				if chrm == centiSNPinfo[0]:
					centiSNP_pos = int(centiSNPinfo[1])
					if SNPpos == centiSNP_pos:
						effect = int(centiSNPinfo[10])
						# is a centiSNP
						if effect == 2:
							ref_priorlodds = float(centiSNPinfo[6])
							alt_priorlodds = float(centiSNPinfo[7])
							if ref_priorlodds - alt_priorlodds > 0:
								Vref[motifidx] = 1
								Valt[motifidx] = 0
							elif ref_priorlodds - alt_priorlodds < 0 :
								Vref[motifidx] = 0
								Valt[motifidx] = 1
	dsQTL_instances[i]['Vref'] = Vref
	dsQTL_instances[i]['Valt'] = Valt

# save
with open('data/dsQTL_instances.txt', 'w') as f:
	f.write("chr\tregion_start\tregion_end\tSNP_position\tdsQTL_label\tCENTIPEDEinstance\tVf\tVref\tValt\n")
	for inst in dsQTL_instances:
		chrm = inst['chr']
		region_start = inst['region'][0]
		region_end = inst['region'][1]
		SNPpos = inst['SNP_pos']
		instance = inst['instance']
		isdsQTL = inst['isdsQTL']
		Vf = inst['Vf']
		Vref = inst['Vref']
		Valt = inst['Valt']
		
		f.write("%s\t%i\t%i\t%i\t%i\t" % (chrm, region_start, region_end, SNPpos, isdsQTL))
		f.writelines("%s\t" % instance)
		f.writelines("%s\t" % Vf)
		f.writelines("%s\t" % Vref)
		f.writelines("%s\n" % Valt)