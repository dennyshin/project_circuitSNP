import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# function for bringing in regions files. NOT FOR MOTIFS
def import_tissue_regions(file_dir):
	regions = {}
	with open(file_dir) as f:
		for line in f:
			row = line.split()
			
			if row[0] not in regions:
				regions[row[0]] = []
			regions[row[0]].append([int(row[1]), int(row[2])])
			
	return regions


# function for creating the regions used for the output layer
def create_regions(chromosomes, tissue, openALL):
	output = dict.fromkeys(chromosomes)
	
	for chrm in chromosomes:
		
		i=0
		regionlist = []
		for region in tissue[chrm]:
			open_region = [region[0], region[1], 1]

			try:
				# | ALL1 | ALL2 | LCL1 |   while no overlap
				while openALL[chrm][i][1] < open_region[0]: # ALL_end is smaller than LCL_start
					# add in closed region
					# [start, end, closed/open]
					closed_region = [openALL[chrm][i][0], openALL[chrm][i][1], 0]
					regionlist.append(closed_region)
					i += 1
			except:
				# there are some LCL_region left
				# | LCL1 ALL1 ALL2 | LCL2 | LCL3 | end
				regionlist.append(open_region)
			
			# now, we must be at an overlap or past it
		
			try:
				# | ALL1 | LCL1 | ALL2 |   ALL_region is past LCL. no overlap
				if openALL[chrm][i][0] > open_region[1]: # ALL_start is bigger than LCL_end
					# insert open region
					regionlist.append(open_region)
		
				# | ALL1 LCL1 | ALL2    overlap exists
				else:
					# insert open region
					regionlist.append(open_region)
			
					try:
						# | ALL1 LCL1 ALL2 ALL3 | ALL4 |   skip until overlap ends
						while openALL[chrm][i][0] <= open_region[1]:
							i += 1
					except:
						# this means ALL has run out during an overlap
						# | LCL1 ALL1 ALL2 | end
						pass
			except:
				pass
				
		# tail end
		# there may still be some ALL_region remaining
		# | LCL1 | ALL1 | ALL2 | ALL3 | end
		for region in openALL[chrm][i:]:
			closed_region = [region[0], region[1], 0]
			regionlist.append(closed_region)
		
		output[chrm] = regionlist
		
	return output


# function for importing in motif files
# creates training input at the same time
# training input is a matrix containing the instances and features
def import_motifs(path, regions):
	training_input = []
	for motiffile in os.listdir(path):
		motif_foot = {}
		with open(os.path.join(path, motiffile)) as f:
			for line in f:
				row = line.split()
				
				if row[0] not in motif_foot:
					motif_foot[row[0]] = []
				motif_foot[row[0]].append([int(row[1]), int(row[2])])
	   
		current_motif = { chrm: motif_foot[chrm] for chrm in (chromosomes & motif_foot.keys()) } # filter out unwanted chromosomes
		motif_col = []
		for chrm in chromosomes:
			i = 0
			for region in regions[chrm]:
				if chrm not in current_motif.keys():
					motif_col.append(0)
				elif i >= len(current_motif[chrm]):
					motif_col.append(0)
				else:
					motif_reg = current_motif[chrm][i]
					if region[1] < motif_reg[0]: # no overlap
						motif_col.append(0)
						# move to next region
					else:
						if region[0] > motif_reg[1]: # no overlap
							motif_col.append(0)
							i += 1 # move motif[i] forward
						else: # must be overlap
							motif_col.append(1)
							# don't move forward as next region might also overlap
					
			# if motif[i] is past the last region in regionlist then that's okay
		
		training_input.append(motif_col)
	
	return training_input

print("starting...")

# bring in LCL and ALL files
openLCL = import_tissue_regions("data/CENTIPEDEdata/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak")
openALL = import_tissue_regions("data/CENTIPEDEdata/wgEncodeRegDnaseClusteredV3.bed")

# build the regions
chromosomes = list(openALL.keys())[:-2] #remove sex chromosomes
regions = create_regions(chromosomes, openLCL, openALL)

print("made regions")

# edit the regions to be 300bps long
for chrm in chromosomes:
	for region in regions[chrm]:
		centroid = (region[0] + region[1]) // 2 # floor division
		region[0] = centroid - 149
		region[1] = centroid + 150

# building the labels
Y = []
for chrm in chromosomes:
	for row in regions[chrm]:
		if row[2] == 0:
			Y.append([1, 0])
		else:
			Y.append([0, 1])
Y = np.array(Y)

print("made Y")

# build instances
path = "data/CENTIPEDEdata/motif.combo" # all motif files
X = np.transpose(import_motifs(path, regions))

print("made X")

# split data
# 70% training, 20% validation, 10% test
train_len = int(np.shape(Y)[0]*0.7) # rounded down to nearest integer
val_len = int(np.shape(Y)[0]*0.9)

DATA = np.concatenate((X,Y), axis=1)
np.random.shuffle(DATA) # row shuffle

train_data = DATA[ :train_len]
val_data = DATA[train_len:val_len]
test_data = DATA[val_len: ]

Xtrain = np.hsplit(train_data, [-2])[0]
Ytrain = np.hsplit(train_data, [-2])[1]
Xval = np.hsplit(val_data, [-2])[0]
Yval = np.hsplit(val_data, [-2])[1]
Xtest = np.hsplit(test_data, [-2])[0]
Ytest = np.hsplit(test_data, [-2])[1]

# make data into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).float()
Xval = torch.from_numpy(Xval).float()
Yval = torch.from_numpy(Yval).float()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).float()

# build the neural net
class Net(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, 2)

		# these are still random for each new model we create
		# emprically it lowers the starting loss
		nn.init.xavier_uniform_(self.fc1.weight) 
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# do not put relu on the last layer!
		return F.softmax(self.fc3(x), dim=1)

# define my device
device = torch.device("cpu")

# define my model
motif_num = Xtrain.shape[1] # this is now an int
model = Net(motif_num).to(device)

# choose optimizer
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# choose my criteria
criterion = nn.BCELoss()

# send data to device
Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
Xval, Yval = Xval.to(device), Yval.to(device)
Xtest, Ytest = Xtest.to(device), Ytest.to(device)

# training and validation
n_epochs = 10
epochs = list(range(1, n_epochs+1))
train_loss = []
val_loss = []
for epoch in epochs:
	model.train() # put the model in train mode
	optimizer.zero_grad() # null my gradients otherwise they will accumulate

	Ytrain_pred = model(Xtrain) # calculate my Y_hat

	loss = criterion(Ytrain_pred, Ytrain) # calculate my loss
	train_loss.append(loss)
	
	print("epoch: ", epoch, f", loss: {loss.item(): f}")

	loss.backward(loss) # finds grad * loss (remember this is a weighted sum, where weight = loss)
	optimizer.step() # update my parameters

	model.eval()
	with torch.no_grad():
		Yval_pred = model(Xval)
	loss = criterion(Yval_pred, Yval)
	val_loss.append(loss)

# test
model.eval()
with torch.no_grad():
	Ytest_pred = model(Xtest)
test_loss = criterion(Ytest_pred, Ytest).numpy()

# plot loss
def plot_loss(training_loss, validation_loss):
	fig, ax = plt.subplots()
	ax.plot(epochs, training_loss, "b", label = "training loss")
	ax.plot(epochs, validation_loss, "g", label = "validation loss")
	ax.set_title("model loss")
	ax.legend()
	fig.savefig('model4.png')

plot_loss(train_loss, val_loss)

print("test loss: ", test_loss)