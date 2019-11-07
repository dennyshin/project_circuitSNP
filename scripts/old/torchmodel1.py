import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import tensorflow as tf

# function for bringing in regions files. NOT FOR MOTIFS
def import_regions(file_dir):
    regions = {}
    with open(file_dir) as f:
        for line in f:
            row = line.split()
            
            if row[0] not in regions:
                regions[row[0]] = []
            regions[row[0]].append([int(row[1]), int(row[2])])
            
    return regions


# function for creating the regions used for the output layer
def create_output(chromosomes, tissue, openALL):
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
        
        current_motif = { chrm: motif_foot[chrm] for chrm in chromosomes } # filter out unwanted chromosomes
        motif_col = []
        for chrm in chromosomes:
            i = 0
            for region in regions[chrm]:
                if i >= len(current_motif[chrm]):
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
    
    return(training_input)


# bring in LCL and ALL files
openLCL = import_regions("data/CENTIPEDEdata/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak")
openALL = import_regions("data/CENTIPEDEdata/wgEncodeRegDnaseClusteredV3.bed")

# building the output layer: Ytrain
chromosomes = list(openALL.keys())[:-2] #remove sex chromosomes
output = create_output(chromosomes, openLCL, openALL)

Ytrain = []
for chrm in chromosomes:
    for row in output[chrm]:
    	if row[2] == 0:
    		Ytrain.append([1, 0])
    	else:
    		Ytrain.append([0, 1])
Ytrain = np.array(Ytrain)

# create Xtrain
path = "data/CENTIPEDEdata/motif.combo"
Xtrain = np.transpose(import_motifs(path, output))

# make X, Y into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).float()

# build the neural net
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)

# define my device
device = torch.device("cpu")

# define my model
model = Net(5).to(device)

# choose optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# choose my criteria
criterion = nn.BCELoss()

# send data to device
Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)

# training
epochs = 50
for epoch in range(1,epochs+1):
	model.train() # put the model in train mode
	optimizer.zero_grad() # null my gradients otherwise they will accumulate

	Ypred = model(Xtrain) # calculate my Y_hat

	loss = criterion(Ypred, Ytrain) # calculate my loss

	print("epoch: ", epoch, f", loss: {loss.item(): f}")

	loss.backward(loss) # finds grad * loss (remember this is a weighted sum, where weight = loss)
	optimizer.step() # update my parameters

