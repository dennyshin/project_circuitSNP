import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# load data
instances = {'chr': [], 'region': [], 'SNP_pos': [], 'Vf': [], 'Vref': [], 'Valt': [], 'dsQTL_label': []}
with open('data/dsQTL_instances.txt', 'r') as f:
	heading = f.readline().split()
	for line in f:
		row = line.split("[")
		metainfo = row[0].split()
		
		chrm = metainfo[0]
		region = [int(metainfo[1]), int(metainfo[2])]
		SNP_pos = int(metainfo[3])
		dsQTL_label = int(metainfo[4])

		Vf = row[2].split(", ")
		Vf[-1] = Vf[-1][0]
		Vref = row[3].split(", ")
		Vref[-1] = Vref[-1][0]
		Valt = row[4].split(", ")
		Valt[-1] = Valt[-1][0]

		Vf = [int(x) for x in Vf]
		Vref = [int(x) for x in Vref]
		Valt = [int(x) for x in Valt]

		instances['chr'].append(chrm)
		instances['region'].append(region)
		instances['SNP_pos'].append(SNP_pos)
		instances['Vf'].append(Vf)
		instances['Vref'].append(Vref)
		instances['Valt'].append(Valt)
		instances['dsQTL_label'].append(dsQTL_label)

print("data loaded")

# make data into torch tensors
Vf = torch.from_numpy(np.array(instances['Vf'])).float()
Vref = torch.from_numpy(np.array(instances['Vref'])).float()
Valt = torch.from_numpy(np.array(instances['Valt'])).float()
isdsQTL = torch.from_numpy(np.array(instances['dsQTL_label'])).float()

# define my device
device = torch.device("cpu")

# define model
class Net(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, 2)

		# these are still random for each new model we create
		nn.init.xavier_uniform_(self.fc1.weight) 
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# do not put relu on the last layer!
		return F.softmax(self.fc3(x), dim=1)

# load model
model = torch.load("finished_models/model7D.pt").to(device)

# optimum threshold
with open('results/predict7D.txt', 'r') as f:
	test_loss = f.readline()
	auPRC = f.readline()
	auROC = f.readline()
	heading = f.readline().split()

	dist = 999
	for line in f:
		row = line.split()

		recall = float(row[2])
		if abs(recall - 0.1) < dist:
			dist = abs(recall - 0.1)
			recall10 = recall
			precision_at_10recall = float(row[1])
			threshold = float(row[0])

print("model loaded")

# send data to device
Vf, Vref, Valt, isdsQTL = Vf.to(device), Vref.to(device), Valt.to(device), isdsQTL.to(device)

# test
model.eval()
with torch.no_grad():
	ref_pred = model(Vref)
	alt_pred = model(Valt)

print("starting analysis...")

# analysis
ref_pred, alt_pred = ref_pred.numpy(), alt_pred.numpy()
isdsQTL = isdsQTL.numpy()

# is the prediction from the reference instance to the alternative instance different?
ref_predictions = np.array([int(x >= threshold) for x in ref_pred[:,1]], dtype=int)
alt_predictions = np.array([int(x >= threshold) for x in alt_pred[:,1]], dtype=int)

isdsQTL_predictions = []
for rpred, apred in zip(ref_predictions, alt_predictions):
	if rpred != apred:
		isdsQTL_predictions.append(1)
	else:
		isdsQTL_predictions.append(0)

with open('results/isdsQTL.txt', 'w') as f:
	f.writelines("%s\n" % list(ref_predictions))
	f.writelines("%s\n" % list(alt_predictions))
	f.writelines("%s\n" % isdsQTL_predictions)