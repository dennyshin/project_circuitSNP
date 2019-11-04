import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# load data
Xtest = np.loadtxt('data/LCL_Xtest.out', dtype=int)
Ytest = np.loadtxt('data/LCL_Ytest.out', dtype=int)

# make data into torch tensors
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).float()

# define my device
device = torch.device("cpu")

# define model
class Net(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, 32)
		self.fc4 = nn.Linear(32, 32)
		self.fc5 = nn.Linear(32, 32)
		self.fc6 = nn.Linear(32, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		# do not put relu on the last layer!
		return F.softmax(self.fc6(x), dim=1)

# load model
model = torch.load("trained_models/model7I.pt").to(device)

# choose my criteria
criterion = nn.BCELoss()

# send data to device
Xtest, Ytest = Xtest.to(device), Ytest.to(device)

# test
model.eval()
with torch.no_grad():
	Ytest_pred = model(Xtest)
test_loss = criterion(Ytest_pred, Ytest)

print("test loss: ", test_loss.numpy())

# analysis
Ytest_pred, Ytest = Ytest_pred.numpy(), Ytest.numpy()
true_labels = np.array(Ytest[:,1], dtype=int)

N = true_labels.shape[0] # total number of test instances
n = sum(true_labels) # total number of true label=1

precision = []
recall = []
specificity = []
FPR = [] # 1-specificity = False Positive Rate
thresholds = np.sort(np.unique(np.around(Ytest_pred[:,1], 2)))
for threshold in thresholds:
	# x is the prediction value for 1 (open region)
	# if x <= threshold, then predict 1
	predictions = np.array([int(x >= threshold) for x in Ytest_pred[:,1]], dtype=int)
	pred_true = np.array(list(zip(predictions, true_labels)))

	true_pos = sum([x[0] == x[1] == 1 for x in pred_true])
	positives = sum([x[0] == 1 for x in pred_true]) # number of times model guess label=1
	true_neg = sum([x[0] == x[1] == 0 for x in pred_true])

	PPV = round(true_pos / positives, 6) # precision
	TPR = round(true_pos / n, 6) # recall
	TNR = round(true_neg / (N-n), 6) # specificity

	precision.append(PPV)
	recall.append(TPR)
	specificity.append(TNR)
	FPR.append(round(1-TNR, 6)) # 1-specificity

# calculate area under curves
auPRC = round(np.trapz(precision, x=recall), 6)
auROC = round(np.trapz(recall, x=FPR), 6)

# plot precision-recall
def plot_PRcurve(recall, precision, imgpath):
	fig, ax = plt.subplots()
	ax.plot(recall, precision, "b.-")
	ax.set_title("PR curve, area = %1.6f" %auPRC)
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	# ax.legend()
	fig.savefig(imgpath)

# plot ROC curve
def plot_ROCcurve(FPR, sensitivity, imgpath):
	fig, ax = plt.subplots()
	ax.plot(FPR, sensitivity, "b.-")
	ax.set_title("ROC curve, area = %1.6f" %auROC)
	ax.set_xlabel('FPR')
	ax.set_ylabel('Sensitivity')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	# ax.legend()
	fig.savefig(imgpath)

plot_PRcurve(recall, precision, 'imgs/model7I_PRcurve.png')
plot_ROCcurve(FPR, recall, 'imgs/model7I_ROCcurve.png')
