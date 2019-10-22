import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# load data
X = np.loadtxt('data/LCL_X.out', dtype=int)
Y = np.loadtxt('data/LCL_Y.out', dtype=int)
print(X.shape)
print(Y.shape)

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
		self.fc1 = nn.Linear(input_dim, 5)
		self.fc2 = nn.Linear(5, 5)
		self.fc3 = nn.Linear(5, 2)

		# these are still random for each new model we create
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
test_loss = criterion(Ytest_pred, Ytest)

# plot loss
def plot_loss(training_loss, validation_loss):
	fig, ax = plt.subplots()
	ax.plot(epochs, training_loss, "b", label = "training loss")
	ax.plot(epochs, validation_loss, "g", label = "validation loss")
	ax.set_title("model loss")
	ax.legend()
	fig.savefig('imgs/model6.png')

plot_loss(train_loss, val_loss)

print("test loss: ", test_loss.numpy())

# confusion matrix, precision, recall
def print_confusion_table(table):
	print("\t", " 0", "\t ", "1")
	print("0", "\t", table[0][0], "\t ", table[0][1])
	print("1", "\t", table[1][0], "\t ", table[1][1])
	
test_preds = [np.argmax(pred) for pred in Ytest_pred.numpy()]
test_trues = [np.argmax(true) for true in Ytest.numpy()]

print()
print("building table...")

confusion_table = np.zeros((2,2))
for pred, true in zip(test_preds, test_trues):
	if pred == true == 0:
		confusion_table[0,0] += 1
	elif pred == true == 1:
		confusion_table[1,1] += 1
	elif pred == 0 and true == 1:
		confusion_table[0,1] += 1
	else:
		confusion_table[1,0] += 1

print_confusion_table(confusion_table)

precision = confusion_table[0][0] / sum(confusion_table[0, ])
recall = confusion_table[0][0] / sum(confusion_table[:, 0])

print("precision: ", precision)
print("recall: ", recall)

# save entire model
torch.save(model, "trained_models/model6.pt")