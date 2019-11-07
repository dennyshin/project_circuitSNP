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
Xtrain = np.loadtxt('data/LCL_Xtrain.out', dtype=int)
Ytrain = np.loadtxt('data/LCL_Ytrain.out', dtype=int)
Xval = np.loadtxt('data/LCL_Xval.out', dtype=int)
Yval = np.loadtxt('data/LCL_Yval.out', dtype=int)

# make data into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).float()
Xval = torch.from_numpy(Xval).float()
Yval = torch.from_numpy(Yval).float()

# define my device
device = torch.device("cpu")

# build the neural net
class Net(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 2)

		# these are still random for each new model we create
		nn.init.xavier_uniform_(self.fc1.weight) 
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# do not put relu on the last layer!
		return F.softmax(self.fc3(x), dim=1)

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

# training and validation
n_epochs = 5000
epochs = list(range(1, n_epochs+1))
train_loss = []
val_loss = []
min_val_loss = 1
for epoch in epochs:
	model.train() # put the model in train mode
	optimizer.zero_grad() # null my gradients otherwise they will accumulate

	Ytrain_pred = model(Xtrain) # calculate my Y_hat

	loss = criterion(Ytrain_pred, Ytrain) # calculate my loss
	train_loss.append(loss)
	
	loss.backward(loss) # finds grad * loss (remember this is a weighted sum, where weight = loss)
	optimizer.step() # update my parameters

	model.eval()
	with torch.no_grad():
		Yval_pred = model(Xval)
	loss = criterion(Yval_pred, Yval)
	val_loss.append(loss)

	print("epoch: ", epoch, f", val_loss: {loss.item(): f}")

	# save model with lowest validation loss
	if loss < min_val_loss:
		min_val_loss = loss
		torch.save(model, "trained_models/model7F.pt")
		opt_epoch = epoch
	elif loss >= min_val_loss + 0.1:
		break

# plot loss
def plot_loss(training_loss, validation_loss):
	fig, ax = plt.subplots()
	ax.plot(epochs, training_loss, "b", label = "training loss")
	ax.plot(epochs, validation_loss, "g", label = "validation loss")
	ax.set_title("model loss")
	ax.legend()
	fig.savefig('imgs/model7F.png')

plot_loss(train_loss, val_loss)

# save loss
with open('trained_models/model7F_opt.txt', 'w') as f:
	f.write("min val loss = %f\t at epoch %i" % (min_val_loss, opt_epoch))
np.savetxt('trained_models/model7F_trainloss.txt', train_loss, fmt='%f')
np.savetxt('trained_models/model7F_valloss.txt', val_loss, fmt='%f')
