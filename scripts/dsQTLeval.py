import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# load data
instances = {'region': [], 'SNP_pos': [], 'Vf': [], 'Vref': [], 'Valt': [], 'dsQTL_label': []}
print("data loaded")

# make data into torch tensors

# define my device
device = torch.device("cpu")

# define model

# load model
# and optimum threshold
model = torch.load("trained_models/.pt").to(device)

print("model loaded")

# choose my criteria
criterion = nn.BCELoss()

# send data to device

# test
model.eval()
with torch.no_grad():
	Ytest_pred = model(Xtest)
test_loss = criterion(Ytest_pred, Ytest)

print("test loss: ", test_loss.numpy())

# analysis
print("starting analysis...")
