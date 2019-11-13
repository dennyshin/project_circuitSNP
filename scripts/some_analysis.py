import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

model_info = {}
for file in os.listdir('results/'):
	if file != 'isdsQTL.txt':
		with open(os.path.join('results', file)) as f:
			model = file[8]
			test_loss = float(f.readline().split()[1])
			auPRC = float(f.readline().split()[1])
			auROC = float(f.readline().split()[1])

			model_info[model] = {'test_loss': test_loss, 'auPRC': auPRC, 'auROC': auROC}

for k, v in model_info.items():
	print(k, v)

# plot models
def plot_models(X, Y, title, imgpath):
	fig, ax = plt.subplots()
	ax.plot(X, Y, "b.")
	ax.set_title(title)
	ax.set_xlabel('models')
	ax.set_ylabel('auPRC')
	# ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	# ax.legend()
	fig.savefig(imgpath)

plot_models(list(model_info.keys()), [x['auPRC'] for x in model_info.values()], "auPRC of models", 'imgs/auPRC_of_models.png')
plot_models(list(model_info.keys()), [x['auROC'] for x in model_info.values()], "auROC of models", 'imgs/auROC_of_models.png')

# define my device
device = torch.device("cpu")

# define model
class Net(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# do not put relu on the last layer!
		return F.softmax(self.fc3(x), dim=1)

# load model
model = torch.load("finished_models/model7D.pt").to(device)

# mock instances
test_instances = torch.from_numpy(np.array([[0]*1372, [1]*1372, [0,1]*int(1372/2)])).float()
print(test_instances)

# send to device
test_instances = test_instances.to(device)

# test
model.eval()
with torch.no_grad():
	test_predictions = model(test_instances)

print(test_predictions)