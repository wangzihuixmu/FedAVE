
import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule


# class Model(FModule):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(784, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 10)

#     def forward(self, x):
#         x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x


# for MNIST 32*32
# class Model(FModule):

# 	def __init__(self, device=None):
# 		super(Model, self).__init__()
# 		self.fc1 = nn.Linear(784, 128)
# 		self.fc2 = nn.Linear(128, 64)
# 		self.fc3 = nn.Linear(64, 10)

# 	def forward(self, x):
# 		x = x.view(-1,  784)
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return F.log_softmax(x, dim=1)

# for MNIST 32*32
# class Model(FModule):

# 	def __init__(self, device=None):
# 		super(Model, self).__init__()
# 		self.conv1 = nn.Conv2d(1, 64, 3, 1)
# 		self.conv2 = nn.Conv2d(64, 16, 7, 1)
# 		self.fc1 = nn.Linear(3 * 3 * 16, 200)
# 		self.fc2 = nn.Linear(200, 10)
# 		# self.fc3 = nn.Linear(64, 10)

# 	def forward(self, x):
# 		x = x.view(-1, 1, 28, 28)
# 		x = torch.tanh(self.conv1(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = torch.tanh(self.conv2(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = x.view(-1, 3 * 3 * 16)
# 		x = torch.tanh(self.fc1(x))
# 		x = self.fc2(x)
# 		# x = self.fc3(x)
# 		return F.log_softmax(x, dim=1)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)

class Model(FModule):

	def __init__(self, device=None):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.conv2 = nn.Conv2d(64, 16, 7, 1)
		self.fc1 = nn.Linear(4 * 4 * 16, 200)
		self.fc2 = nn.Linear(200, 10)

	def forward(self, x):
		x = x.view(-1, 1, 32, 32)
		x = torch.tanh(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = torch.tanh(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 16)
		x = torch.tanh(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

