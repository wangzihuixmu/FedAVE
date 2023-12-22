from builtins import next, super
from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F

# class Model(FModule):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 64, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1600, 384),
#             nn.ReLU(),
#             nn.Linear(384, 192),
#             nn.ReLU(),
#             nn.Linear(192, 10),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.flatten(1)
#         return self.decoder(x)

#rffl
class Model(FModule):
	def __init__(self, in_channels=3, n_kernels=16, out_dim=10, device=None):
		super(Model, self).__init__()

		self.ingraph = False
		self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
		self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, out_dim)
	
	def get_device(self):
		return next(self.parameters()).device

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)