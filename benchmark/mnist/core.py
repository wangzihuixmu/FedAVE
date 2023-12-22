from builtins import super
from tkinter import SEL
import torch
from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='mnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/mnist/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        # train_data = datasets.MNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        # self.test_data = datasets.MNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

        # train_indices, valid_indices = self.get_train_valid_indices(len(train_data), 0.9)
        # self.train_data = Custom_Dataset(train_data.data[train_indices], train_data.targets[train_indices], transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        # self.validation = Custom_Dataset(train_data.data[valid_indices], train_data.targets[valid_indices], transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


        train_data = FastMNIST('.data', train=True, download=True)
        self.test_data = FastMNIST('.data', train=False, download=True)

        train_indices, valid_indices = self.get_train_valid_indices(len(train_data), 0.9)

        self.train_data = Custom_Dataset(train_data.data[train_indices], train_data.targets[train_indices])
        self.validation = Custom_Dataset(train_data.data[valid_indices],train_data.targets[valid_indices])
        self.test_data = Custom_Dataset(self.test_data.data, self.test_data.targets)

    def get_train_valid_indices(self, n_samples, train_val_split_ratio):
        indices = list(range(n_samples))
        random.seed(11253)
        random.shuffle(indices)
        split_point = int(n_samples * train_val_split_ratio)
        train_indices, valid_indices = indices[:split_point], indices[split_point:]
        return train_indices, valid_indices

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1].tolist() for did in range(len(self.train_data))]
        # train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        valid_x = [self.validation[did][0].tolist() for did in range(len(self.validation))]
        valid_y = [self.validation[did][1].tolist() for did in range(len(self.validation))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        # test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1].tolist() for did in range(len(self.test_data))]    
        self.train_data = {'x':train_x, 'y':train_y}
        self.validation = {'x':valid_x, 'y':valid_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

class Custom_Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.data = X
        self.targets = y
        self.count = len(X)
        self.transform = transform
    
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        a = self.transform(self.data[idx])
        return a, self.targets[idx]

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]
        return self.data[idx], self.targets[idx]


# 可删除
class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
		# self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target
