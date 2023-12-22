from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
import torch

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='emnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/emnist/data',
                                      )
        self.num_classes = 25
        self.save_data = self.XYData_to_json

    # leeters: 总共145600张，26类，每一类包含相同数据，每一类训练集4800张，测试集800张
    def load_data(self):
        # train_data = datasets.EMNIST(self.rawdata_path, split='letters', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]))
        # self.test_data = datasets.EMNIST(self.rawdata_path, split='letters', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]))

        train_data = FastEMNIST(self.rawdata_path, split='letters', train=True, download=True)
        self.test_data = FastEMNIST(self.rawdata_path, split='letters', train=False, download=True)

        train_indices, valid_indices = self.get_train_valid_indices(len(train_data), 0.9)

        self.train_data = Custom_Dataset(train_data.data[train_indices], train_data.targets[train_indices])
        self.validation = Custom_Dataset(train_data.data[valid_indices],train_data.targets[valid_indices])
        self.test_data = Custom_Dataset(self.test_data.data, self.test_data.targets)
        a = 0

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



class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

# 可删除
class FastEMNIST(EMNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		# self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
		self.data = self.data.sub_(0.1307).div_(0.3081)
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