from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from torchvision.datasets import CIFAR10
import torch
import random
from torch.utils.data import Dataset

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='cifar10',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/cifar10/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        train_data = FastCIFAR10(self.rawdata_path, train=True, download=True)
        self.test_data = FastCIFAR10(self.rawdata_path, train=False, download=True)

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
        valid_x = [self.validation[did][0].tolist() for did in range(len(self.validation))]
        valid_y = [self.validation[did][1].tolist() for did in range(len(self.validation))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
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
class FastCIFAR10(CIFAR10):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()

		for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
			self.data[:,i].sub_(mean).div_(std)

		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target