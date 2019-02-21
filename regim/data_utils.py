import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

class DataUtils:
    @staticmethod
    def ensure_tensor(a):
        if type(a) is torch.Tensor:
            return a
        elif type(a) is np.ndarray:
            return torch.from_numpy(a)
        else: # handle all other types convertible by numpy
            return torch.from_numpy(np.array(a))

    @staticmethod
    def channel_norm(ds, channel_dim=None):
        # collect tensors in list
        l = [DataUtils.ensure_tensor(data) for data, *_ in ds]
        # join back all tensors so the first dimension is count of tensors
        l = torch.stack(l, dim=0) #size: [N, X, Y, ...] or [N, C, X, Y, ...]

        if channel_dim is None:
            # add redundant first dim
            l = l.unsqueeze(0)

        else:
            # swap channel dimension to first
            l = torch.transpose(l, 0, channel_dim).contiguous() #size: [C, N, X, Y, ...]
        # collapse all except first dimension
        l = l.view(l.size(0), -1) #size: [C, N*X*Y]
        mean = torch.mean(l, dim=1) #size: [C]
        std = torch.std(l, dim=1) #size: [C]
        return (mean, std)

    @staticmethod
    def sample_by_class(ds, k):
        class_counts = {}
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for data, label in ds:
            c = label.item()
            class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts[c] <= k:
                train_data.append(torch.unsqueeze(data, 0))
                train_label.append(torch.unsqueeze(label, 0))
            else:
                test_data.append(torch.unsqueeze(data, 0))
                test_label.append(torch.unsqueeze(label, 0))
        train_data = torch.cat(train_data)
        train_label = torch.cat(train_label)
        test_data = torch.cat(test_data)
        test_label = torch.cat(test_label)

        return (TensorDataset(train_data, train_label), 
            TensorDataset(test_data, test_label))

    @staticmethod
    def mnist_datasets(linearize=False):
        mnist_transforms=[
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        if linearize:
            mnist_transforms.append(ReshapeTransform((-1,)))
        train_ds = datasets.MNIST('../data', train=True, download=True, 
            transform=transforms.Compose(mnist_transforms));
        test_ds = datasets.MNIST('../data', train=False, 
            transform=transforms.Compose(mnist_transforms));
        return train_ds, test_ds

    @staticmethod
    def sample_traintest_by_class(train_ds, test_ds, k=None):  
        if k is not None:
            train_ds_part, test_ds_part = DataTools.sample_by_class(train_ds, k)
            test_ds_part = test_ds
        else:
            train_ds_part, test_ds_part = train_ds, test_ds

        return train_ds_part, test_ds_part

    @staticmethod
    def split_dataset(ds, train_frac=0.6, test_frac=None, validate_frac=0.0):
        if test_frac is None and validate_frac is None:
            raise ValueError("both test_frac and validation_frac should not be None")

        total_len = len(ds)
        train_len = int(total_len * train_frac)

        test_frac = test_frac or (1 - train_frac - validate_frac)
        test_len = int(total_len * test_frac)

        validate_frac = validate_frac or (1 - train_frac - test_frac)
        validate_len = int(total_len * validate_frac)

        train_len = total_len-validate_len-test_len
        
        if validate_len > 0:
            train_ds, test_ds, validate_ds = torch.utils.data.random_split(ds, 
                (train_len, test_len, validate_len))
            return (train_ds, test_ds, validate_ds)
        else:
            train_ds, test_ds = torch.utils.data.random_split(ds, 
                (train_len, test_len))
            return (train_ds, test_ds)

    @staticmethod
    def get_dataloaders(config, train_ds, test_ds):
        kwargs_train = {'pin_memory': True} if config.train_config.use_cuda else {}
        kwargs_test = {'pin_memory': True} if config.test_config.use_cuda else {}

        train_loader = torch.utils.data.DataLoader(train_ds,
            batch_size=config.train_config.batch_size, shuffle=True, **kwargs_train) \
                if train_ds is not None else None

        test_loader = torch.utils.data.DataLoader(test_ds,
            batch_size=config.test_config.batch_size, shuffle=True, **kwargs_test) \
                if test_ds is not None else None

        return train_loader, test_loader
