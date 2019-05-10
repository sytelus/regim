from regim import *
from vae_cnn import *
import torch.nn
import torch
import math
import numpy as np
import random

class Image2ImageFolder(datasets.ImageFolder):
    def __getitem__(self, *kargs, **kwargs):
        sample, target = super(Image2ImageFolder, self).__getitem__(*kargs, **kwargs)
        if sample.max().item() > 1:
            print("bad img found")
        return sample, sample

def main(argv):
    config = Config.from_args(train_batch_size=16, epochs=50, rand_seed=1, learning_rate=1e-3)

    train_root = "D:\\datasets\\fruits-360\\Training"
    val_root = "D:\\datasets\\fruits-360\\Test"

    train_loader = torch.utils.data.DataLoader(
        Image2ImageFolder(train_root, transform=transforms.ToTensor()),
        batch_size = config.train_config.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        Image2ImageFolder(val_root, transform=transforms.ToTensor()),
        batch_size = config.test_config.batch_size, shuffle=True)

    pipeline = Pipeline(config)

    model = VAE_CNN()

    pipeline.run(model, train_loader, test_loader, task_type=Pipeline.TaskType.regression, 
        loss_module=None, optimizer='adam')

if __name__ == '__main__':
    sys.exit(main(sys.argv))
