from regim import *
from vae_cnn import *
import torch.nn
import torch
import math
import numpy as np
import random
import torchvision.datasets as datasets

class Image2ImageFolder(datasets.ImageFolder):
    def __getitem__(self, *kargs, **kwargs):
        sample, target = super(Image2ImageFolder, self).__getitem__(*kargs, **kwargs)
        if sample.max().item() > 1:
            print("bad img found")
        return sample, sample

def main(argv):

    arg_parser = argparse.ArgumentParser(description='PyTorch Deep Learning Pipeline')
    arg_parser.add_argument('--train-root', type=str, default="/datasets/fruits-360/Training",
                        help='Directory for train dataset for fruits (default: /datasets/fruits-360/Training)')
    arg_parser.add_argument('--test-root', type=str, default="/datasets/fruits-360/Test",
                        help='Directory for test dataset for fruits (default: /datasets/fruits-360/Test)')

    config = Config.from_args(train_batch_size=16, epochs=50, rand_seed=1, learning_rate=1e-3, arg_parser=arg_parser)

    train_root = config.args.train_root
    val_root = config.args.test_root

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
