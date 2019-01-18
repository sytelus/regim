import torch
import argparse


class Config:
    class CommonEpochConfig:
        def __init__(self, batch_size):
            self.batch_size = batch_size
    class TrainEpochConfig(CommonEpochConfig):
        def __init__(self, batch_size):
            super(Config.TrainEpochConfig, self).__init__(batch_size)
            self.lr = 0.01
            self.momentum = 0.5
    class TestEpochConfig(CommonEpochConfig):
        def __init__(self, batch_size):
            super(Config.TestEpochConfig, self).__init__(batch_size)
            self.batch_size = batch_size

    def __init__(self, train_batch_size, test_batch_size):
        self.train_config = Config.TrainEpochConfig(train_batch_size)
        self.test_config = Config.TestEpochConfig(test_batch_size)
        self.epochs = 100
        self.seed = 1


    @staticmethod
    def from_args(train_batch_size=64, test_batch_size=1000, 
                  learning_rate=0.01, momentum=0.5, weight_decay=0,
                  rand_seed=42,no_cuda_train=False, no_cuda_test=False):

        parser = argparse.ArgumentParser(description='PyTorch Deep Learning Pipeline')
        parser.add_argument('--batch-size', type=int, default=train_batch_size, metavar='N', #64
                            help='input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                            help='input batch size for testing')
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train')
        parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR', #0.01
                            help='learning rate')
        parser.add_argument('--momentum', type=float, default=momentum, metavar='M',
                            help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=weight_decay, metavar='WD',
                            help='Weight Decay')
        parser.add_argument('--no-cuda-train', action='store_true', default=no_cuda_train,
                            help='disables CUDA training')
        parser.add_argument('--no-cuda-test', action='store_true', default=no_cuda_test,
                            help='disables CUDA testing')
        parser.add_argument('--seed', type=int, default=rand_seed, metavar='S',
                            help='random seed (default: 42)')

        args = parser.parse_args()

        config = Config(args.batch_size, args.test_batch_size)

        config.train_config.use_cuda = not args.no_cuda_train and torch.cuda.is_available()
        config.train_config.device = torch.device("cuda" if config.train_config.use_cuda else "cpu")
        config.train_config.lr = args.lr
        config.train_config.momentum = args.momentum
        config.train_config.weight_decay = args.weight_decay
        
        config.test_config.use_cuda = not args.no_cuda_test and torch.cuda.is_available()
        config.test_config.device = torch.device("cuda" if config.test_config.use_cuda else "cpu")

        config.epochs = args.epochs

        if config.seed is not None:
            torch.manual_seed(config.seed)

        return config
