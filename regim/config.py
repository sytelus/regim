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
    class LogConfig:
        def __init__(self, model_graph=False, false_preds=False, param_histo_freq=0,
                     param_histo_next_epoch=0, basic=False, debug_verbosity=-1):
            self.model_graph = model_graph
            self.false_preds = false_preds
            self.param_histo_freq = param_histo_freq
            self.param_histo_next_epoch = param_histo_next_epoch
            self.basic = basic
            self.debug_verbosity = debug_verbosity

    def __init__(self, train_batch_size, test_batch_size, debug_verbosity):
        self.train_config = Config.TrainEpochConfig(train_batch_size)
        self.test_config = Config.TestEpochConfig(test_batch_size)
        self.log_config = Config.LogConfig(debug_verbosity=debug_verbosity)
        self.epochs = 100
        self.seed = 1


    @staticmethod
    def from_args(train_batch_size=64, test_batch_size=1000, 
                  learning_rate=0.01, momentum=0.0, weight_decay=0, debug_verbosity=None,
                  rand_seed=42,no_cuda_train=False, no_cuda_test=False, epochs=100):

        parser = argparse.ArgumentParser(description='PyTorch Deep Learning Pipeline')
        parser.add_argument('--batch-size', type=int, default=train_batch_size, metavar='N', #64
                            help='input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                            help='input batch size for testing')
        parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
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
        parser.add_argument('--debug_verbosity', type=int, default=debug_verbosity, metavar='S',
                            help='Debug Verbosity (default: -1)')

        args = parser.parse_args()

        config = Config(args.batch_size, args.test_batch_size, debug_verbosity=args.debug_verbosity)

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
