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

    def __init__(self, args):
        self.args = args
        self.train_config = Config.TrainEpochConfig(args.batch_size)
        self.test_config = Config.TestEpochConfig(args.test_batch_size)
        self.log_config = Config.LogConfig(debug_verbosity=args.debug_verbosity)
        self.epochs = args.epochs
        self.seed = args.seed


    @staticmethod
    def from_args(train_batch_size=64, test_batch_size=1000, 
                  learning_rate=0.01, momentum=0.0, weight_decay=0, debug_verbosity=None,
                  rand_seed=42,no_cuda_train=False, no_cuda_test=False, epochs=100,
                  arg_parser:argparse.ArgumentParser=None):

        parser = arg_parser or argparse.ArgumentParser(description='PyTorch Deep Learning Pipeline')
        parser.add_argument('--batch-size', type=int, default=train_batch_size, #64
                            help='input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=test_batch_size,
                            help='input batch size for testing')
        parser.add_argument('--epochs', type=int, default=epochs,
                            help='number of epochs to train')
        parser.add_argument('--lr', type=float, default=learning_rate, #0.01
                            help='learning rate')
        parser.add_argument('--momentum', type=float, default=momentum, 
                            help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=weight_decay,
                            help='Weight Decay')
        parser.add_argument('--no-cuda-train', action='store_true', default=no_cuda_train,
                            help='disables CUDA training')
        parser.add_argument('--no-cuda-test', action='store_true', default=no_cuda_test,
                            help='disables CUDA testing')
        parser.add_argument('--seed', type=int, default=rand_seed,
                            help='random seed (default: 42)')
        parser.add_argument('--debug-verbosity', type=int, default=debug_verbosity,
                            help='Debug Verbosity (default: -1)')


        args = parser.parse_args()

        config = Config(args)

        config.train_config.use_cuda = not args.no_cuda_train and torch.cuda.is_available()
        config.train_config.device = torch.device("cuda" if config.train_config.use_cuda else "cpu")
        config.train_config.lr = args.lr
        config.train_config.momentum = args.momentum
        config.train_config.weight_decay = args.weight_decay
        
        config.test_config.use_cuda = not args.no_cuda_test and torch.cuda.is_available()
        config.test_config.device = torch.device("cuda" if config.test_config.use_cuda else "cpu")

        if config.seed is not None:
            torch.manual_seed(config.seed)

        return config
