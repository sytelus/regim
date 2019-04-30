import torch
from torchvision import datasets, transforms

from .train_test import TrainTest
#from .probe import Probe
from .tensorwatch_probe import *
from .metrics import Metrics, ClassificationMetrics

class Pipeline:
    class TaskType:
        classification = 0
        regression = 1

    def __init__(self, config):
        self.config = config


    def find_lr(self, model, train_loader, task_type, 
                 loss_module, optimizer_name, initializer=None, 
                 start=1E-8, stop=10, steps=100):

        train_test = TrainTest(model, self.config, loss_module, 
                               optimizer, None, initializer, lr=1)

        if task_type == Pipeline.TaskType.classification:
            train_metrics = ClassificationMetrics(train_test.train_callbacks)
        elif task_type == Pipeline.TaskType.regression:
            train_metrics = Metrics(train_test.train_callbacks)
        else:
            raise NotImplementedError()

        train_probe = TensorWatchProbe('mnist_official_pipeline', 'train', self.config.train_config, 
                                 model, train_test.train_callbacks, train_metrics, self.config.log_config)

        train_test.find_lr(train_loader, start, stop, steps)


    def run(self, model, train_loader, test_loader, task_type, 
                 loss_module, optimizer, initializer=None, scheduler=None):

        train_test = TrainTest(model, self.config, loss_module,
                               optimizer, scheduler, initializer)

        if task_type == Pipeline.TaskType.classification:
            train_metrics = ClassificationMetrics(train_test.train_callbacks)
            test_metrics = ClassificationMetrics(train_test.test_callbacks)
        elif task_type == Pipeline.TaskType.regression:
            train_metrics = Metrics(train_test.train_callbacks)
            test_metrics = Metrics(train_test.test_callbacks)
        else:
            raise NotImplementedError()

        train_probe = TensorWatchProbe('mnist_official_pipeline', 'train', self.config.train_config, 
                                 model, train_test.train_callbacks, train_metrics, self.config.log_config, port_offset=0)
        test_probe = TensorWatchProbe('mnist_official_pipeline', 'test', self.config.test_config, 
                                model, train_test.test_callbacks, test_metrics, self.config.log_config, port_offset=1)

        train_test.fit(train_loader, test_loader)
