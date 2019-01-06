import torch
from torchvision import datasets, transforms

from .train_test import TrainTest
from .probe import Probe
from .metrics import Metrics, ClassificationMetrics

class Pipeline:
    class TaskType:
        classification = 0
        regression = 1

    def __init__(self, config):
        self.config = config

    def run(self, model, train_loader, test_loader, task_type, 
                 loss_module, optimizer, initializer=None):

        train_test = TrainTest(model, self.config, optimizer, loss_module, initializer)

        if task_type == Pipeline.TaskType.classification:
            train_metrics = ClassificationMetrics(train_test.train_callbacks)
            test_metrics = ClassificationMetrics(train_test.test_callbacks)
        elif task_type == Pipeline.TaskType.regression:
            train_metrics = Metrics(train_test.train_callbacks)
            test_metrics = Metrics(train_test.test_callbacks)
        else:
            raise NotImplementedError()

        train_probe = Probe('mnist_official_pipeline', 'train', self.config.train_config, 
                                 model, train_test.train_callbacks, train_metrics)
        test_probe = Probe('mnist_official_pipeline', 'test', self.config.test_config, 
                                model, train_test.test_callbacks, test_metrics)

        train_test.train_model(self.config.epochs, train_loader, test_loader)
