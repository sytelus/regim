import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from .event import Event

class TrainTest:
    class Callbacks:
        def __init__(self):
            self.before_start = Event()
            self.before_batch = Event()
            self.before_epoch = Event()

            self.after_end = Event()
            self.after_batch = Event()
            self.after_epoch = Event()

    def get_optimizer(self, name, model, config):
        if name=='sgd':
            return optim.SGD(model.parameters(), config.train_config.lr, config.train_config.momentum)
        else:
            raise ValueError('Optimizer named {} is not supported'.format(name))

    def __init__(self, model, config, 
                 optimizer, loss_module, initializer):

        self.model = model
        self.train_callbacks = TrainTest.Callbacks()
        self.test_callbacks = TrainTest.Callbacks()

        self.train_device = config.train_config.device
        self.test_device = config.test_config.device

        if isinstance(optimizer, str):
            optimizer = self.get_optimizer(optimizer, model, config)

        if initializer is not None:
            model.apply(initializer)

        self.optimizer = optimizer or optim.SGD(model.parameters(), config.train_config.lr, config.train_config.momentum)
        self.loss_module = loss_module or torch.nn.NLLLoss(reduction='none')

        if self.train_device == self.test_device:
            self.model.to(self.train_device)

    def train_epoch(self, train_loader):
        # if train/test device is not same then we need to move model to different device each time
        if self.train_device != self.test_device:
            self.model.to(self.train_device)

        self.model.train()

        self.train_callbacks.before_epoch.notify(self, train_loader)

        for input, label in train_loader:
            self.train_callbacks.before_batch.notify(self, input, label)
            input, label = input.to(self.train_device), label.to(self.train_device)
            self.optimizer.zero_grad()
            output = self.model(input)
            # For NLLLoss: output is log probability of each class, label is class ID
            #default reduction is averaging loss over each sample
            loss = loss_all = self.loss_module(output, label) 
            if len(loss_all.shape) != 0:
                loss = loss_all.mean()
            else:
                loss_all = torch.Tensor(label.shape)
                loss_all.fill_(loss.item())
            loss.backward()
            self.optimizer.step()

            #self.model.eval() # disable layers like dropout
            #with torch.no_grad():
            #    output = self.model(input)
            #self.model.train()

            self.train_callbacks.after_batch.notify(self, input, label, output, loss.item(), loss_all)

        self.train_callbacks.after_epoch.notify(self, train_loader)

    def test_epoch(self, test_loader):
        if self.train_device != self.test_device:
            self.model.to(self.test_device)
        self.model.eval()

        self.test_callbacks.before_epoch.notify(self, test_loader)

        with torch.no_grad():
            for input, label in test_loader:
                self.test_callbacks.before_batch.notify(self, input, label)
                input, label = input.to(self.test_device), label.to(self.test_device)
                output = self.model(input)
                loss = loss_all = self.loss_module(output, label)
                if len(loss_all.shape) != 0:
                    loss = loss_all.mean()
                else:
                    loss_all = torch.Tensor(label.shape)
                    loss_all.fill_(loss.item())
                self.test_callbacks.after_batch.notify(self, input, label, output, loss.item(), loss_all)
                
        self.test_callbacks.after_epoch.notify(self, test_loader)

    def train_model(self, epochs, train_loader, test_loader):
        self.train_callbacks.before_start.notify(self, epochs, train_loader)
        self.test_callbacks.before_start.notify(self, epochs, test_loader)

        for epoch in range(epochs):
            self.train_epoch(train_loader)
            self.test_epoch(test_loader)

        self.train_callbacks.after_end.notify(self, epochs, train_loader)
        self.test_callbacks.after_end.notify(self, epochs, test_loader)
