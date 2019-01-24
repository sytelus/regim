import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .event import Event
from .grad_rat_sched import GradientRatioScheduler

class TrainTest:
    class Callbacks:
        def __init__(self):
            self.before_start = Event()
            self.before_batch = Event()
            self.before_epoch = Event()

            self.after_end = Event()
            self.after_batch = Event()
            self.after_epoch = Event()


    def get_optimizer_params(self, model, scheduler_name, optimizer_name):
        if scheduler_name is None:
            return model.parameters()
        elif scheduler_name=='grad_rat':
            return GradientRatioScheduler.get_params_base_lr(model, lr)
        else:
            raise ValueError('Scheduler named {} is not supported'.format(scheduler_name))
    def get_scheduler(self, scheduler_name, optimizer):
        if scheduler_name is None:
            return None
        elif scheduler_name=='grad_rat':
            return GradientRatioScheduler(optimizer)
        else:
            raise ValueError('Scheduler named {} is not supported'.format(scheduler_name))
            
    def get_optimizer(self, name, model, config, lr=None, weight_decay=None):
        lr = lr or config.train_config.lr
        weight_decay=weight_decay or config.train_config.weight_decay

        if name=='sgd':
            return optim.SGD(self.param_lr, lr, 
                config.train_config.momentum, weight_decay=weight_decay)
        elif name=='adam':
            return optim.Adam(self.param_lr, lr, weight_decay=weight_decay)
        else:
            raise ValueError('Optimizer named {} is not supported'.format(name))

    def __init__(self, model, config, 
                 optimizer, loss_module, initializer, scheduler):

        self.model = model
        self.config = config
        self.train_callbacks = TrainTest.Callbacks()
        self.test_callbacks = TrainTest.Callbacks()
        self.param_lr = []
        self.scheduler_name = scheduler if isinstance(scheduler, str) else None
        self.optimizer_name = optimizer if isinstance(optimizer, str) else None

        self.train_device = config.train_config.device
        self.test_device = config.test_config.device
        
        self.param_lr = self.get_optimizer_params(self.model, self.scheduler_name, 
            self.optimizer_name)
        optimizer = self.get_optimizer(self.optimizer_name, model, config)

        #TODO not consistent?
        self.optimizer = optimizer or optim.SGD(model.parameters(), \
            config.train_config.lr, config.train_config.momentum)

        self.scheduler= self.get_scheduler(self.scheduler_name, optimizer)

        if initializer is not None:
            model.apply(initializer)

        self.loss_module = loss_module or torch.nn.NLLLoss(reduction='none')

        if self.train_device == self.test_device:
            self.model.to(self.train_device)

    def train_epoch(self, train_loader, optimizer=None, batch_sched=None, max_batches=None):
        optimizer = optimizer or self.optimizer
        self.batch_sched = batch_sched # provide access to callbacks

        # if train/test device is not same then we need to move model to different device each time
        if self.train_device != self.test_device:
            self.model.to(self.train_device)

        self.model.train()

        self.train_callbacks.before_epoch.notify(self, train_loader)
        
        batch_index = -1
        for input, label in train_loader:
            batch_index += 1
            if max_batches is not None and batch_index >= max_batches:
                break;

            self.train_callbacks.before_batch.notify(self, input, label)
            input, label = input.to(self.train_device), label.to(self.train_device)
            optimizer.zero_grad()
            output = self.model(input)
            # For NLLLoss: output is log probability of each class, label is class ID
            #default reduction is averaging loss over each sample
            loss = loss_all = self.loss_module(output, label) 
            if len(loss_all.shape) != 0:
                loss = loss_all.mean()
                #loss = loss_all.sum()
            else:
                loss_all = torch.Tensor(label.shape)
                loss_all.fill_(loss.item())
            loss.backward()
            optimizer.step()

            #TODO need to remove this - this is becoming confusing
            if batch_sched is not None:
                batch_sched.step()
            if self.scheduler is not None:
                #TODO check if method exist
                self.scheduler.on_after_batch()

            #self.model.eval() # disable layers like dropout
            #with torch.no_grad():
            #    output = self.model(input)
            #self.model.train()

            self.train_callbacks.after_batch.notify(self, input, label, output, 
                loss.item(), loss_all)

        self.train_callbacks.after_epoch.notify(self, train_loader)

        if self.scheduler is not None:
            self.scheduler.step()

        return batch_index

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

    def find_lr(self, train_loader, start=1E-8, stop=5.5, steps=100):
        def exp_ann(i):
            lr = start * (stop/start) ** (i / (steps-1))
            return lr

        optimizer = None
        if self.optimizer_name is not None:
            optimizer = self.get_optimizer(self.optimizer_name, self.model, 
                self.config, lr=1)
        else:
            raise NotImplementedError()

        scheduler = lr_scheduler.LambdaLR(optimizer, exp_ann)

        self.train_callbacks.before_start.notify(self, None, train_loader)

        batch_count = 0
        while batch_count < steps:
            batch_count = batch_count + 1 + \
                self.train_epoch(train_loader, optimizer=optimizer, batch_sched=scheduler, 
                    max_batches=steps)

        self.train_callbacks.after_end.notify(self, None, train_loader)
