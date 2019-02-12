import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .event import Event
from .grad_rat_sched import GradientRatioScheduler
from .weighted_mse_loss import WeightedMseLoss
from . import utils

class TrainTest:
    class Callbacks:
        def __init__(self):
            self.before_start = Event()
            self.before_batch = Event()
            self.before_epoch = Event()

            self.after_end = Event()
            self.after_batch = Event()
            self.after_epoch = Event()

    class BatchState:
        def __init__(self, row, output=None, loss=None, loss_all=None):
            self.row, self.output, self.loss, self.loss_all = \
                row, output, loss, loss_all
            self.input, self.target, self.in_weight, self.tar_weight = row
            self.stop_training = False

    @staticmethod
    def get_optimizer_params(model, scheduler):
        if scheduler is None or not isinstance(scheduler, str):
            return model.parameters()
        elif scheduler=='grad_rat':
            return GradientRatioScheduler.get_opt_params(model, lr)
        else:
            raise ValueError('Scheduler named {} is not supported'.format(scheduler))

    @staticmethod
    def get_scheduler(scheduler_name, optimizer, train_callbacks):
        if scheduler_name is None:
            return None
        elif scheduler_name=='grad_rat':
            return GradientRatioScheduler(train_callbacks, optimizer)
        else:
            raise ValueError('Scheduler named {} is not supported'.format(scheduler_name))
            
    @staticmethod
    def get_optimizer(opt_params, name, model, config, lr=None, weight_decay=None):
        lr = lr or config.train_config.lr
        weight_decay=weight_decay or config.train_config.weight_decay

        if name=='sgd':
            return optim.SGD(opt_params, lr, 
                config.train_config.momentum, weight_decay=weight_decay)
        elif name=='adam':
            return optim.Adam(opt_params, lr, weight_decay=weight_decay)
        else:
            raise ValueError('Optimizer named {} is not supported'.format(name))


    def __init__(self, model, config, loss_module, 
                 optimizer, scheduler, initializer, lr=None, weight_decay=None):

        self.train_callbacks = TrainTest.Callbacks()
        self.test_callbacks = TrainTest.Callbacks()

        self.model = model
        self.config = config
        self.train_device = config.train_config.device
        self.test_device = config.test_config.device
        
        self.param_lr = TrainTest.get_optimizer_params(self.model, scheduler)

        if isinstance(optimizer, str):
            self.optimizer = TrainTest.get_optimizer(self.param_lr, optimizer,
                model, config, lr, weight_decay)
        else:
            self.optimizer = optimizer or optim.SGD(self.param_lr,
                lr or config.train_config.lr, config.train_config.momentum)

        if isinstance(scheduler, str):
            self.scheduler = TrainTest.get_scheduler(self.scheduler_name, optimizer,
                self.train_callbacks)
        else:
            self.scheduler = scheduler

        if initializer is not None:
            model.apply(initializer)

        self.loss_module = loss_module or torch.nn.NLLLoss(reduction='none')

        if self.train_device == self.test_device:
            self.model.to(self.train_device)

    @staticmethod
    def get_loss(loss_module, output, target, in_weight, tar_weight):
        # For NLLLoss: output is log probability of each class, target is class ID
        # default reduction is averaging loss over each sample
        if isinstance(loss_module, WeightedMseLoss):
            loss = loss_all = loss_module(output, target, tar_weight) 
        else: # other losses with target weights not supported yet
            loss = loss_all = loss_module(output, target) 

        # if loss isn't scaler value, compute its mean
        if len(loss_all.shape) != 0:
            loss = loss_all.mean()
            #loss = loss_all.sum()
        else:
            loss_all = torch.Tensor(target.shape)
            loss_all.fill_(loss.item())

        return loss_all, loss

    @staticmethod
    def to_std_tuple(row):
        unpacker = lambda a0,a1=None,a2=None,a3=None:(a0,a1,a2,a3)
        input, target, in_weight, tar_weight = unpacker(*row)
        return input, target, in_weight, tar_weight

    @staticmethod
    def to_device(device, row):
        return tuple(val.to(device) if val is not None else None for val in row)

    def train_epoch(self, train_loader):
        # if train/test device is not same then we need to move model to different device each time
        if self.train_device != self.test_device:
            self.model.to(self.train_device)

        self.model.train()

        self.train_callbacks.before_epoch.notify(self, train_loader)
        
        stop_training = False
        for row in train_loader:
            row = TrainTest.to_std_tuple(row)
            batch_state = TrainTest.BatchState(row)
            self.train_callbacks.before_batch.notify(self, batch_state)
            if batch_state.stop_training:
                stop_training = True
                break

            row = input, target, in_weight, tar_weight = TrainTest.to_device(self.train_device, row)

            self.optimizer.zero_grad()
            output = self.model(input)

            loss_all, loss = TrainTest.get_loss(self.loss_module, output, target, 
                in_weight, tar_weight)

            loss.backward()
            self.optimizer.step()

            batch_state = TrainTest.BatchState(row, output, loss.item(), loss_all)
            self.train_callbacks.after_batch.notify(self, batch_state)
            if batch_state.stop_training:
                stop_training = True
                break

        self.train_callbacks.after_epoch.notify(self, train_loader)

        if self.scheduler is not None:
            self.scheduler.step()

        return stop_training

    def test_epoch(self, test_loader):
        if self.train_device != self.test_device:
            self.model.to(self.test_device)
        self.model.eval()

        self.test_callbacks.before_epoch.notify(self, test_loader)

        with torch.no_grad():
            stop_training = False
            for row in test_loader:
                row = TrainTest.to_std_tuple(row)
                batch_state = TrainTest.BatchState(row)
                self.test_callbacks.before_batch.notify(self, batch_state)
                if batch_state.stop_training:
                    stop_training = True
                    break
    
                row = input, target, in_weight, tar_weight = TrainTest.to_device(self.test_device,row)

                output = self.model(input)
                
                loss_all, loss = TrainTest.get_loss(self.loss_module, output, target, 
                    in_weight, tar_weight)

                batch_state = TrainTest.BatchState(row, output, loss.item(), loss_all)
                self.test_callbacks.after_batch.notify(self, batch_state)
                if batch_state.stop_training:
                    stop_training = True
                    break
                
        self.test_callbacks.after_epoch.notify(self, test_loader)
        return stop_training

    def fit(self, train_loader, test_loader=None, epochs=None):
        epochs = epochs or self.config.epochs

        self.train_callbacks.before_start.notify(self, epochs, train_loader)
        if test_loader is not None:
            self.test_callbacks.before_start.notify(self, epochs, test_loader)

        for epoch in range(epochs):
            stop_training = self.train_epoch(train_loader)
            if stop_training:
                break
            if test_loader is not None:
                stop_training = self.test_epoch(test_loader)
                if stop_training:
                    break

        self.train_callbacks.after_end.notify(self, epochs, train_loader)
        if test_loader is not None:
            self.test_callbacks.after_end.notify(self, epochs, test_loader)

    def find_lr(self, train_loader, start=1E-8, stop=5.5, steps=100):
        def exp_ann(i):
            lr = start * (stop/start) ** (i / (steps-1))
            return lr

        scheduler = lr_scheduler.LambdaLR(self.optimizer, exp_ann)
        self.scheduler = scheduler # allow access in callbacks
        batch_index = 0

        def on_after_batch(train_test, batch_state):
            scheduler.step()
            batch_index += 1
            if batch_index > steps:
                batch_state.stop_training = True

        self.train_callbacks.after_batch.subscribe(on_after_batch)

        self.fit(train_loader)
