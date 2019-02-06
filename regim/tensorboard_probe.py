import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import utils as tvutils

from .probe import *

class TensorboardProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = Probe.LogSettings(), log_dir='d:/tlogs/'):
        super(TensorboardProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings)
        self.log_writer = SummaryWriter(log_dir + exp_name + '/' + run_name)

    def on_after_batch(self, train_test, row, output, loss, loss_all):
        super(TensorboardProbe, self).on_after_batch(train_test, row, output, loss, loss_all)

        input, label, *_ = row

        # dump model diagram on first batch
        if self.log_settings.model_graph and self.metrics.stats.batch_index == 1:
            self.log_writer.add_graph(self.model, input, verbose = True)

        # log false predictions
        if self.log_settings.false_preds:
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pairs = zip(input, pred.eq(label.view_as(pred)))
            incorrect =list( img for img, is_correct in pairs if not is_correct )
            img = tvutils.make_grid(incorrect)
            self.log_writer.add_image("False Pred", img, self.metrics.stats.batch_index)

    def on_after_epoch(self, test_train, loader):
        super(TensorboardProbe, self).on_after_epoch(test_train, loader)

        epoch = self.metrics.stats.epoch_index
        loss = self.metrics.stats.epoch_loss
        accuracy = self.metrics.stats.epoch_accuracy

        if self.log_settings.basic:
            self.log_writer.add_scalar('loss', loss, epoch)
            self.log_writer.add_scalar('accuracy', accuracy, epoch)

        # histogram logging
        if self.log_settings.param_histo_freq > 0 and self.log_settings.param_histo_next_epoch == epoch:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                    self.log_writer.add_histogram(name + "/" + "weight", m.weight, epoch)
                    self.log_writer.add_histogram(name + "/" + "bias", m.bias, epoch)           
            self.log_settings.param_histo_next_epoch += self.log_settings.param_histo_freq