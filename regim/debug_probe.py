from tensorboardX import SummaryWriter
import torch.nn as nn
from torchvision import utils as tvutils

class DebugProbe:
    class LogSettings:
        model_graph = False
        false_preds = False
        param_histo_freq = 0
        param_histo_next_epoch = 0
        basic = False

    def __init__(self, exp_name, run_name, epoch_config, model, callbacks, metrics, log_dir='d:/tlogs/', log_settings = LogSettings()):
        self.log_writer = SummaryWriter(log_dir + exp_name + '/' + run_name)
        self.epoch_config = epoch_config
        self.run_name = run_name
        self.log_settings = log_settings
        self.metrics = metrics
        self.model = model
        
        callbacks.before_start.subscribe(self.on_before_start)
        callbacks.after_epoch.subscribe(self.on_after_epoch)
        callbacks.after_batch.subscribe(self.on_after_batch)

    def on_before_start(self, train_test, epochs, loader):
        print("[{}] CUDA: {}, Batch Size: {}, Data Size: {}, Batches: {}".format(self.run_name, 
            self.epoch_config.use_cuda, self.epoch_config.batch_size, len(loader.dataset), len(loader)))

    def on_after_batch(self, train_test, input, label, output, loss):
        # dump model diagram on first batch
        if self.log_settings.model_graph and self.metrics.metrics['batch_count'] == 1:
            self.log_writer.add_graph(self.model, input, verbose = True)

        # log false predictions
        if self.log_settings.false_preds:
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pairs = zip(input, pred.eq(label.view_as(pred)))
            incorrect =list( img for img, is_correct in pairs if not is_correct )
            img = tvutils.make_grid(incorrect)
            self.log_writer.add_image("False Pred", img, self.metrics.metrics['batch_count'])

    def on_after_epoch(self, *args,**kwargs):
        epoch = self.metrics.metrics['epoch_count']
        loss = self.metrics.metrics['loss_epoch']
        acuracy = self.metrics.metrics.get('accuracy_epoch', -1)

        if self.log_settings.basic:
            self.log_writer.add_scalar('loss', loss, epoch)
            self.log_writer.add_scalar('accuracy', acuracy, epoch)

        # histogram logging
        if self.log_settings.param_histo_freq > 0 and self.log_settings.param_histo_next_epoch == epoch:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                    self.log_writer.add_histogram(name + "/" + "weight", m.weight, epoch)
                    self.log_writer.add_histogram(name + "/" + "bias", m.bias, epoch)           
            self.log_settings.param_histo_next_epoch += self.log_settings.param_histo_freq

        print("[{}] Epoch: {}, loss: {:.4f}, accuracy: {:.4f}, Time: {:.2f}".format(self.run_name, epoch, 
            loss, acuracy, self.metrics.metrics['epoch_time']))
