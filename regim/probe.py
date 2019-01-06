
class Probe:
    class LogSettings:
        model_graph = False
        false_preds = False
        param_histo_freq = 0
        param_histo_next_epoch = 0
        basic = False

    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = LogSettings()):
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
        print("[{}] Batch: {}, loss: {:.4f}, accuracy: {:.4f}, Time: {:.2f}".format(self.run_name, 
            self.metrics.metrics['batch_count'], loss, self.metrics.metrics.get('accuracy_batch', -1), 
            self.metrics.metrics['batch_time']))

    def on_after_epoch(self, test_train, dataset):
        epoch = self.metrics.metrics['epoch_count']
        loss = self.metrics.metrics['loss_epoch']
        accuracy = self.metrics.metrics.get('accuracy_epoch', -1)

        print("[{}] Epoch: {}, loss: {:.4f}, accuracy: {:.4f}, Time: {:.2f}".format(self.run_name, epoch, 
            loss, accuracy, self.metrics.metrics['epoch_time']))
