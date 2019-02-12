
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

    def on_after_batch(self, train_test, batch_state):
        # print("[{}] {}".format(self.run_name, self.metrics.get_after_batch_summary()))
        pass

    def on_after_epoch(self, test_train, loader):
        print("[{}] {}".format(self.run_name, self.metrics.get_after_epoch_summary()))
