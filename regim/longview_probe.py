from .probe import *
import longview as lv

class LongviewProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = LogSettings()):
        super(LongviewProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, log_dir, metrics)
        self.lv = lv.WatchServer()
        self.lv.log_globals(m=self.model, t=self.metrics)

    def on_after_batch(self, train_test, input, label, output, loss):
        super(LongviewProbe, self).on_after_batch(train_test, input, label, output, loss)
        self.lv.log_event("batch", i=input, lbl=label, o=output, l=loss)

    def on_after_epoch(self, test_train, dataset):
        super(LongviewProbe, self).on_after_batch(*args, **kwargs)
        self.lv.end_event("batch")
