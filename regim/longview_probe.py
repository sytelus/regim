from .probe import *
import longview as lv

class LongviewProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = Probe.LogSettings()):
        super(LongviewProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings)
        self.lv = lv.WatchServer()
        self.lv.log_globals(model=self.model, metrics=metrics.stats)

    def on_after_batch(self, train_test, input, label, output, loss):
        super(LongviewProbe, self).on_after_batch(train_test, input, label, output, loss)
        self.lv.log_event("batch", input=input, label=label, output=output, loss=loss)

    def on_after_epoch(self, test_train, dataset):
        super(LongviewProbe, self).on_after_epoch(test_train, dataset)
        self.lv.end_event("batch")
