from .probe import *
import longview as lv

class LongviewProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = Probe.LogSettings()):
        super(LongviewProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings)
        self.lv = lv.WatchServer()
        self.metrics = metrics
        self.lv.set_vars(model=model, metrics=metrics.stats)

        #TODO manage client server connections better
        #lv.wait_key("Press any key to continue...")

    def on_after_batch(self, train_test, batch_state):
        super(LongviewProbe, self).on_after_batch(train_test, batch_state)
        self.lv.set_vars("batch", x=self.metrics.stats.epochf, batch=batch_state, tt=train_test)

    def on_after_epoch(self, train_test, loader):
        super(LongviewProbe, self).on_after_epoch(train_test, loader)
        self.lv.end_event("batch")
        self.lv.set_vars("epoch", tt=train_test)

    #TODO log end epoch

