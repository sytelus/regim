from .probe import *
import longview as lv

class LongviewProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = Probe.LogSettings()):
        super(LongviewProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings)
        self.lv = lv.WatchServer()
        self.metrics = metrics
        self.lv.log_globals(model=model, metrics=metrics.stats)

        #lv.wait_key("Press any key to continue...")

    def on_after_batch(self, train_test, input, label, output, loss, loss_all):
        super(LongviewProbe, self).on_after_batch(train_test, input, label, output, loss, loss_all)
        self.lv.log_event("batch", x=self.metrics.stats.epochf,
            input=input, label=label, output=output, loss=loss, loss_all=loss_all, 
            tt=train_test)

    def on_after_epoch(self, train_test, loader):
        super(LongviewProbe, self).on_after_epoch(train_test, loader)
        self.lv.end_event("batch")
        self.lv.log_event("epoch", tt=train_test)

    #TODO log end epoch

