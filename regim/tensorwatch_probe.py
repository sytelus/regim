from .probe import *
import tensorwatch as tw

class TensorWatchProbe(Probe):
    def __init__(self, exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings = Probe.LogSettings()):
        super(TensorWatchProbe, self).__init__(exp_name, run_name, epoch_config, model, 
            callbacks, metrics, log_settings)
        self.tw = tw.WatchServer()
        self.metrics = metrics
        self.tw.set_globals(model=model, metrics=metrics.stats)

        #TODO manage client server connections better
        #tw.wait_key("Press any key to continue...")

    def on_after_batch(self, train_test, batch_state):
        super(TensorWatchProbe, self).on_after_batch(train_test, batch_state)
        self.tw.observe(event_name='batch', x=self.metrics.stats.epochf, batch=batch_state, tt=train_test)

    def on_after_epoch(self, train_test, loader):
        super(TensorWatchProbe, self).on_after_epoch(train_test, loader)
        self.tw.end_event('batch')
        self.tw.observe(event_name='epoch', tt=train_test)

    #TODO log end epoch

