import timeit

class Metrics:
    class Stats:
        def __init__(self):
            self.reset()
        def reset(self):
            self.epoch_index = -1
            self.epoch_loss:float = None
            self.epoch_time = 0.
            self.batch_index = -1
            self.batch_loss:float = None
            self.batch_time = 0.
            self.loss_sum = 0.
            self.correct_sum = 0
            self.epoch_count = 0
            self.batch_count = 0
            self.epochf = 0

    def __init__(self, callbacks):
        self.stats = Metrics.Stats()
        callbacks.before_start.subscribe(self.on_before_start)
        callbacks.before_batch.subscribe(self.on_before_batch)
        callbacks.before_epoch.subscribe(self.on_before_epoch)
        callbacks.after_batch.subscribe(self.on_after_batch)
        callbacks.after_epoch.subscribe(self.on_after_epoch)

    def on_before_start(self, train_test, epochs, loader):
        self.stats.reset()
        self.stats.epoch_count = epochs

    def on_before_epoch(self, train_test, loader):
        self.epoch_start_time = timeit.default_timer()
        self.stats.loss_sum = 0
        self.stats.batch_index = -1
        self.stats.epoch_index += 1
        self.stats.epochf = self.stats.epoch_index
        self.stats.batch_count = len(loader)

    def on_before_batch(self, train_test, batch_state):
        self.batch_start_time = timeit.default_timer()
        self.stats.batch_index += 1
        self.stats.epochf = self.stats.epoch_index + \
            (float(self.stats.batch_index) / self.stats.batch_count)

    def on_after_batch(self, test_train, batch_state):
        self.stats.batch_time = timeit.default_timer() - self.epoch_start_time
        self.stats.loss_sum += batch_state.loss_all.sum().item()
        self.stats.batch_loss = batch_state.loss
        self.stats.batch_loss_all = batch_state.loss_all

    def on_after_epoch(self, test_train, loader):
        self.stats.epoch_time = timeit.default_timer() - self.epoch_start_time
        self.stats.epoch_loss = self.stats.loss_sum / len(loader.dataset)

    def get_after_batch_summary(self):
        return "Batch: {}, loss: {:.4f}, Time: {:.2f}".format( \
            self.stats.epochf, self.stats.batch_loss, self.stats.batch_time)
    def get_after_epoch_summary(self):
        return "Epoch: {}, loss: {:.4f}, Time: {:.2f}".format( \
            self.stats.epoch_index, self.stats.epoch_loss, self.stats.epoch_time)

class ClassificationMetrics(Metrics):
    def __init__(self, *args,**kwargs):
        super(ClassificationMetrics, self).__init__(*args,**kwargs)
        self.stats.correct_sum = 0
        self.stats.epoch_accuracy = 0.
        self.stats.batch_accuracy = 0.

    def on_before_epoch(self, train_test, loader):
        super(ClassificationMetrics, self).on_before_epoch(train_test, loader)
        self.stats.correct_sum = 0

    def on_after_batch(self, test_train, batch_state):
        super(ClassificationMetrics, self).on_after_batch(test_train, batch_state)
        # get the index of the max log-probability
        pred = batch_state.output.max(1, keepdim=True)[1]
        batch_correct = pred.eq(batch_state.target.view_as(pred)).sum().item()
        self.stats.correct_sum += batch_correct
        self.stats.batch_accuracy = float(batch_correct) / len(batch_state.input)

    def on_after_epoch(self, test_train, loader):
        super(ClassificationMetrics, self).on_after_epoch(test_train, loader)
        self.stats.epoch_accuracy = self.stats.correct_sum / len(loader.dataset)

    def get_after_batch_summary(self):
        s = super(ClassificationMetrics, self).get_after_batch_summary()
        return "{}, accuracy: {:.4f}".format( \
            s, self.stats.batch_accuracy)
    def get_after_epoch_summary(self):
        s = super(ClassificationMetrics, self).get_after_epoch_summary()
        return "{}, accuracy: {:.4f}".format( \
            s, self.stats.epoch_accuracy)


        

