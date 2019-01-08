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

    def on_before_batch(self, train_test, input, label):
        self.batch_start_time = timeit.default_timer()
        self.stats.batch_index += 1
        self.stats.epochf = self.stats.epoch_index + \
            (float(self.stats.batch_index) / self.stats.batch_count)

    def on_after_batch(self, test_train, input, label, output, loss):
        self.stats.batch_time = timeit.default_timer() - self.epoch_start_time
        self.stats.loss_sum += loss * len(input)
        self.stats.batch_loss = loss

    def on_after_epoch(self, test_train, loader):
        self.stats.epoch_time = timeit.default_timer() - self.epoch_start_time
        self.stats.epoch_loss = self.stats.loss_sum / len(loader.dataset)

class ClassificationMetrics(Metrics):
    def __init__(self, *args,**kwargs):
        super(ClassificationMetrics, self).__init__(*args,**kwargs)
        self.stats.correct_sum = 0
        self.stats.epoch_accuracy = 0.
        self.stats.batch_accuracy = 0.

    def on_before_epoch(self, train_test, loader):
        super(ClassificationMetrics, self).on_before_epoch(train_test, loader)
        self.stats.correct_sum = 0

    def on_after_batch(self, test_train, input, label, output, loss):
        super(ClassificationMetrics, self).on_after_batch(test_train, input, label, output, loss)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        batch_correct = pred.eq(label.view_as(pred)).sum().item()
        self.stats.correct_sum += batch_correct
        self.stats.batch_accuracy = float(batch_correct) / len(input)

    def on_after_epoch(self, test_train, loader):
        super(ClassificationMetrics, self).on_after_epoch(test_train, loader)
        self.stats.epoch_accuracy = self.stats.correct_sum / len(loader.dataset)


        

