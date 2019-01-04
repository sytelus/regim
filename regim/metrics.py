import timeit

class Metrics:
    def __init__(self, callbacks):
        self.metrics = {}

        self.metrics['epoch_count'] = 0
        self.metrics['loss_sum'] = 0
        self.metrics['loss_epoch'] = 0
        self.metrics['batch_count'] = 0
        self.metrics['batch_time'] = 0.
        self.metrics['epoch_time'] = 0.
        
        callbacks.before_batch.subscribe(self.on_before_batch)
        callbacks.before_epoch.subscribe(self.on_before_epoch)
        callbacks.after_batch.subscribe(self.on_after_batch)
        callbacks.after_epoch.subscribe(self.on_after_epoch)

    def on_before_epoch(self, *args,**kwargs):
        self.epoch_start_time = timeit.default_timer()
        self.metrics['loss_sum'] = 0
        self.metrics['batch_count'] = 0
        self.metrics['epoch_count'] += 1

    def on_before_batch(self, *args,**kwargs):
        self.batch_start_time = timeit.default_timer()
        self.metrics['batch_count'] += 1

    def on_after_batch(self, test_train, input, label, output, loss):
        self.metrics['batch_time'] = timeit.default_timer() - self.epoch_start_time
        self.metrics['loss_sum'] += loss * len(input)

    def on_after_epoch(self, test_train, dataset):
        self.metrics['epoch_time'] = timeit.default_timer() - self.epoch_start_time
        self.metrics['loss_epoch'] = self.metrics['loss_sum'] / len(dataset)

class ClassificationMetrics(Metrics):
    def on_before_epoch(self, *args,**kwargs):
        super(ClassificationMetrics, self).on_before_epoch(*args,**kwargs)
        self.metrics['correct_sum'] = 0
        self.metrics['accuracy_epoch'] = 0

    def on_after_batch(self, test_train, input, label, output, loss):
        super(ClassificationMetrics, self).on_after_batch(test_train, input, label, output, loss)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        self.metrics['correct_sum'] += pred.eq(label.view_as(pred)).sum().item()

    def on_after_epoch(self, test_train, dataset):
        super(ClassificationMetrics, self).on_after_epoch(test_train, dataset)
        self.metrics['accuracy_epoch'] = self.metrics['correct_sum'] / len(dataset)


        

