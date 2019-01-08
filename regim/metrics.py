import timeit

class Metrics:
    def __init__(self, callbacks):
        self.metrics = {}

        self.vals.epoch_count = 0
        self.vals.loss_sum = 0.
        self.vals.epoch_loss = 0.
        self.vals.batch_count = 0
        self.vals.batch_time = 0.
        self.vals.epoch_time = 0.
        self.vals.batch_accuracy = 0.
        self.vals.epoch_accuracy = 0.
        self.vals.correct_sum = 0
                
        callbacks.before_batch.subscribe(self.on_before_batch)
        callbacks.before_epoch.subscribe(self.on_before_epoch)
        callbacks.after_batch.subscribe(self.on_after_batch)
        callbacks.after_epoch.subscribe(self.on_after_epoch)

    def on_before_epoch(self, *args,**kwargs):
        self.epoch_start_time = timeit.default_timer()
        self.vals.loss_sum = 0
        self.vals.batch_count = 0
        self.vals.epoch_count += 1

    def on_before_batch(self, *args,**kwargs):
        self.batch_start_time = timeit.default_timer()
        self.vals.batch_count += 1

    def on_after_batch(self, test_train, input, label, output, loss):
        self.vals.batch_time = timeit.default_timer() - self.epoch_start_time
        self.vals.loss_sum += loss * len(input)

    def on_after_epoch(self, test_train, dataset):
        self.vals.epoch_time = timeit.default_timer() - self.epoch_start_time
        self.vals.epoch_loss = self.vals.loss_sum / len(dataset)

class ClassificationMetrics(Metrics):
    def on_before_epoch(self, *args,**kwargs):
        super(ClassificationMetrics, self).on_before_epoch(*args,**kwargs)
        self.vals.correct_sum = 0
        self.vals.epoch_accuracy = 0

    def on_after_batch(self, test_train, input, label, output, loss):
        super(ClassificationMetrics, self).on_after_batch(test_train, input, label, output, loss)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        batch_correct = pred.eq(label.view_as(pred)).sum().item()
        self.vals.correct_sum += batch_correct
        self.vals.batch_accuracy = batch_correct / len(input)

    def on_after_epoch(self, test_train, dataset):
        super(ClassificationMetrics, self).on_after_epoch(test_train, dataset)
        self.vals.epoch_accuracy = self.vals.correct_sum / len(dataset)


        

