from regim import *
from mnist_official_model import MnistOfficialModel
from mnist_mlp_model import MnistMlpModel
import torch.nn
import torch
import math
import numpy as np
import random
#import rolling

def main():
    config = Config.from_args(momentum=0.0, learning_rate=0.1)

    use_mlp = True

    if not use_mlp:
        train_ds, test_ds = DataUtils.mnist_datasets()
        model = MnistOfficialModel()
    else:
        train_ds, test_ds = DataUtils.mnist_datasets(True)
        model = MnistMlpModel()

    train_loader, test_loader = DataUtils.get_dataloaders(config, train_ds, test_ds)

    pipeline = Pipeline(config)

    #pipeline.find_lr(model, train_loader, task_type=Pipeline.TaskType.classification, 
    #    loss_module=torch.nn.NLLLoss(reduction='none'), optimizer='sgd', stop=2)

    pipeline.run(model, train_loader, test_loader, task_type=Pipeline.TaskType.classification, 
        loss_module=torch.nn.NLLLoss(reduction='none'), optimizer='sgd')

def find_lr():
    config = Config.from_args(no_cuda_train=True, no_cuda_test=True)
    config.epochs = 3

    use_mlp = True

    if not use_mlp:
        train_ds, test_ds = DataUtils.mnist_datasets()
        model = MnistOfficialModel()
    else:
        train_ds, test_ds = DataUtils.mnist_datasets(True)
        model = MnistMlpModel()

    loss_module = torch.nn.NLLLoss()
    train_loader, test_loader = DataUtils.get_dataloaders(config, train_ds, test_ds)

    pipeline = Pipeline(config)
    pipeline.run(model, train_loader, test_loader, task_type=Pipeline.TaskType.classification, 
        loss_module=torch.nn.NLLLoss(), optimizer='sgd')

    train_loader_iter = iter(train_loader)
    input_t, label_t = next(train_loader_iter)
    input_t = input_t[0:1,:]
    label_t = label_t[0:1]

    train_device = config.train_config.device
    test_device = config.test_config.device
    model.to(train_device)
    model.eval()
    with torch.no_grad():
        input = torch.rand_like(input_t)*2-1
        input = input.to(train_device)
        output = model(input)
        label = torch.randint_like(label_t, output.shape[1]) 
        label = label.to(train_device)

        loss = loss_module(output, label)
        print(loss)

        max_log = 0
        max_samples = 1000

        for m in model.children():
            sh = m.weight.shape
            max_e = 8
            sample_count = 0
            last_time = getTime()
            logs = []
            for sample_count in range(max_samples):
                i = random.randrange(0, sh[0])
                j = random.randrange(0, sh[1])
                w0 = m.weight[i,j].item()
                last_loss = loss
                for k in range(max_e):
                    w = w0 + math.pow(10, -(max_e-k-1))
                    m.weight[i,j] = w
                    output = model(input)
                    this_loss = loss_module(output, label)
                    this_log = abs(math.log10(this_loss/last_loss))
                    if this_log > 0:
                        logs.append(this_log)
                    #max_log = max(max_log, this_log)
                    #if this_log > 1:
                    #    print(i, j, this_loss, last_loss, k, w)
                    last_loss = this_loss
                m.weight[i,j] = w0
                sample_count += 1
                #if sample_count % 100 == 0:
                #    elapsed = getElapsedTime(last_time)
                #    print("Elapsed: ", elapsed)
                #    last_time = getTime()
            n=len(logs)
            #print(m, n, list(rolling.Max(logs, n)), list(rolling.Mean(logs, n)), list(rolling.Median(logs, n)), list(rolling.Min(logs, n)))

if __name__ == '__main__':
    main()