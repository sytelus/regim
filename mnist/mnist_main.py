from regim import *
from mnist_official_model import MnistOfficialModel
from mnist_mlp_model import MnistMlpModel
import torch.nn
import torch
import math
import numpy as np
import random


def main(argv):

    #Baseline: lr=0.01, momentum=0.5
    #[train] CUDA: True, Batch Size: 64, Data Size: 60000, Batches: 938
    #[test] CUDA: True, Batch Size: 1000, Data Size: 10000, Batches: 10
    #[train] Epoch: 0, loss: 0.7641, Time: 37.08, accuracy: 0.7639
    #[test] Epoch: 0, loss: 0.2806, Time: 4.76, accuracy: 0.9156
    #[train] Epoch: 1, loss: 0.3453, Time: 36.01, accuracy: 0.8961
    #[test] Epoch: 1, loss: 0.2004, Time: 4.44, accuracy: 0.9389
    #[train] Epoch: 2, loss: 0.2725, Time: 42.38, accuracy: 0.9198
    #[test] Epoch: 2, loss: 0.1649, Time: 5.25, accuracy: 0.9506
    #[train] Epoch: 3, loss: 0.2313, Time: 40.64, accuracy: 0.9315
    #[test] Epoch: 3, loss: 0.1408, Time: 4.69, accuracy: 0.9587
    #[train] Epoch: 4, loss: 0.2052, Time: 37.60, accuracy: 0.9394
    #[test] Epoch: 4, loss: 0.1238, Time: 4.93, accuracy: 0.9618
    
    config = Config.from_args(momentum=0.5)

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
    sys.exit(main(sys.argv))