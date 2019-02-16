import regim
import torch
import sys

def main(argv):
    train_ds, test_ds = regim.DataUtils.mnist_datasets()
    #ds = torch.utils.data.ConcatDataset((train_ds, test_ds))
    m, v = regim.DataUtils.channel_norm(train_ds)
    print(m, v)

if __name__ == '__main__':
    ret = main(sys.argv)
    sys.exit(ret)