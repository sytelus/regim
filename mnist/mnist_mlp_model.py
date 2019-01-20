import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistMlpModel(nn.Module):
    def __init__(self):
        super(MnistMlpModel, self).__init__()
        self.fc0 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(self.fc0(input_))
        h1_d = F.dropout(h1, p=0.5, training=self.training)  # drop rate 0.25, keep rate 0.75
        h2 = F.relu(self.fc1(h1_d))
        h2_d = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc2(h2_d)
        h4 = F.log_softmax(h3, dim=1)
        return h4

