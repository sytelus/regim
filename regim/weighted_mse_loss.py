from torch.nn.modules.loss import _Loss

class WeightedMseLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedMseLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weight):
        ele_loss = weight * ((input - target) ** 2)
        row_loss = ele_loss.sum(dim=tuple(range(1, len(ele_loss.shape))))
        weight_sum = weight.sum(dim=tuple(range(1, len(weight.shape))))
        batch_item_loss = row_loss / weight_sum

        if self.reduction=='mean':
            return batch_item_loss.mean()
        elif self.reduction=='none':
            return batch_item_loss
        elif self.reduction=='sum':
            return batch_item_loss.sum()
        else:
            raise ValueError("reduction is not supported: {}".format(reduction))

