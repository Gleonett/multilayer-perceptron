import torch
from torch import Tensor

from nn import BaseLoss


class BCELoss(BaseLoss):

    def update_output(self, input: Tensor, target: Tensor):
        return -torch.where(target == 1, input.log(), (1 - input).log())

    def update_grad(self, input: Tensor, target: Tensor):
        return torch.where(target == 1, -1 / input, 1 / (1 - input))


def __bce_test(LossTest):
    shape = (200, 1)
    test = LossTest(torch.nn.BCELoss(), BCELoss(), 1e-4)
    inp = torch.rand(shape)
    y = torch.randint(low=0, high=2, size=shape).float()
    test(inp, y)


if __name__ == '__main__':
    from test.loss_test import LossTest

    __bce_test(LossTest)
