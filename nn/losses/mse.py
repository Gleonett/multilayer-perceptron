import torch
from torch import Tensor

from nn import BaseLoss


class MSELoss(BaseLoss):

    def update_output(self, input: Tensor, target: Tensor):
        return (target - input) ** 2

    def update_grad(self, input: Tensor, target: Tensor):
        return (input - target) * 2


def __mse_test(LossTest):
    shape = (200, 1)
    test = LossTest(torch.nn.MSELoss(), MSELoss(), 1e-4)
    inp = torch.rand(shape)
    y = torch.rand(shape)
    test(inp, y)


if __name__ == '__main__':
    from test.loss_test import LossTest

    __mse_test(LossTest)
