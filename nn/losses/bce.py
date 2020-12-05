import torch
from torch import Tensor

from nn.losses.base_loss import BaseLoss

EPSILON = 1e-7

class BCELoss(BaseLoss):

    def update_output(self, input: Tensor, target: Tensor):
        input = torch.clamp(input, EPSILON, 1.0 - EPSILON)
        return -torch.where(target == 1, input.log(), (1 - input).log())

    def update_grad(self, input: Tensor, target: Tensor):
        input = torch.clamp(input, EPSILON, 1.0 - EPSILON)
        return torch.where(target == 1, -1 / input, 1 / (1 - input))


def __bce_test(LossTest):
    shape = (200, 1)
    test = LossTest(torch.nn.BCELoss(), BCELoss(), 1e-4)
    inp = torch.rand((shape[0], 2))
    inp = torch.nn.Softmax(dim=1)(inp)
    inp = inp[:, [0]]
    y = torch.randint(low=0, high=2, size=shape).float()
    test(inp, y)


if __name__ == '__main__':
    from test.loss_test import LossTest

    __bce_test(LossTest)
