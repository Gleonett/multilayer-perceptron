import torch

from nn import BaseLoss


class BCELoss(BaseLoss):

    def forward(self, p: torch.Tensor, y: torch.Tensor):
        return -torch.where(y == 1, p.log(), (1 - p).log()).mean()

    def backward(self, p: torch.Tensor, y: torch.Tensor):
        return torch.where(y == 1, -1 / p, 1 / (1 - p)) / p.nelement()


def __bce_test(LossTest):
    shape = (200, 2)
    test = LossTest(torch.nn.BCELoss(), BCELoss(), 1e-4)
    inp = torch.rand(shape)
    y = torch.randint(low=0, high=2, size=shape).float()
    test(inp, y)


if __name__ == '__main__':
    from test.loss_test import LossTest

    __bce_test(LossTest)
