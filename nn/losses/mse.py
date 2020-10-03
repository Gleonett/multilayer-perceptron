import torch

from nn import BaseLoss


class MSELoss(BaseLoss):

    def forward(self, p: torch.Tensor, y: torch.Tensor):
        return ((y - p) ** 2).mean()

    def backward(self, p: torch.Tensor, y: torch.Tensor):
        return -2 * (y - p) / p.nelement()


def __mse_test(LossTest):
    shape = (200, 2)
    test = LossTest(torch.nn.MSELoss(), MSELoss(), 1e-4)
    inp = torch.rand(shape)
    y = torch.rand(shape)
    test(inp, y)


if __name__ == '__main__':
    from test.loss_test import LossTest

    __mse_test(LossTest)
