import torch

import nn


class Softmax(nn.BaseLayer):

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x: torch.Tensor):
        x_exp = x.exp()
        return x_exp / x_exp.sum(dim=0)

    def backward(self, grad):
        # not implemented
        return grad


def __softmax_test(LayerTest):
    test = LayerTest(torch.nn.Softmax(dim=0), Softmax())
    shape = (2,)
    test(torch.nn.functional.binary_cross_entropy,
         torch.randint(low=0, high=2, size=shape).float(),
         shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __softmax_test(LayerTest)
