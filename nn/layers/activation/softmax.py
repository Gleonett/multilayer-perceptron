import torch

import nn


class Softmax(nn.BaseLayer):

    def __init__(self, dim=1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        x_exp = x.exp()
        return x_exp / x_exp.sum(dim=self.dim, keepdim=True)

    def backward(self, grad):
        inp, = self.saved_for_backward
        p = self.forward(inp)
        dA = torch.sum(grad * p, dim=self.dim, keepdim=True)
        return p * (grad - dA)


def __softmax_test(LayerTest):
    shape = (500, 32)
    dim = len(shape) - 1
    test = LayerTest(torch.nn.Softmax(dim=dim), Softmax(dim), 1e-4)
    test(torch.nn.functional.binary_cross_entropy,
         torch.randint(low=0, high=2, size=shape).float(),
         shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __softmax_test(LayerTest)
