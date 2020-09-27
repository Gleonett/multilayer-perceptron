import torch

import nn


class ReLU(nn.BaseLayer):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0)

    def backward(self, grad):
        x, = self.saved_for_backward
        grad = grad.clone()
        grad[x < 0] = 0
        return grad


def __relu_test(LayerTest):
    test = LayerTest(torch.nn.ReLU(), ReLU())
    shape = (500,32)
    test(torch.nn.functional.mse_loss, torch.randn(shape), shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __relu_test(LayerTest)
