import torch
from torch import Tensor

from nn.layers.base_layer import BaseLayer


class ReLU(BaseLayer):

    def update_output(self, input: Tensor):
        return input.clamp(min=0)

    def update_grad(self, grad: Tensor):
        input = self.input
        grad = grad.clone()
        grad[input < 0] = 0
        return grad


def __relu_test(LayerTest):
    test = LayerTest(torch.nn.ReLU(), ReLU())
    shape = (500, 32)
    test(torch.nn.functional.mse_loss, torch.randn(shape), shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __relu_test(LayerTest)
