import torch
from torch import Tensor

from nn.layers.base_layer import BaseLayer


class Softmax(BaseLayer):

    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim
        self.forward_output = None

    def update_output(self, input: Tensor):
        x_exp = (input - input.max(dim=1, keepdim=True).values).exp()
        self.forward_output = x_exp / x_exp.sum(dim=self.dim, keepdim=True)
        return self.forward_output

    def update_grad(self, grad: Tensor):
        p = self.forward_output
        dA = (grad * p).sum(dim=self.dim, keepdim=True)
        return p * (grad - dA)

def __softmax_test(LayerTest):
    shape = (500, 2)
    dim = -1
    test = LayerTest(torch.nn.Softmax(dim=dim), Softmax(dim), 1e-4)
    test(torch.nn.functional.binary_cross_entropy,
         torch.randint(low=0, high=2, size=shape).float(),
         shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __softmax_test(LayerTest)
