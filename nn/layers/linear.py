import torch

from nn.layers.base_layer import BaseLayer
from nn.init import initializers


class Linear(BaseLayer):

    grad_W: torch.Tensor
    grad_b: torch.Tensor

    def __init__(self, n_in, n_out, initializer: str = 'none'):
        super(Linear, self).__init__()

        self.in_out = (n_in, n_out)
        self.W = None
        self.b = None

        if initializer != 'none':
            assert initializer in initializers,\
                "No such initializer: '{}'".format(initializer)

            init = initializers[initializer]
            self.W, self.b = init(n_in, n_out)
            self.grad_W = torch.zeros_like(self.W)
            self.grad_b = torch.zeros_like(self.b)

    def update_output(self, input):
        return input @ self.W.T + self.b

    def update_grad(self, grad):
        new_grad = grad @ self.W
        self.grad_W = grad.T @ self.input
        self.grad_b = grad.sum(dim=0)
        return new_grad

    def zero_grad(self):
        self.grad_W.fill_(0)
        self.grad_b.fill_(0)

    def get_params(self):
        return self.W, self.b

    def get_grad_params(self):
        return self.grad_W, self.grad_b

    def __str__(self):
        return '{} {}->{}'.format(type(self).__name__, *self.in_out)


def __linear_test(LayerTest):
    shape = (500, 32)
    in_out = (shape[1], 16)

    torch_method = torch.nn.Linear(*in_out)

    my_method = Linear(*in_out)
    my_method.W = torch_method.weight.detach().clone()
    my_method.b = torch_method.bias.detach().clone()

    test = LayerTest(torch_method, my_method)
    test(torch.nn.functional.mse_loss,
         torch.rand((shape[0], in_out[1])),
         shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __linear_test(LayerTest)

