import torch

from nn.layers.base_layer import BaseLayer


class Linear(BaseLayer):

    grad_W: torch.Tensor
    grad_b: torch.Tensor

    def __init__(self, n_in, n_out, initializer=None):
        super(Linear, self).__init__()

        self.in_out = (n_in, n_out)
        self.W = None
        self.b = None
        if initializer:
            self.W, self.b = initializer(n_in, n_out)
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
