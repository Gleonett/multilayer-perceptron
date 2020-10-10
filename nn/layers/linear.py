import torch

from nn import BaseLayer


class Linear(BaseLayer):

    def __init__(self, n_in, n_out, initializer=None):
        super(Linear, self).__init__()

        self.W = None
        self.b = None
        self.grad_W = None
        self.grad_b = None
        if initializer:
            self.W, self.b = initializer(n_in, n_out)
            self.grad_W = torch.zeros_like(self.W)
            self.grad_b = torch.zeros_like(self.b)

    def update_output(self, input):
        return input @ self.W + self.b

    def update_grad(self, grad):
        new_grad = grad @ self.W.t()
        self.grad_W = grad.t() @ self.input
        self.grad_b = grad.sum(dim=0)
        return new_grad

    def zero_grad(self):
        self.grad_W.fill_(0)
        self.grad_b.fill_(0)

    def get_parameters(self):
        return self.W, self.b

    def get_grad_parameters(self):
        return self.gradW, self.gradb

    def __str__(self):
        shape = self.W.shape
        return 'Linear %d->%d' % (shape[0], shape[1])
