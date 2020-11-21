from torch import Tensor


class BaseLayer(object):

    def __init__(self):
        self.train_mode = True
        self.input = None

    def update_output(self, input: Tensor):
        raise NotImplementedError

    def update_grad(self, grad: Tensor):
        raise NotImplementedError

    def forward(self, input: Tensor):
        if self.train_mode:
            self.save_for_backward(input)
        return self.update_output(input)

    def backward(self, grad: Tensor):
        return self.update_grad(grad)

    def __call__(self, input: Tensor):
        return self.forward(input)

    def zero_grad(self):
        pass

    def get_params(self):
        return []

    def get_grad_params(self):
        return []

    def save_for_backward(self, input: Tensor):
        self.input = input.clone()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def __str__(self):
        return type(self).__name__
