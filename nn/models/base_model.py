from torch import Tensor

from nn.layers.base_layer import BaseLayer


class BaseModel(object):

    def __init__(self):
        self.train_mode = False
        self.modules: [BaseLayer] = []

    def forward(self, input: Tensor):
        raise NotImplementedError

    def backward(self, grad: Tensor):
        raise NotImplementedError

    def __call__(self, input: Tensor):
        return self.forward(input)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def get_params(self):
        return [m.get_params() for m in self.modules]

    def get_grad_params(self):
        return [m.get_grad_params() for m in self.modules]

    def add(self, module):
        self.modules.append(module)

    def train(self):
        self.train_mode = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.train_mode = False
        for module in self.modules:
            module.eval()
