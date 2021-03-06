import torch
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

    def to_dict(self):
        d = dict()
        for i, module in enumerate(self.modules):
            d[i] = module.get_params()
        return d

    def save(self, path: str, scale: None):
        d = dict()
        d['model'] = self.to_dict()
        if scale is not None:
            d['scale'] = scale.get_params()
        torch.save(d, path)

    def load(self, path: str, scale: None, device: torch.device = "cpu"):
        d = torch.load(path)
        for params, module in zip(d['model'].values(), self.modules):
            module.set_params(*[param.to(device) for param in params])
        if scale is not None:
            scale.set_params(*[param.to(device) for param in d['scale']])

    def to_device(self, device: torch.device):
        for m in self.modules:
            m.to_device(device)

    def __str__(self):
        buf = "Model - " + type(self).__name__ + ":\n"
        for module in self.modules:
            buf += str(module) + "\n"
        return buf
