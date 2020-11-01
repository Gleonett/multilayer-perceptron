from torch import Tensor

from nn.models.base_model import BaseModel


class Sequential(BaseModel):

    def forward(self, input: Tensor):
        pred = input
        for module in self.modules:
            pred = module(pred)
        return pred

    def backward(self, grad: Tensor):
        for module in self.modules[::-1]:
            grad = module.backward(grad)
        return grad
