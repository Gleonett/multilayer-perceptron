from torch import Tensor

class BaseLoss(object):

    def update_output(self, input: Tensor, target: Tensor):
        raise NotImplementedError

    def update_grad(self, input: Tensor, target: Tensor):
        raise NotImplementedError

    def forward(self, input: Tensor, target: Tensor):
        return self.update_output(input, target).mean()

    def backward(self, input: Tensor, target: Tensor):
        otp = self.update_grad(input, target)
        return otp / input.shape[0]

    def __call__(self, input: Tensor, target: Tensor):
        return self.forward(input, target)
