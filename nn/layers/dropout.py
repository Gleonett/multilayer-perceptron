import torch

from nn.layers.base_layer import BaseLayer


class Dropout(BaseLayer):

    def __init__(self, p: float = 0.2):
        assert 0 <= p <= 1

        super().__init__()
        self.keep_probability = 1 - p
        self.mask = None

        self.scale = (1 / (self.keep_probability + 1e-7))

    def update_output(self, input):
        if self.train_mode:
            self.mask = torch.rand(input.shape, device=self.device) < self.keep_probability
            input *= self.mask
        return input * self.scale

    def update_grad(self, grad):
        return grad * self.mask


def __dropout_test(LayerTest):
    # Works only with 1 and 0 due to random
    test = LayerTest(torch.nn.Dropout(p=1), Dropout(p=1))
    shape = (500, 32)
    test(torch.nn.functional.mse_loss, torch.randn(shape), shape)


if __name__ == '__main__':
    from test.layer_test import LayerTest

    __dropout_test(LayerTest)
