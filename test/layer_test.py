import torch

from test.base import BaseTest


class LayerTest(BaseTest):

    def test(self, loss, loss_inp, shape):
        x_test = torch.randn(shape)
        x_gt = x_test.clone()
        x_gt.requires_grad = True

        gt = self.gt_method(x_gt)
        gt.retain_grad()
        if loss_inp is None:
            c = loss(gt)
        else:
            c = loss(gt, loss_inp)
        c.backward()

        test = self.test_method(x_test)
        test_grad = self.test_method.backward(gt.grad)

        dist = self.distance(gt, test, -1)
        grad_dist = self.distance(x_gt.grad, test_grad, -1)
        return dist, grad_dist
