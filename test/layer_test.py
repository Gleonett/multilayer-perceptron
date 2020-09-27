import torch

import nn
from test import PrintColors, cls_path


class LayerTest(object):

    def __init__(self, gt_l, test_l: nn.BaseLayer, eps=1e-6):
        self.gt_l = gt_l
        self.test_l = test_l
        self.eps = eps

    @staticmethod
    def distance(x, y, dim):
        return torch.norm(x - y, dim=dim).sum()

    @staticmethod
    def insert_middle_substr(substring, char, size=70):
        assert len(char) == 1
        buf_char = '+' if char == '-' * len(char) else '-'
        substring = ('{:^' + str(size) + '}').format(buf_char + substring + buf_char)
        return substring.replace(' ', char).replace(buf_char, ' ')

    def __call__(self, loss, loss_inp, shape):
        """Base test for layers"""
        x_test = torch.randn(shape)
        x_gt = x_test.clone()
        x_gt.requires_grad = True

        gt = self.gt_l(x_gt)
        gt.retain_grad()
        c = loss(gt, loss_inp)
        c.backward()

        test = self.test_l(x_test)
        test_grad = self.test_l.backward(gt.grad)

        name = cls_path(self.test_l)
        print(PrintColors.HEADER + self.insert_middle_substr(name, '#', size=35))

        dist = self.distance(gt, test, len(shape) - 1)
        color = PrintColors.OKGREEN if dist <= self.eps else PrintColors.FAIL
        print(color + "FORWARD  distance\t: {:.6f}".format(dist))

        grad_dist = self.distance(x_gt.grad, test_grad, len(shape) - 1)
        color = PrintColors.OKGREEN if grad_dist <= self.eps else PrintColors.FAIL
        print(color + "BACKWARD distance\t: {:.6f}".format(grad_dist))
        print(PrintColors.ENDC)
