import torch

import nn
from test import PrintColors, cls_path


class LayerTest(object):

    def __init__(self, gt_l, test_l: nn.BaseLayer):
        self.gt_l = gt_l
        self.test_l = test_l

    def __call__(self, x=None, shape=(1000,)):
        """Base test for layers"""
        x_test = torch.randn(shape)
        x_gt = x_test.clone()
        x_gt.requires_grad = True

        gt = self.gt_l(x_gt)
        gt.retain_grad()
        c = torch.nn.functional.mse_loss(gt, torch.randn(shape))
        c.backward()

        test = self.test_l(x_test)
        test_grad = self.test_l.backward(gt.grad)

        name = cls_path(self.test_l)
        if gt.isclose(test).all():
            print(PrintColors.OKGREEN + name + "\tFORWARD \t: OK")
        else:
            print(PrintColors.FAIL + name + "\tFORWARD \t: FAIL")

        if x_gt.grad.isclose(test_grad).all():
            print(PrintColors.OKGREEN + name + "\tBACKWARD\t: OK")
        else:
            print(PrintColors.FAIL + name + "\tBACKWARD\t: FAIL")
