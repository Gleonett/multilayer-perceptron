
from test.base import BaseTest


class LossTest(BaseTest):

    def test(self, y_pred, y_gt):
        y_pred_gt = y_pred.clone()
        y_pred_gt.requires_grad = True
        gt = self.gt_method(y_pred_gt, y_gt)
        gt.backward()

        test = self.test_method(y_pred, y_gt)
        test_grad = self.test_method.backward(y_pred, y_gt)

        dist = self.distance(gt, test, -1)
        grad_dist = self.distance(y_pred_gt.grad, test_grad, -1)
        return dist, grad_dist
