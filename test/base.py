import torch

from test import PrintColors, cls_path


class BaseTest(object):

    def __init__(self, gt_method, test_method, eps=1e-6):
        self.gt_method = gt_method
        self.test_method = test_method
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

    def print_header(self):
        name = cls_path(self.test_method)
        print(PrintColors.HEADER + self.insert_middle_substr(name, '#', size=35))

    def print_result(self, dist, mode: str):
        color = PrintColors.OKGREEN if dist <= self.eps else PrintColors.FAIL
        print(color + "{}  distance\t: {:.6f}".format(mode, dist) + PrintColors.ENDC)

    def test(self, *args, **kwargs) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        dist_forward, dist_backward = self.test(*args, **kwargs)
        self.print_header()
        self.print_result(dist_forward, "FORWARD ")
        self.print_result(dist_backward, "BACKWARD")
