import torch
from torch import Tensor

from nn.preprocessing.minmax_scale import MinMaxScale


class StandardScale(object):
    """
    This method is widely used for normalization in
    many machine learning algorithms
    x' = (x - mean(x)) / std(x)
    """
    mean: Tensor
    std: Tensor

    def fit(self, x: Tensor):
        """
        Obtain values for future scaling
        :param x: tensor of shape (num_samples, num_features)
        :return: None
        """
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale given set values
        :param x: tensor of shape (num_samples, num_features)
        :return: tensor of shape (num_samples, num_features)
        """
        return (x - self.mean) / self.std

    def get_params(self) -> (Tensor, Tensor):
        return self.mean, self.std

    def set_params(self, mean: Tensor, std: Tensor):
        self.mean = mean
        self.std = std
