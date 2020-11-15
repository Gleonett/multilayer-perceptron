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

    def to_dictionary(self) -> {str: Tensor}:
        """
        Put scale parameters to dictionary
        :return: None
        """
        return {"mean": self.mean, "std": self.std}

    def from_dictionary(self,
                        dictionary: {str: Tensor},
                        device: torch.device,
                        dtype: torch.dtype):
        """
        Load scale parameters from dictionary
        :return: None
        """
        self.mean = dictionary["mean"].to(device, dtype)
        self.std = dictionary["std"].to(device, dtype)
