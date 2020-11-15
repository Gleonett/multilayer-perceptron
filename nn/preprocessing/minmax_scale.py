import torch
from torch import Tensor


class MinMaxScale(object):
    """
    It is the simplest method and consists in rescaling the
    range of features to scale the range in [0, 1].
    x' = (x - min(x)) / (max(x) - min(x))
    """
    min: Tensor
    max: Tensor

    def fit(self, x: Tensor):
        """
        Obtain values for future scaling
        :param x: tensor of shape (num_samples, num_features)
        :return: None
        """
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale given set values
        :param x: tensor of shape (num_samples, num_features)
        :return: tensor of shape (num_samples, num_features)
        """
        return (x - self.min) / (self.max - self.min)

    def to_dictionary(self) -> {str: Tensor}:
        """
        Put scale parameters to dictionary
        :return: None
        """
        return {"min": self.min, "max": self.max}

    def from_dictionary(self,
                        dictionary: {str: Tensor},
                        device: torch.device,
                        dtype: torch.dtype):
        """
        Load scale parameters from dictionary
        :return: None
        """
        self.min = dictionary["min"].to(device, dtype)
        self.max = dictionary["max"].to(device, dtype)