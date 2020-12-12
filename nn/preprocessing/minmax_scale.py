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

    def get_params(self) -> (Tensor, Tensor):
        return self.min, self.max

    def set_params(self, min, max):
        self.min = min
        self.max = max
