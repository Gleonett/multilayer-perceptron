
from .bce import BCELoss
from .mse import MSELoss

losses = {
    "bce": BCELoss,
    "mse": MSELoss,
}
