from nn.losses.base_loss import BaseLoss
from nn.models.base_model import BaseModel


def get_model(input_shape: int, model_config: dict) -> BaseModel:
    from nn.layers.linear import Linear
    from nn.layers.dropout import Dropout
    from nn.layers.activation.relu import ReLU
    from nn.layers.activation.softmax import Softmax
    from nn.models.sequential import Sequential

    model = Sequential()
    model.add(Linear(n_in=input_shape, **model_config['linear_1']))
    model.add(ReLU())
    model.add(Dropout(**model_config['dropout_1']))
    model.add(Linear(**model_config['linear_2']))
    model.add(ReLU())
    model.add(Dropout(**model_config['dropout_2']))
    model.add(Linear(**model_config['linear_3']))
    model.add(ReLU())
    model.add(Dropout(**model_config['dropout_3']))
    model.add(Linear(n_out=2, **model_config['linear_4']))
    model.add(Softmax())
    return model


def get_scaler(key: str):
    from nn.preprocessing import scalers
    assert key in scalers
    return scalers[key]()


def get_loss(key: str) -> BaseLoss:
    from nn.losses import losses
    assert key in losses
    return losses[key]()
