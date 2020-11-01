import torch
import numpy as np

from nn.models.sequential import Sequential
from nn.init.he import he
from nn.layers.linear import Linear
from nn.layers.activation.relu import ReLU
from nn.layers.activation.softmax import Softmax
from nn.losses.bce import BCELoss
from nn.optim.sgd import sgd_momentum


torch.manual_seed(42)
# np.random.seed(0)


if __name__ == '__main__':
    model = Sequential()

    model.add(Linear(16, 32, he))
    model.add(ReLU())
    model.add(Linear(32, 16, he))
    model.add(ReLU())
    model.add(Linear(16, 2, he))
    model.add(Softmax())
    model.train()

    criterion = BCELoss()

    x = torch.rand((32, 16))
    y = torch.randint(low=0, high=2, size=(32, 2)).float()

    state = {}

    for i in range(30):
        output = model.forward(x)

        loss = criterion(output, y)
        print(loss)
        grad = criterion.backward(output, y)
        # print(grad)

        model.backward(grad)

        params = model.get_params()
        grad_params = model.get_grad_params()

        sgd_momentum(params, grad_params, 1e-4, state)



