import torch
import numpy as np

from utils.data import Dataset
from utils.pytorch import get_device, to_tensor
from nn.preprocessing.minmax_scale import MinMaxScale
from nn.preprocessing.standard_scale import StandardScale

from nn.init.he import he
from nn.init.xavier import xavier
from nn.init.standard import standard
from nn.losses.bce import BCELoss
from nn.layers.linear import Linear
from nn.optim.sgd import sgd_momentum
from nn.layers.activation.relu import ReLU
from nn.models.sequential import Sequential
from nn.layers.activation.softmax import Softmax


seed = 21
torch.manual_seed(seed)
np.random.seed(seed)


def prep_data(path, columns, device):

    data = Dataset.read_csv(path, header=None)
    # data = Dataset(data.replace(0, None).fillna(method='ffill'))

    X_train, y_train, X_test, y_test = data.split(train_part=0.75)

    print(np.squeeze(np.argwhere(~(X_train == 0).any(axis=0))))

    X_train = X_train[:, columns]
    X_test = X_test[:, columns]

    mask = ~(X_train == 0).any(axis=1)
    print((~mask).sum())
    # X_train = X_train[mask]
    # y_train = y_train[mask]

    print("M train: {:.2f}%".format(y_train[:, 0].sum() / y_train.shape[0] * 100))
    print("M test: {:.2f}%".format(y_test[:, 0].sum() / y_test.shape[0] * 100))

    X_train = to_tensor(X_train, device, torch.float32)
    y_train = to_tensor(y_train, device, torch.uint8)

    X_test = to_tensor(X_test, device, torch.float32)
    y_test = to_tensor(y_test, device, torch.uint8)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    path = "data/data.csv"
    device = "cpu"
    device = get_device(device)
    # columns = [0, 1, 6, 9, 10, 15, 16, 17, 19, 24, 26, 28, 29]
    # columns = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29]
    columns = range(0, 30)

    X_train, y_train, X_test, y_test = prep_data(path, columns, device)
    # scale = MinMaxScale()
    scale = StandardScale()
    scale.fit(X_train)

    X_train = scale(X_train)
    X_test = scale(X_test)

    input_shape = X_train.shape[1]
    print("input shape:", input_shape)

    model = Sequential()
    model.add(Linear(input_shape, 32, he))
    model.add(ReLU())
    model.add(Linear(32, 16, he))
    model.add(ReLU())
    model.add(Linear(16, 2, he))
    model.add(Softmax())
    model.train()

    criterion = BCELoss()

    state = {}

    pred = model.forward(X_test)
    error = criterion(pred, y_test)
    print("result", error)

    results = []

    for i in range(1000):
        output = model.forward(X_train)

        loss = criterion(output, y_train)
        grad = criterion.backward(output, y_train)
        # print(grad)

        model.backward(grad)

        params = model.get_params()
        grad_params = model.get_grad_params()

        if i % 10 == 0:
            pred = model.forward(X_test)
            error = 1 - criterion(pred, y_test)
            # print("+" * 20)
            # print("iteration:", i)
            # print("loss: {:.6f}".format(loss))
            # print("eval: {:.6f}".format(error))
            results.append([error, loss, i])

        sgd_momentum(params, grad_params, 1e-4, state)

    print("========== RESULT ==========")
    results = np.array(results)
    idxs = np.argsort(results[:, 0])[::-1]
    results = results[idxs]

    for result in results[:5]:
        print("epoch:", int(result[2]))
        print("loss: {:.6f}".format(result[1]))
        print("accuracy: {:.6f}".format(result[0]))
        print('-' * 25)

    X = torch.cat([X_train, X_test])
    y = torch.cat([y_train, y_test])

    pred = model.forward(X)
    error = 1 - criterion(pred, y)
    print(error)
