import torch
import numpy as np

from utils.data import Dataset
from utils.config import Config
from utils.profiler import Profiler
from utils.pytorch import get_device, to_tensor, accuracy
from nn.preprocessing import scalers

from nn.init.he import he
from nn.losses.bce import BCELoss
from nn.layers.linear import Linear
from nn.layers.activation.relu import ReLU
from nn.layers.activation.softmax import Softmax
from nn.optim.sgd import sgd_momentum
from nn.models.sequential import Sequential


def prep_data(X, y, device):
    X = to_tensor(X, device, torch.float32)
    y = to_tensor(y, device, torch.uint8)
    return X, y


def get_model(input_shape):
    model = Sequential()
    model.add(Linear(input_shape, 16, he))
    model.add(ReLU())
    model.add(Linear(16, 16, he))
    model.add(ReLU())
    model.add(Linear(16, 2, he))
    model.add(Softmax())
    return model


def train(data, model, scale, config):
    device = get_device(config.device)
    X_train, y_train, X_test, y_test = data.split(config.train_part, shuffle=True)
    X_train, y_train = prep_data(X_train, y_train, device)
    X_test, y_test = prep_data(X_test, y_test, device)

    print("first label in train part: {:.2f}%"
          .format(y_train[:, 0].sum().float() / y_train.shape[0] * 100))
    print("first label in test part: {:.2f}%"
          .format(y_test[:, 0].sum().float() / y_test.shape[0] * 100))

    scale.fit(X_train)
    X_train = scale(X_train)
    X_test = scale(X_test)


    criterion = BCELoss()
    state = {}
    results = []

    profiler = Profiler("TRAIN TIME")
    for i in range(config.epochs):
        profiler.tick()
        output = model(X_train)

        loss = criterion(output, y_train)
        grad = criterion.backward(output, y_train)

        model.backward(grad)

        params = model.get_params()
        grad_params = model.get_grad_params()

        sgd_momentum(params, grad_params, config.lr, state)
        if i % 10 == 0:
            pred = model(X_test)
            val_loss = criterion(pred, y_test)
            error = accuracy(torch.argmin(pred, dim=1), y_test[:, 0])
            results.append([error, loss, val_loss, i])
        profiler.tock()
    print(profiler)
    print("========== RESULT ==========")
    results = np.array(results)
    idxs = np.argsort(results[:, 0])[::-1]
    results = results[idxs]

    for result in results[:5]:
        print("epoch:", int(result[3]))
        print("loss: {:.6f}".format(result[1]))
        print("val_loss: {:.6f}".format(result[2]))
        print("accuracy: {:.6f}".format(result[0]))
        print('-' * 25)


def evaluate(data, model, scale, config):
    device = get_device(config.device)
    X, y, _, _ = data.split(1, shuffle=False)
    X, y = prep_data(X, y, device)

    X = scale(X)

    model.eval()

    profiler = Profiler("INFERENCE TIME")
    profiler.tick()
    pred = model(X)
    profiler.tock()
    print(profiler)

    criterion = BCELoss()
    error = criterion(pred, y)
    print("BCE: {:.4f}".format(error))

    acc = accuracy(torch.argmin(pred, dim=1), y[:, 0])
    print("accuracy: {:.4f}".format(acc))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data.csv",
                        help="Path to train dataset")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to train dataset")
    args = parser.parse_args()

    config = Config(args.config)

    if config.seed:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    data = Dataset.read_csv(args.data, header=None)

    scale = scalers[config.scale]()

    model = get_model(input_shape=30)
    print(model)

    train(data, model, scale, config)

    evaluate(data, model, scale, config)

