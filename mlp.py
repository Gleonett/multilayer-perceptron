import torch
import numpy as np

from utils.data import Dataset
from utils.config import Config
from utils.profiler import Profiler
from utils.train_history import TrainHistory
from utils.pytorch import get_device, to_tensor, accuracy
from nn.preprocessing import scalers

from nn.init.he import he
from nn.losses.bce import BCELoss
from nn.losses.mse import MSELoss
from nn.layers.linear import Linear
from nn.layers.dropout import Dropout
from nn.layers.activation.relu import ReLU
from nn.layers.activation.softmax import Softmax
from nn.optim.sgd import SGD
from nn.models.sequential import Sequential


def prep_data(X, y, device):
    X = to_tensor(X, device, torch.float32)
    y = to_tensor(y, device, torch.uint8)
    return X, y


def get_model(input_shape):
    model = Sequential()
    model.add(Linear(input_shape, 32, he))
    model.add(ReLU())
    # model.add(Dropout(p=0.2))
    model.add(Linear(32, 32, he))
    model.add(ReLU())
    # model.add(Dropout(p=0.2))
    model.add(Linear(32, 16, he))
    model.add(ReLU())
    model.add(Linear(16, 2, he))
    model.add(Softmax())
    return model


def batch_iterator(X, y, batch_size=None, permute=True):
    if permute:
        permutation = torch.randperm(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    if batch_size is None:
        batch_size = X.shape[0]
    for i in range(0, X.shape[0], batch_size):
        begin, end = i, min(i + batch_size, X.shape[0])
        yield X[begin:end], y[begin:end]


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
    # criterion = MSELoss()
    optimizer = SGD(model, lr=config.lr, momentum=0)
    results = []

    profiler = Profiler("TRAIN TIME")
    history = TrainHistory(config.epochs, ["loss", "val_loss", "acc", "val_acc"])
    for i in range(1, config.epochs):
        for batch_X, batch_y in batch_iterator(X_train, y_train, config.batch_size):
            profiler.tick()
            output = model(batch_X)

            loss = criterion(output, batch_y)
            grad = criterion.backward(output, batch_y)

            model.backward(grad)

            optimizer.optimise()
            # model.zero_grad()

            profiler.tock()

        # if i % 2 == 1:
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        test_acc = accuracy(test_pred[:, 0] > 0.5, y_test[:, 0])

        pred = model(X_train)
        loss = criterion(pred, y_train)
        acc = accuracy(pred[:, 0] > 0.5, y_train[:, 0])

        history.update(i, loss, test_loss, acc, test_acc)
        history.print_progress()
    history.visualize()

    print(profiler)
    print("========== RESULT ==========")

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

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    data = Dataset.read_csv(args.data, header=None)

    scale = scalers[config.scale]()

    model = get_model(input_shape=30)
    print(model)

    train(data, model, scale, config)

    evaluate(data, model, scale, config)
