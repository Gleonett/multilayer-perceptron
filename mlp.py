import torch
import numpy as np
from pathlib import Path

from utils.data import Dataset
from utils.config import Config
from utils.profiler import Profiler
from utils.pytorch import get_device, to_tensor, accuracy

from general_functions import get_model, get_scaler, get_loss


def prep_data(X, y, device):
    X = to_tensor(X, device, torch.float32)
    y = to_tensor(y, device, torch.uint8)
    return X, y


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


def train(data, model, scale, config, device):
    from nn.optim.sgd import SGD
    from utils.train_history import TrainHistory

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

    criterion = get_loss(config.loss)
    optimizer = SGD(model, **config.sgd_params)

    profiler = Profiler("TRAIN TIME")
    history = TrainHistory(config.epochs, ["loss", "val_loss", "acc", "val_acc"])

    for i in range(1, config.epochs + 1):
        profiler.tick()
        for batch_X, batch_y in batch_iterator(X_train, y_train,
                                               config.batch_size, permute=True):
            output = model(batch_X)

            loss = criterion(output, batch_y)
            grad = criterion.backward(output, batch_y)

            model.backward(grad)

            optimizer.optimise()

        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        test_acc = accuracy(torch.argmax(test_pred, dim=1),
                            torch.argmax(y_test, dim=1))

        pred = model(X_train)
        acc = accuracy(torch.argmax(pred, dim=1),
                       torch.argmax(y_train, dim=1))

        history.update(i, loss, test_loss, acc, test_acc)
        history.print_progress()

        profiler.tock()
    history.visualize()
    print(profiler)


def evaluate(data, model, scale, device):
    X, y, _, _ = data.split(1, shuffle=False)
    X, y = prep_data(X, y, device)

    X = scale(X)

    model.eval()

    profiler = Profiler("INFERENCE TIME")
    profiler.tick()
    pred = model(X)
    profiler.tock()
    print(profiler)

    criterion = get_loss('bce')
    error = criterion(pred, y)
    print("loss: {:.4f}".format(error))

    acc = accuracy(torch.argmin(pred, dim=1), y[:, 0])
    print("accuracy: {:.4f}".format(acc))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default="data/data.csv",
                        help="Path to train dataset. Default: data/data.csv")
    parser.add_argument("--config", type=Path, default="config.yaml",
                        help="Path to config.yaml. Default: config.yaml")
    parser.add_argument("--weights", type=Path, default="data/model.torch",
                        help="Path to model weights save/load. Default: data/model.torch")
    parser.add_argument("-t", action="store_true", help="Train model")
    parser.add_argument("-e", action="store_true", help="Evaluate model")
    args = parser.parse_args()

    assert args.data.exists(), "{} - does not exist".format(args.data)
    assert args.config.exists(), "{} - does not exist".format(args.config)

    config = Config(args.config)
    device = get_device(config.device)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    data = Dataset.read_csv(args.data, header=None)

    scale = get_scaler(config.scale)

    model = get_model(input_shape=30,
                      model_config=config.model)

    if args.t:
        print(model)
        train(data, model, scale, config, device)
        model.save(args.weights, scale)

    if args.e:
        assert args.weights.exists(), "{} - does not exist".format(args.weights)
        print(model)
        model.load(args.weights, scale, device)
        evaluate(data, model, scale, device)
