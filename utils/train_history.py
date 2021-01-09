import numpy as np
from matplotlib import pyplot as plt


class TrainHistory(object):

    def __init__(self, epochs: int, metrics_names: [str]):
        self.epochs = epochs
        self.names = metrics_names
        self.values = dict()

    def update(self, epoch, *args):
        assert len(args) == len(self.names)
        self.values[epoch] = list(args)

    def print_progress(self):
        last_epoch = sorted(self.values.keys())[-1]
        form_e = len(str(self.epochs))
        epoch = "Epoch: {: >{}} / {: >{}}".format(last_epoch, form_e, self.epochs, form_e)
        metrics = []
        for name, val in zip(self.names, self.values[last_epoch]):
            metrics.append("{} - {:.4f}".format(name, val))
        print(' | '.join([epoch, *metrics]))

    def visualize(self):
        epochs = np.array(list(self.values.keys()))
        values = np.array(list(self.values.values()))

        _, ax = plt.subplots()

        for idx, label in enumerate(self.names):
            ax.plot(epochs, values[:, idx], label=label)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metrics')
        # ax.set_title('Logistic Regression, batch size: {}'
        #              .format(model.batch_size))
        ax.legend(loc="upper right")
        plt.show()
