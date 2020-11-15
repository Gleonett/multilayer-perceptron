import numpy as np
import pandas as pd
from abc import ABC


class Dataset(pd.DataFrame, ABC):

    @staticmethod
    def read_csv(csv_path: str, **kwargs):
        """
        Read .csv file
        :param csv_path: path to .csv file
        :return: Dataset
        """
        return Dataset(pd.read_csv(csv_path, **kwargs))

    def split(self, train_part: float):
        assert 0 <= train_part <= 1

        permuted = np.random.permutation(self)

        train_num = int(self.shape[0] * train_part)
        X_train = permuted[:train_num, 2:].astype(np.float32)
        y_train = (permuted[:train_num, 1] == 'M')
        y_train = np.vstack([y_train, ~y_train]).T.astype(np.uint8)

        X_test = permuted[train_num:, 2:].astype(np.float32)
        y_test = (permuted[train_num:, 1] == 'M')
        y_test = np.vstack([y_test, ~y_test]).T.astype(np.uint8)

        return X_train, y_train, X_test, y_test


