import numpy as np
import pandas as pd
from abc import ABC


class Dataset(pd.DataFrame):

    @staticmethod
    def read_csv(csv_path: str, **kwargs):
        """
        Read .csv file
        :param csv_path: path to .csv file
        :return: Dataset
        """
        return Dataset(pd.read_csv(csv_path, **kwargs))

    def split(self, train_part: float, shuffle: bool):
        assert 0 <= train_part <= 1, "train_part mast be: 0 <= train_part <= 1"

        if shuffle:
            shuffled = np.random.permutation(self)
        else:
            shuffled = self.to_numpy()

        m_idxs = np.argwhere(shuffled[:, 1] == 'M').squeeze()
        b_idxs = np.argwhere(shuffled[:, 1] == 'B').squeeze()

        m_idxs_train = m_idxs[:int(m_idxs.shape[0] * train_part)]
        b_idxs_train = b_idxs[:int(b_idxs.shape[0] * train_part)]
        idxs_train = np.sort(np.hstack([m_idxs_train, b_idxs_train]))

        m_idxs_test = m_idxs[int(m_idxs.shape[0] * train_part):]
        b_idxs_test = b_idxs[int(b_idxs.shape[0] * train_part):]
        idxs_test = np.sort(np.hstack([m_idxs_test, b_idxs_test]))

        X_train = shuffled[idxs_train, 2:].astype(np.float32)
        y_train = (shuffled[idxs_train, 1] == 'M')
        y_train = np.vstack([y_train, ~y_train]).T.astype(np.uint8)

        X_test = shuffled[idxs_test, 2:].astype(np.float32)
        y_test = (shuffled[idxs_test, 1] == 'M')
        y_test = np.vstack([y_test, ~y_test]).T.astype(np.uint8)

        return X_train, y_train, X_test, y_test
