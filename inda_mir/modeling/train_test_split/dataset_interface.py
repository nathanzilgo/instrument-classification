import pandas as pd
import pickle
import os


class DatasetInterface:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        labels: list[str],
    ) -> None:
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.labels = labels.copy()

    def get_features_names(self):
        return self.X_train.drop(
            ['filename', 'frame', 'track_id', 'sample_path'], axis=1
        ).columns

    def get_numpy_data(self):
        X_train, y_train = self.get_numpy_train_data()
        X_test, y_test = self.get_numpy_test_data()
        return X_train, X_test, y_train, y_test

    def get_numpy_train_data(self):
        X_train = self.X_train.drop(
            ['filename', 'frame', 'track_id', 'sample_path'], axis=1
        ).to_numpy()
        y_train = self.y_train.to_numpy()
        return X_train, y_train

    def get_numpy_test_data(self):
        X_test = self.X_test.drop(
            ['filename', 'frame', 'track_id', 'sample_path'], axis=1
        ).to_numpy()
        y_test = self.y_test.to_numpy()
        return X_test, y_test

    def train_tracks(self):
        return set(self.X_train['track_id'])

    def test_tracks(self):
        return set(self.X_train['test_id'])

    def save(self, path, filename):
        pickle.dump(self, open(os.path.join(path, filename) + '.data', 'wb'))
