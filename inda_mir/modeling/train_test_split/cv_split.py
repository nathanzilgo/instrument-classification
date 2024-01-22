import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class CVSplitter:
    """
    Inda Mir Cross-Validation Splitter.
    """

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame, y: pd.DataFrame, groups: int = None):
        """
        Split the dataset into multiple train-test splits.

        Parameters:
            X : pd.DataFrame
                The input feature DataFrame.

            y : pd.DataFrame
                The target variable DataFrame.

            groups : int, optional
                The number of groups to split the dataset into. Default is None.

        Yields:
            train : list
                The indices of the training samples.

            test : list
                The indices of the testing samples.
        """

        for _ in range(self.n_splits):

            aux = X[['track_id']].copy()
            aux['label'] = y
            aux.drop_duplicates(inplace=True)

            train_tracks, test_tracks = train_test_split(
                aux['track_id'], stratify=aux['label'], train_size=0.7
            )

            train = list(np.where(X['track_id'].isin(train_tracks))[0])
            test = list(np.where(X['track_id'].isin(test_tracks))[0])
            yield train, test

    def get_n_splits(self, X: pd.DataFrame, y: pd.DataFrame, groups: int):
        return self.n_splits
