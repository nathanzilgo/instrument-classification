import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from typing import List

from inda_mir.modeling.train_test_split.splitter import (
    TrainTestSplitter,
    DatasetInterface,
)


class StratifiedKFoldSplitter(TrainTestSplitter):
    def split(
        self, track_data: pd.DataFrame, **kwargs
    ) -> List[DatasetInterface]:
        return self._split(track_data, **kwargs)

    def _split(
        self,
        track_data: pd.DataFrame,
        k: int,
        random_state: int,
    ) -> List[DatasetInterface]:

        track_labels = track_data[['track_id', 'label']].drop_duplicates()
        folds = StratifiedKFold(
            n_splits=k, random_state=random_state, shuffle=True
        ).split(
            track_labels['track_id'].to_numpy(),
            track_labels['label'].to_numpy(),
        )

        features_folds = []
        for train, test in folds:
            train_tracks = track_labels['track_id'].iloc[train]
            test_tracks = track_labels['track_id'].iloc[test]

            train_dataset = track_data[
                track_data['track_id'].isin(train_tracks)
            ]
            test_dataset = track_data[track_data['track_id'].isin(test_tracks)]

            X_train, y_train = (
                train_dataset.drop(
                    ['label'],
                    axis=1,
                ),
                train_dataset['label'],
            )
            X_test, y_test = (
                test_dataset.drop(
                    ['label'],
                    axis=1,
                ),
                test_dataset['label'],
            )
            labels = np.array(sorted(list(set(track_data['label']))))
            features_folds.append(
                DatasetInterface(X_train, X_test, y_train, y_test, labels)
            )

        return features_folds
