import pandas as pd
import numpy as np
import numpy.typing as npt

from sklearn.model_selection import train_test_split

from typing import Tuple

from inda_mir.modeling.train_test_split.splitter import TrainTestSplitter


class RandomTrainTestSplit(TrainTestSplitter):
    def split(
        self,
        track_metadata: pd.DataFrame,
        track_features: pd.DataFrame,
        train_size: float,
        random_state: int,
    ) -> Tuple[npt.ArrayLike]:

        track_data = pd.merge(
            track_features,
            track_metadata[['track_id', 'sample_path', 'label']],
            left_on='filename',
            right_on='sample_path',
            how='left',
        )

        train_tracks, test_tracks = train_test_split(
            list(set(track_metadata['track_id'])),
            train_size=train_size,
            random_state=random_state,
        )
        train_dataset = track_data[track_data['track_id'].isin(train_tracks)]
        test_dataset = track_data[track_data['track_id'].isin(test_tracks)]

        X_train, y_train = (
            train_dataset.drop(
                ['filename', 'frame', 'track_id', 'sample_path', 'label'],
                axis=1,
            ).to_numpy(),
            train_dataset['label'].to_numpy(),
        )
        X_test, y_test = (
            test_dataset.drop(
                ['filename', 'frame', 'track_id', 'sample_path', 'label'],
                axis=1,
            ).to_numpy(),
            test_dataset['label'].to_numpy(),
        )
        labels = np.array(sorted(list(set(track_data['label']))))

        return X_train, X_test, y_train, y_test, labels
