import pandas as pd

from inda_mir.modeling.train_test_split.splitter import TrainTestSplitter


class RandomTrainTestSplit(TrainTestSplitter):
    def split(
        self,
        track_metadata: pd.DataFrame,
        track_features: pd.DataFrame,
        frac: float,
        random_state: int,
    ) -> tuple:

        track_data = pd.merge(
            track_features,
            track_metadata[['track_id', 'sampled_path', 'label']],
            left_on='filename',
            right_on='sampled_path',
            how='left',
        )

        train_tracks = track_metadata.sample(
            frac=frac, random_state=random_state
        )['track_id'].to_list()
        train_index = track_data['track_id'].isin(train_tracks)
        train_dataset = track_data[train_index]
        test_dataset = track_data[~train_index]

        X_train, y_train = (
            train_dataset.drop(['label'], axis=1).to_numpy(),
            train_dataset['label'].to_numpy(),
        )
        X_test, y_test = (
            test_dataset.drop(['label'], axis=1).to_numpy(),
            test_dataset['label'].to_numpy(),
        )

        return X_train, X_test, y_train, y_test
