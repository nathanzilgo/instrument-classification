import pandas as pd

from collections import Counter

from inda_mir.modeling.train_test_split.kfold_splitter import (
    StratifiedKFoldSplitter,
)

track_data = [
    [4, 7, 8, 5],
    [4, 3, 2, 5],
    [3, 8, 4, 1],
    [0, 4, 5, 0],
    [1, 2, 8, 8],
    [3, 8, 1, 7],
    [2, 7, 6, 2],
    [4, 1, 5, 7],
    [4, 1, 5, 7],
    [4, 1, 5, 7],
]

sample_proportion = len(track_data)
track_data = pd.DataFrame(track_data, columns=list('ABCD'))
track_data['track_id'] = list(range(sample_proportion))
track_data['label'] = [
    'drums' if i % 2 == 0 else 'bass' for i in range(sample_proportion)
]


def test_split_number_folds():
    skf = StratifiedKFoldSplitter()
    n_folds = 5
    folds = skf.split(track_data, k=n_folds, random_state=0)
    assert len(folds) == n_folds


def test_split_no_intersection():
    skf = StratifiedKFoldSplitter()
    n_folds = 5
    folds = skf.split(track_data, k=n_folds, random_state=0)

    for fold in folds:
        assert len(fold.X_train) + len(fold.X_test) == sample_proportion
        assert (
            len(
                set(fold.X_train['track_id']).intersection(
                    fold.X_test['track_id']
                )
            )
            == 0
        )


def test_split_keep_class_proportion():
    skf = StratifiedKFoldSplitter()
    n_folds = 5
    folds = skf.split(track_data, k=n_folds, random_state=0)

    c_data = Counter(track_data['label'])

    for fold in folds:

        c_train = Counter(fold.y_train)
        c_test = Counter(fold.y_test)

        for label in fold.labels:
            assert c_train[label] / len(fold.y_train) == c_data[label] / len(
                track_data
            )
            assert c_test[label] / len(fold.y_test) == c_data[label] / len(
                track_data
            )
