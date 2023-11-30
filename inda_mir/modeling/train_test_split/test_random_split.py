import pandas as pd

from numpy.testing import assert_array_equal

from inda_mir.modeling.train_test_split.random_splitter import (
    RandomTrainTestSplit,
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


def test_split_train_size_50():
    r = RandomTrainTestSplit()
    train_size = 0.5
    X_train_data, X_test_data, y_train_data, y_test_data, labels = r._split(
        track_data, train_size=train_size, random_state=0
    )
    assert len(X_train_data) == train_size * sample_proportion
    assert len(X_test_data) == int((1 - train_size) * sample_proportion)
    assert len(y_train_data) == train_size * sample_proportion
    assert len(y_test_data) == int((1 - train_size) * sample_proportion)
    assert (
        len(
            set(X_train_data['track_id']).intersection(X_test_data['track_id'])
        )
        == 0
    )
    assert_array_equal(labels, ['bass', 'drums'])


def test_split_train_size_70():
    r = RandomTrainTestSplit()
    train_size = 0.7
    X_train_data, X_test_data, y_train_data, y_test_data, labels = r._split(
        track_data, train_size=train_size, random_state=0
    )
    assert len(X_train_data) == train_size * sample_proportion
    assert len(X_test_data) == int((1 - train_size) * sample_proportion)
    assert len(y_train_data) == train_size * sample_proportion
    assert len(y_test_data) == int((1 - train_size) * sample_proportion)
    assert (
        len(
            set(X_train_data['track_id']).intersection(X_test_data['track_id'])
        )
        == 0
    )
    assert_array_equal(labels, ['bass', 'drums'])
