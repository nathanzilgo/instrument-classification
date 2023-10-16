import pandas as pd

from unittest import mock

from numpy.testing import assert_array_equal

from inda_mir.modeling.train_test_split import DatasetInterface

x_train_data = [
    [4, 7, 8, 5],
    [4, 3, 2, 5],
    [3, 8, 4, 1],
    [0, 4, 5, 0],
    [1, 2, 8, 8],
]

x_test_data = [[3, 8, 1, 7], [2, 7, 6, 2], [4, 1, 5, 7]]

N_SAMPLES_TRAIN = len(x_train_data)
N_SAMPLES_TEST = len(x_test_data)

train_ids = [100 + i for i in range(N_SAMPLES_TRAIN)]
test_ids = [1000 + i for i in range(N_SAMPLES_TEST)]

x_train = pd.DataFrame(x_train_data, columns=list('ABCD'))
x_test = pd.DataFrame(x_test_data, columns=list('ABCD'))

x_train['filename'] = 'filename'
x_train['frame'] = 0
x_train['track_id'] = train_ids

x_test['filename'] = 'filename'
x_test['frame'] = 0
x_test['track_id'] = test_ids

y_train = pd.DataFrame(
    ['drums' for _ in range(N_SAMPLES_TRAIN)], columns=['label']
)
y_test = pd.DataFrame(
    ['drums' for _ in range(N_SAMPLES_TEST)], columns=['label']
)


def test_get_train_tracks():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    assert d.train_tracks() == set(train_ids)


def test_get_test_tracks():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    assert d.test_tracks() == set(test_ids)


def test_get_features_names():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    assert_array_equal(d.get_features_names(), list('ABCD'))


def test_get_numpy_data():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    X_train_data, X_test_data, y_train_data, y_test_data = d.get_numpy_data()
    assert_array_equal(X_train_data, x_train_data)
    assert_array_equal(y_train_data, y_train.to_numpy())
    assert_array_equal(X_test_data, x_test_data)
    assert_array_equal(y_test_data, y_test.to_numpy())


def test_get_numpy_train_data():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    X_train_data, y_train_data = d.get_numpy_train_data()
    assert_array_equal(X_train_data, x_train_data)
    assert_array_equal(y_train_data, y_train.to_numpy())


def test_get_numpy_test_data():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])
    X_test_data, y_test_data = d.get_numpy_test_data()
    assert_array_equal(X_test_data, x_test_data)
    assert_array_equal(y_test_data, y_test.to_numpy())


def test_save():
    d = DatasetInterface(x_train, x_test, y_train, y_test, ['drums'])

    m_op = mock.mock_open()
    m_pickle = mock.mock_open()
    with mock.patch('builtins.open', m_op):
        with mock.patch('pickle.dump', m_pickle):
            d.save('/path/to/save')

    assert mock.call('/path/to/save.data', 'wb') in m_op.mock_calls
    assert m_pickle.call_args[0][0] == d
