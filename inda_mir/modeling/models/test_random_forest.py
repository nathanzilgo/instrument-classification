import pytest

from typing import List
from unittest import mock

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from numpy.testing import assert_equal
from numpy import asarray

from inda_mir.modeling.models import RandomForestClassifier


def _build_model_mock(rf_mock: mock.Mock, pred: List, n_features: int = 2):
    rf_mock_fitted = mock.Mock()
    rf_mock_fitted.predict_proba.return_value = asarray(pred)
    rf_mock_fitted.feature_importances_ = [0 for _ in range(n_features)]
    rf_mock.return_value = rf_mock_fitted


def _build_encoder_mock(le_mock: mock.Mock):
    le_mock_fitted = mock.Mock()
    classes = ['bass', 'drums', 'vocals']
    le_mock_fitted.inverse_transform = lambda x: [classes[i] for i in x]
    le_mock_fitted.transform = lambda c: [classes.index(i) for i in c]
    le_mock.return_value = le_mock_fitted


def test_init():
    rf = RandomForestClassifier()
    assert rf.name == 'RandomForest'
    assert rf.model is not None
    assert rf.le is not None

    with pytest.raises(NotFittedError):
        check_is_fitted(rf.model)


def test_fitted():
    rf = RandomForestClassifier()
    rf.fit([[0, 0], [0, 0]], [0, 1])
    assert check_is_fitted(rf.model) is None
    assert rf.le is not None
    assert rf._get_feature_importance() is not None
    assert rf.get_params() is not None


@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_proba(rf_mock: mock.Mock()):
    expected_pred = [[0.5, 0.5]]
    _build_model_mock(rf_mock, expected_pred)
    rf = RandomForestClassifier()
    print(rf_mock.mock_calls)
    pred = rf.predict_proba([])
    assert_equal(pred, expected_pred)


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_below_threshold_50_equal_preds(
    rf_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[1 / 3, 1 / 3, 1 / 3]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([])
    assert pred == ['other']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_below_threshold_50(rf_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.4, 0.45, 0.15]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([])
    assert pred == ['other']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_equal_threshold_50_equal_prob_first(
    rf_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[0.5, 0.5, 0]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([])
    assert pred == ['bass']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_equal_threshold_50_equal_prob_mid(
    rf_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[0, 0.5, 0.5]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([])
    assert pred == ['drums']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_above_threshold_50(rf_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0, 0.4, 0.6]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([])
    assert pred == ['vocals']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_above_threshold_30(rf_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.2, 0.2, 0.6]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([], threshold=0.3)
    assert pred == ['vocals']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_predict_below_threshold_40(rf_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.35, 0.35, 0.3]]
    _build_model_mock(rf_mock, expected_pred)
    _build_encoder_mock(le_mock)
    rf = RandomForestClassifier()
    pred = rf.predict([], threshold=0.4)
    assert pred == ['other']


@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_get_feature_importance(rf_mock: mock.Mock):
    n_features = 2
    _build_model_mock(rf_mock, [])
    rf = RandomForestClassifier()
    expected = {f'feature {i}': 0 for i in range(n_features)}
    assert rf.get_feature_importance() == expected


@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_get_feature_importance_with_name(rf_mock: mock.Mock):
    n_features = 2
    _build_model_mock(rf_mock, [])
    rf = RandomForestClassifier()
    names = [f'f{i}' for i in range(n_features)]
    expected = {names[i]: 0 for i in range(n_features)}
    assert rf.get_feature_importance(names) == expected


@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_get_feature_importance_with_name_wrong_len(rf_mock: mock.Mock):
    n_features = 2
    _build_model_mock(rf_mock, [])
    rf = RandomForestClassifier()
    names = [f'f{i}' for i in range(n_features + 1)]

    with pytest.raises(IndexError):
        rf.get_feature_importance(names)


@mock.patch('sklearn.ensemble.RandomForestClassifier')
def test_save(rf_mock: mock.Mock):
    _build_model_mock(rf_mock, [])
    rf = RandomForestClassifier()
    rf.fit([[0, 0], [0, 0]], [0, 1])

    m_op = mock.mock_open()
    m_pickle = mock.mock_open()
    m_json = mock.mock_open()
    with mock.patch('builtins.open', m_op):
        with mock.patch('pickle.dump', m_pickle):
            with mock.patch('json.dump', m_json):
                rf.save_model('/path/to/', 'model')

    assert mock.call('/path/to/model.pkl', 'wb') in m_op.mock_calls
    assert mock.call('/path/to/model.json', 'w') in m_op.mock_calls
    assert m_json.call_args[0][0] == rf.get_params()
    assert m_pickle.call_args[0][0] == rf
