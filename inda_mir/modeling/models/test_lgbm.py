import pytest

from typing import List
from unittest import mock

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from numpy.testing import assert_equal
from numpy import asarray

from inda_mir.modeling.models import LightGBMClassifier


def _build_model_mock(lgc_mock: mock.Mock, pred: List, n_features: int = 2):
    lgc_mock_fitted = mock.Mock()
    lgc_mock_fitted.predict_proba.return_value = asarray(pred)
    lgc_mock_fitted.feature_importances_ = [0 for _ in range(n_features)]
    lgc_mock.return_value = lgc_mock_fitted


def _build_encoder_mock(le_mock: mock.Mock):
    le_mock_fitted = mock.Mock()
    classes = ['bass', 'drums', 'vocals']
    le_mock_fitted.inverse_transform = lambda x: [classes[i] for i in x]
    le_mock_fitted.transform = lambda c: [classes.index(i) for i in c]
    le_mock.return_value = le_mock_fitted


def test_init():
    lgc = LightGBMClassifier()
    assert lgc.name == 'LightGBM'
    assert lgc.model is not None
    assert lgc.le is not None

    with pytest.raises(NotFittedError):
        check_is_fitted(lgc.model)


def test_fitted():
    lgc = LightGBMClassifier()
    lgc.fit([[0, 0], [0, 0]], [0, 1])
    assert check_is_fitted(lgc.model) is None
    assert lgc.le is not None
    assert lgc._get_feature_importance() is not None
    assert lgc.get_params() is not None


@mock.patch('lightgbm.LGBMClassifier')
def test_predict_proba(lgc_mock: mock.Mock()):
    expected_pred = [[0.5, 0.5]]
    _build_model_mock(lgc_mock, expected_pred)
    lgc = LightGBMClassifier()
    pred = lgc.predict_proba([])
    assert_equal(pred, expected_pred)


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_below_threshold_50_equal_preds(
    lgc_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[1 / 3, 1 / 3, 1 / 3]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([])
    assert pred == ['other']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_below_threshold_50(lgc_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.4, 0.45, 0.15]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([])
    assert pred == ['other']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_equal_threshold_50_equal_prob_first(
    lgc_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[0.5, 0.5, 0]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([])
    assert pred == ['bass']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_equal_threshold_50_equal_prob_mid(
    lgc_mock: mock.Mock, le_mock: mock.Mock
):
    expected_pred = [[0, 0.5, 0.5]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([])
    assert pred == ['drums']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_above_threshold_50(lgc_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0, 0.4, 0.6]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([])
    assert pred == ['vocals']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_above_threshold_30(lgc_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.2, 0.2, 0.6]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([], threshold=0.3)
    assert pred == ['vocals']


@mock.patch('sklearn.preprocessing.LabelEncoder')
@mock.patch('lightgbm.LGBMClassifier')
def test_predict_below_threshold_40(lgc_mock: mock.Mock, le_mock: mock.Mock):
    expected_pred = [[0.35, 0.35, 0.3]]
    _build_model_mock(lgc_mock, expected_pred)
    _build_encoder_mock(le_mock)
    lgc = LightGBMClassifier()
    pred = lgc.predict([], threshold=0.4)
    assert pred == ['other']


@mock.patch('lightgbm.LGBMClassifier')
def test_get_feature_importance(lgc_mock: mock.Mock):
    n_features = 2
    _build_model_mock(lgc_mock, [])
    lgc = LightGBMClassifier()
    expected = {f'feature {i}': 0 for i in range(n_features)}
    assert lgc.get_feature_importance() == expected


@mock.patch('lightgbm.LGBMClassifier')
def test_get_feature_importance_with_name(lgc_mock: mock.Mock):
    n_features = 2
    _build_model_mock(lgc_mock, [])
    lgc = LightGBMClassifier()
    names = [f'f{i}' for i in range(n_features)]
    expected = {names[i]: 0 for i in range(n_features)}
    assert lgc.get_feature_importance(names) == expected


@mock.patch('lightgbm.LGBMClassifier')
def test_get_feature_importance_with_name_wrong_len(lgc_mock: mock.Mock):
    n_features = 2
    _build_model_mock(lgc_mock, [])
    lgc = LightGBMClassifier()
    names = [f'f{i}' for i in range(n_features + 1)]

    with pytest.raises(IndexError):
        lgc.get_feature_importance(names)


@mock.patch('lightgbm.LGBMClassifier')
def test_save(lgc_mock: mock.Mock):
    _build_model_mock(lgc_mock, [])
    lgc = LightGBMClassifier()
    lgc.fit([[0, 0], [0, 0]], [0, 1])

    m_op = mock.mock_open()
    m_pickle = mock.mock_open()
    m_json = mock.mock_open()
    with mock.patch('builtins.open', m_op):
        with mock.patch('pickle.dump', m_pickle):
            with mock.patch('json.dump', m_json):
                lgc.save_model('/path/to/', 'model')

    assert mock.call('/path/to/model.pkl', 'wb') in m_op.mock_calls
    assert mock.call('/path/to/model.json', 'w') in m_op.mock_calls
    assert m_json.call_args[0][0] == lgc.get_params()
    assert m_pickle.call_args[0][0] == lgc
