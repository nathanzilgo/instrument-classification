from unittest import mock

import pytest

import pandas

from inda_mir.modeling.feature_extractor import FreesoundExtractor
from inda_mir.utils.logger import logger


@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_internal_extract(os_mock: mock.Mock, json_mock: mock.Mock):
    os_mock.system.return_value = None
    json_mock.return_value = {
        'lowlevel': {'a': {'a': 0, 'b': 0}, 'b': {'a': 0, 'b': 0}}
    }

    fe = FreesoundExtractor()
    features = fe._extract('/path/to/file')
    assert 'a_a' in features
    assert 'a_b' in features
    assert 'b_a' in features
    assert 'b_b' in features
    for f in features:
        assert features[f] == 0


@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_extract_not_saving_df(os_mock: mock.Mock, json_mock: mock.Mock):
    os_mock.system.return_value = None
    json_mock.return_value = {
        'lowlevel': {'a': {'a': 0, 'b': 0}, 'b': {'a': 0, 'b': 0}}
    }

    expected_features_names = ['a_a', 'a_b', 'b_a', 'b_b']

    fe = FreesoundExtractor()
    features_names, features_set = fe.extract(
        ['/path/to/file1', '/path/to/file2'], '/path/to/output', save_df=False
    )

    for f in expected_features_names:
        assert f in features_names

    for features in features_set:
        for f in features:
            assert features[f] == 0


@mock.patch('inda_mir.utils.logger')
@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_extract_exception(
    os_mock: mock.Mock, json_mock: mock.Mock, logger_mock: mock.Mock
):
    os_mock.system.return_value = None
    json_mock.side_effect = [
        None,
        {'lowlevel': {'a': {'a': 0, 'b': 0}, 'b': {'a': 0, 'b': 0}}},
    ]

    expected_features_names = ['a_a', 'a_b', 'b_a', 'b_b']

    fe = FreesoundExtractor()
    logger_mock = mock.Mock()
    with mock.patch.object(logger, 'error', logger_mock):
        features_names, features_set = fe.extract(
            ['/path/to/file1', '/path/to/file2'],
            '/path/to/output',
            save_df=False,
        )

    assert logger_mock.called
    assert len(features_set) == 1

    for f in expected_features_names:
        assert f in features_names

    for features in features_set:
        for f in features:
            assert features[f] == 0


@mock.patch('inda_mir.utils.logger')
@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_extract_empty_feature_list(
    os_mock: mock.Mock, json_mock: mock.Mock, logger_mock: mock.Mock
):
    os_mock.system.return_value = None
    json_mock.side_effect = None

    fe = FreesoundExtractor()
    logger_mock = mock.Mock()
    with mock.patch.object(logger, 'error', logger_mock):
        features_names, features_set = fe.extract(
            ['/path/to/file1', '/path/to/file2'],
            '/path/to/output',
            save_df=True,
        )

    assert len(logger_mock.call_args_list) == 3
    assert len(features_set) == 0
    assert len(features_names) == 0


@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_extract_saving_df(os_mock: mock.Mock, json_mock: mock.Mock):
    os_mock.system.return_value = None
    json_mock.return_value = {
        'lowlevel': {'a': {'a': 0, 'b': 0}, 'b': {'a': 0, 'b': 0}}
    }

    fe = FreesoundExtractor()
    pandas_mock = mock.Mock()
    with mock.patch.object(pandas.DataFrame, 'to_csv', pandas_mock):
        fe.extract(['/path/to/file1', '/path/to/file2'], '/path/to/output')

    assert pandas_mock.called
    assert (
        mock.call('/path/to/output', sep=',', index=False)
        in pandas_mock.mock_calls
    )


def test_extract_saving_df_diff_lengths():

    fe = FreesoundExtractor()

    with pytest.raises(ValueError):
        fe.save_features_as_df(['file'], [], '/path/to/output')


@mock.patch('json.load')
@mock.patch(
    'inda_mir.modeling.feature_extractor.essentia_freesound_extractor.os'
)
def test_extract_checkpoint(os_mock: mock.Mock, json_mock: mock.Mock):
    os_mock.system.return_value = None
    json_mock.return_value = {
        'lowlevel': {'a': {'a': 0, 'b': 0}, 'b': {'a': 0, 'b': 0}}
    }

    expected_checkpoint = {
        'files': ['/path/to/file1', '/path/to/file2'],
        'features': [
            {'a_a': 0, 'a_b': 0, 'b_a': 0, 'b_b': 0},
            {'a_a': 0, 'a_b': 0, 'b_a': 0, 'b_b': 0},
        ],
    }

    fe = FreesoundExtractor()
    m_op: mock.Mock = mock.mock_open()
    m_pickle: mock.Mock = mock.mock_open()
    with mock.patch('builtins.open', m_op):
        with mock.patch('pickle.dump', m_pickle):
            fe.extract(
                ['/path/to/file1', '/path/to/file2'],
                '/path/to/output',
                checkpoint=True,
                save_df=False,
            )

    assert mock.call('/path/to/output.checkpoint', 'wb') in m_op.mock_calls
    m_pickle.assert_called_once_with(expected_checkpoint, m_op.return_value)
