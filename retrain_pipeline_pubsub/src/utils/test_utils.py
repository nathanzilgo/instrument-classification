from unittest import mock
import pandas as pd
from retrain_pipeline_pubsub.src.utils import (
    is_unbalanced,
    save_extracted_tracks,
    sort_dataset,
    split_train_test,
)
from retrain_pipeline_pubsub.src.settings import settings


def test_is_unbalanced_false():
    data = {
        'track_id': [1, 2, 3, 4],
        'label': ['keyboards', 'keyboards', 'guitar', 'guitar'],
        'dataset': ['train', 'test', 'train', 'train'],
    }
    features_df = pd.DataFrame(data)

    result = is_unbalanced(features_df)

    assert result == False


def test_is_unbalanced_true():
    data = {
        'track_id': [1, 2, 3, 4],
        'label': ['keyboards', 'keyboards', 'keyboards', 'bass'],
        'dataset': ['train', 'test', 'train', 'train'],
    }
    features_df = pd.DataFrame(data)

    result = is_unbalanced(features_df)

    assert result == True


@mock.patch('retrain_pipeline_pubsub.src.utils.bigquery.Client')
def test_save_extracted_tracks(client_mock: mock.Mock):
    df = pd.DataFrame(
        {
            'track_id': [1],
            'audio_url': ['url1'],
            'label': ['A'],
            'dataset': ['train'],
        }
    )

    save_extracted_tracks(df)

    expected_query = (
        f'{settings.SAVE_EXTRACTED_TRACKS_QUERY}(1, url1, A, train);'
    )
    client_mock.return_value.query.assert_called_once_with(expected_query)


@mock.patch('random.randint')
def test_sort_dataset_train(random_mock: mock.Mock):
    random_mock.return_value = 7
    df = pd.DataFrame(
        {
            'track_id': [1],
            'audio_url': ['url1'],
            'label': ['A'],
        }
    )

    result_df = sort_dataset(df)

    assert result_df['dataset'][0] == 'train'


@mock.patch('random.randint')
def test_sort_dataset_test(random_mock: mock.Mock):
    random_mock.return_value = 8
    df = pd.DataFrame(
        {
            'track_id': [1],
            'audio_url': ['url1'],
            'label': ['A'],
        }
    )

    result_df = sort_dataset(df)

    assert result_df['dataset'][0] == 'test'


def test_split_train_test():
    data = {
        'filename': ['file1', 'file2', 'file3'],
        'frame': [100, 200, 300],
        'track_id': [1, 2, 3],
        'label': [0, 1, 0],
        'dataset': ['train', 'test', 'train'],
    }
    features_df = pd.DataFrame(data)

    X_train, y_train, X_test, y_test = split_train_test(features_df.copy())

    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert len(y_test) == 1
