from unittest import mock

import pandas as pd
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from retrain_pipeline_pubsub.src.settings import settings
from retrain_pipeline_pubsub.src.retrain_model import (
    download_cached_data,
    retrain_model,
    train_new_model,
)


@mock.patch('retrain_pipeline_pubsub.src.retrain_model.train_new_model')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.process_tracks')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_cached_data')
@mock.patch('pandas.read_csv')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.query_to_df')
def test_retrain_model(
    query_to_df_mock: mock.Mock,
    read_csv_mock: mock.Mock,
    download_data_mock: mock.Mock,
    process_tracks_mock: mock.Mock,
    train_model_mock: mock.Mock,
) -> None:
    read_csv_mock.return_value = pd.DataFrame(
        {
            'filename': ['file1', 'file2', 'file3'],
            'frame': [1, 2, 3],
            'track_id': [101, 102, 103],
            'label': ['A', 'B', 'C'],
            'dataset': ['train', 'train', 'test'],
        }
    )
    query_to_df_mock.return_value = pd.DataFrame(
        {
            'track_id': [101, 102, 104],
            'feature_1': [0.5, 0.8, 0.3],
            'feature_2': [0.2, 0.1, 0.7],
        }
    )

    retrain_model()

    query_to_df_mock.assert_called_once_with(settings.TRACKS_QUERY)
    read_csv_mock.assert_called_once_with(settings.TRAINED_FEATURES_PATH)
    download_data_mock.assert_called_once_with(
        query_to_df_mock.return_value, read_csv_mock.return_value
    )
    process_tracks_mock.assert_called_once_with(query_to_df_mock.return_value)
    train_model_mock.assert_called_once()


@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_tracks')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_artifact')
@mock.patch('os.path.exists')
def test_download_cached_data_with_features_download(
    mock_path: mock.Mock,
    download_artifact_mock: mock.Mock,
    download_tracks_mock: mock.Mock,
) -> None:
    query_df = {
        'track_id': [101, 102, 104],
        'feature_1': [0.5, 0.8, 0.3],
        'feature_2': [0.2, 0.1, 0.7],
    }
    trained_features = {
        'filename': ['file1', 'file2', 'file3'],
        'frame': [1, 2, 3],
        'track_id': [101, 102, 103],
        'label': ['A', 'B', 'C'],
        'dataset': ['train', 'train', 'test'],
    }
    mock_path.return_value = False

    download_cached_data(query_df, trained_features)

    mock_path.assert_called_once_with(settings.TRAINED_FEATURES_PATH)
    download_artifact_mock.assert_called_once_with(
        artifact_type=ArtifactType.FEATURES, filename=settings.TRAINED_FEATURES
    )
    download_tracks_mock.assert_called_once_with(
        query_df,
        output_dir=settings.TRACKS_OUTPUT_DIR,
        metadata_path=settings.TRACKS_METADATA_PATH,
        bucket_name=settings.BUCKET_NAME,
        skip_tracks_already_trained=True,
        trained_features=trained_features,
    )


@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_tracks')
@mock.patch('os.path.exists')
def test_download_cached_data_without_features_download(
    mock_path: mock.Mock,
    download_tracks_mock: mock.Mock,
) -> None:
    query_df = {
        'track_id': [101, 102, 104],
        'feature_1': [0.5, 0.8, 0.3],
        'feature_2': [0.2, 0.1, 0.7],
    }
    trained_features = {
        'filename': ['file1', 'file2', 'file3'],
        'frame': [1, 2, 3],
        'track_id': [101, 102, 103],
        'label': ['A', 'B', 'C'],
        'dataset': ['train', 'train', 'test'],
    }
    mock_path.return_value = True

    download_cached_data(query_df, trained_features)

    mock_path.assert_called_once_with(settings.TRAINED_FEATURES_PATH)
    download_tracks_mock.assert_called_once_with(
        query_df,
        output_dir=settings.TRACKS_OUTPUT_DIR,
        metadata_path=settings.TRACKS_METADATA_PATH,
        bucket_name=settings.BUCKET_NAME,
        skip_tracks_already_trained=True,
        trained_features=trained_features,
    )


@mock.patch('pandas.read_csv')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.LightGBMClassifier')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.upload_artifact')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.feature_extraction')
def test_train_new_model(
    feature_extraction_mock: mock.Mock,
    upload_artifact_mock: mock.Mock,
    lgbm_mock: mock.Mock,
    read_csv_mock: mock.Mock,
) -> None:
    trained_features = pd.DataFrame(
        {
            'filename': ['file1', 'file2', 'file3'],
            'frame': [1, 2, 3],
            'track_id': [101, 102, 103],
            'label': ['A', 'B', 'C'],
            'dataset': ['train', 'train', 'test'],
        }
    )

    read_csv_mock.return_value = trained_features

    train_new_model(trained_features)

    feature_extraction_mock.assert_called_once_with(
        retrain=True, trained_features=trained_features
    )
    upload_artifact_mock.assert_has_calls(
        [
            mock.call(ArtifactType.FEATURES, settings.TRAINED_FEATURES_PATH),
            mock.call(ArtifactType.MODEL, settings.RETRAIN_OUTPUT_PATH),
        ]
    )
    read_csv_mock.assert_called_once_with(settings.TRAINED_FEATURES_PATH)
    lgbm_mock.return_value.fit.assert_called_once()
    lgbm_mock.return_value.save_model.assert_called_once_with(
        path=settings.RETRAIN_OUTPUT_PATH,
        model_name=settings.MODEL_OUTPUT_NAME,
    )
