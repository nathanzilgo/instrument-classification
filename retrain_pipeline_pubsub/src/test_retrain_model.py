from unittest import mock

import pandas as pd
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from retrain_pipeline_pubsub.src.settings import settings
from retrain_pipeline_pubsub.src.retrain_model import (
    download_cached_data,
    retrain_model,
    train_new_model,
)


@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_tracks')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.download_artifact')
def test_download_cached_data(
    download_artifact_mock: mock.Mock,
    download_tracks_mock: mock.Mock,
) -> None:
    query_df = {
        'track_id': [101, 102, 104],
        'feature_1': [0.5, 0.8, 0.3],
        'feature_2': [0.2, 0.1, 0.7],
    }

    download_cached_data(query_df)

    download_artifact_mock.assert_called_once_with(
        artifact_type=ArtifactType.FEATURES, filename=settings.TRAINED_FEATURES
    )
    download_tracks_mock.assert_called_once_with(
        query_df,
        output_dir=settings.TRACKS_OUTPUT_DIR,
        metadata_path=settings.TRACKS_METADATA_PATH,
        bucket_name=settings.BUCKET_NAME,
    )


@mock.patch('retrain_pipeline_pubsub.src.retrain_model.LightGBMClassifier')
@mock.patch('retrain_pipeline_pubsub.src.retrain_model.upload_artifact')
def test_train_new_model(
    upload_artifact_mock: mock.Mock,
    lgbm_mock: mock.Mock,
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

    train_new_model(trained_features)

    upload_artifact_mock.assert_called_once_with(
        ArtifactType.MODEL, settings.RETRAIN_OUTPUT_PATH
    )
    lgbm_mock.return_value.fit.assert_called_once()
    lgbm_mock.return_value.save_model.assert_called_once_with(
        path=settings.RETRAIN_OUTPUT_PATH,
        model_name=settings.MODEL_OUTPUT_NAME,
    )
