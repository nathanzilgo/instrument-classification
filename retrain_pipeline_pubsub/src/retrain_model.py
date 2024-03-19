import logging
import pandas as pd
from inda_mir.modeling.models.lgbm import LightGBMClassifier
from inda_mir.utils.gcs_interface import download_artifact, upload_artifact
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from retrain_pipeline_pubsub.src.utils import (
    is_unbalanced,
    save_extracted_tracks,
    sort_dataset,
)
from retrain_pipeline_pubsub.src.utils import split_train_test
from scripts.download_tracks import download_tracks
from scripts.feature_extraction import feature_extraction
from scripts.process_tracks import process_tracks

from scripts.util.gcloud import query_to_df
from retrain_pipeline_pubsub.src.settings import settings


def retrain_model() -> None:
    query_df = query_to_df(settings.TRACKS_QUERY).rename(
        columns={'instrument': 'label', 'id': 'track_id'}
    )
    query_df = sort_dataset(query_df)

    download_cached_data(query_df)
    process_tracks(query_df)

    trained_features = pd.read_csv(settings.TRAINED_FEATURES_PATH)
    feature_extraction(retrain=True, trained_features=trained_features)
    upload_artifact(ArtifactType.FEATURES, settings.TRAINED_FEATURES_PATH)

    save_extracted_tracks(query_df)
    if is_unbalanced(train_features=trained_features):
        logging.info('Unbalanced dataset! Aborting retraining!')
        return

    train_new_model(pd.read_csv(settings.TRAINED_FEATURES_PATH))


def train_new_model(trained_features: pd.DataFrame) -> None:
    X_train, y_train, _, _ = split_train_test(trained_features)

    lgbm = LightGBMClassifier()
    lgbm.fit(X_train, y_train)

    lgbm.save_model(
        path=settings.RETRAIN_OUTPUT_PATH,
        model_name=settings.MODEL_OUTPUT_NAME,
    )
    upload_artifact(ArtifactType.MODEL, settings.RETRAIN_OUTPUT_PATH)


def download_cached_data(query_df: pd.DataFrame) -> None:
    download_artifact(
        artifact_type=ArtifactType.FEATURES,
        filename=settings.TRAINED_FEATURES,
    )

    download_tracks(
        query_df,
        output_dir=settings.TRACKS_OUTPUT_DIR,
        metadata_path=settings.TRACKS_METADATA_PATH,
        bucket_name=settings.BUCKET_NAME,
    )


if __name__ == '__main__':
    retrain_model()
