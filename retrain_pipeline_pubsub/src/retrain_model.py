import os
import pandas as pd
from inda_mir.modeling.models.lgbm import LightGBMClassifier
from inda_mir.utils.gcs_interface import download_artifact, upload_artifact
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from retrain_pipeline_pubsub.src.utils.split_train_test import split_train_test
from scripts.download_tracks import download_tracks
from scripts.feature_extraction import feature_extraction
from scripts.process_tracks import process_tracks

from scripts.util.gcloud import query_to_df
from retrain_pipeline_pubsub.src.settings import settings


def retrain_model() -> None:
    query_df = query_to_df(settings.TRACKS_QUERY)
    trained_features = pd.read_csv(settings.TRAINED_FEATURES_PATH)

    download_cached_data(query_df, trained_features)

    process_tracks(query_df)

    train_new_model()


def train_new_model(trained_features: pd.DataFrame) -> None:
    # Feature extraction using existing features csv
    feature_extraction(retrain=True, trained_features=trained_features)
    upload_artifact(ArtifactType.FEATURES, settings.TRAINED_FEATURES_PATH)

    # Train / test split
    features_extracted = pd.read_csv(settings.TRAINED_FEATURES_PATH)
    X_train, y_train, _, _ = split_train_test(features_extracted)

    lgbm = LightGBMClassifier()
    lgbm.fit(X_train, y_train)

    lgbm.save_model(
        path=settings.RETRAIN_OUTPUT_PATH,
        model_name=settings.MODEL_OUTPUT_NAME,
    )
    upload_artifact(ArtifactType.MODEL, settings.RETRAIN_OUTPUT_PATH)


def download_cached_data(
    query_df: pd.DataFrame, trained_features: pd.DataFrame
) -> None:
    if not os.path.exists(settings.TRAINED_FEATURES_PATH):
        download_artifact(
            artifact_type=ArtifactType.FEATURES,
            filename=settings.TRAINED_FEATURES,
        )

    download_tracks(
        query_df,
        output_dir=settings.TRACKS_OUTPUT_DIR,
        metadata_path=settings.TRACKS_METADATA_PATH,
        bucket_name=settings.BUCKET_NAME,
        skip_tracks_already_trained=True,  # Skip tracks that are already in the trained features
        trained_features=trained_features,
    )


if __name__ == '__main__':
    retrain_model()
