from datetime import datetime
import logging
import pandas as pd
from inda_mir.modeling.evaluation import (
    plot_confusion_matrix_tracklevel,
    print_classification_report,
)
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
    query_df = query_to_df(settings.TRACKS_QUERY)
    query_df = sort_dataset(query_df)

    download_cached_data(query_df)
    tracks_csv = pd.read_csv(settings.TRACKS_METADATA_PATH)[
        ['track_id', 'label', 'dataset']
    ]
    process_tracks(tracks=tracks_csv)

    trained_features = pd.read_csv(settings.TRAINED_FEATURES_PATH)
    feature_extraction(retrain=True, trained_features=trained_features)
    upload_artifact(ArtifactType.FEATURES, settings.TRAINED_FEATURES_PATH)

    save_extracted_tracks(query_df)
    if is_unbalanced(train_features=trained_features):
        logging.info('Unbalanced dataset! Aborting retraining!')
        return

    train_new_model(pd.read_csv(settings.TRAINED_FEATURES_PATH))


def train_new_model(trained_features: pd.DataFrame) -> None:
    X_train, y_train, X_test, y_test = split_train_test(trained_features)

    lgbm = LightGBMClassifier()
    lgbm.fit(X_train, y_train)

    model_name = (
        f'lgbm_retrained_{datetime.now().strftime("%H_%M_%S_%d_%m_%Y")}'
    )
    lgbm.save_model(
        path=settings.RETRAIN_OUTPUT_PATH,
        model_name=model_name,
    )

    cr_filename = print_classification_report(
        y_test,
        lgbm.predict(X_test, threshold=0.7),
        labels=lgbm.classes_,
        metrics_path=settings.METRICS_OUTPUT_PATH,
    )
    cm_filename = plot_confusion_matrix_tracklevel(
        lgbm,
        lgbm.predict(X_test, threshold=0.7),
        y_test,
        trained_features[trained_features['dataset'] == 'test'],
        threshold=0.7,
        metrics_path=settings.METRICS_OUTPUT_PATH,
    )
    upload_artifact(ArtifactType.METRICS, f'{cm_filename}.png')
    upload_artifact(ArtifactType.METRICS, cr_filename)
    upload_artifact(ArtifactType.MODEL, f'{model_name}.pkl')


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
