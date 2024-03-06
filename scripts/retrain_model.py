import os
import pandas as pd
from inda_mir.modeling.evaluation import (
    plot_confusion_matrix_tracklevel,
    print_classification_report,
)
from inda_mir.modeling.models.lgbm import LightGBMClassifier
from inda_mir.utils.gcs_interface import download_artifact, upload_artifact
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from scripts.feature_extraction import feature_extraction
from scripts.download_tracks import download_tracks
from scripts.process_tracks import process_tracks

from scripts.util.gcloud import query_to_df
from scripts.util.config import instrument_classification_config as icc

OUTPUT_DIR = icc['dirs']['RAW_TRACKS']
METADATA_PATH = icc['metadata']['RAW_TRACKS']
BUCKET_NAME = icc['params']['download_tracks']['BUCKET_NAME']
TRACKS_QUERY = icc['params']['download_tracks']['TRACKS_QUERY']
TRAINED_FEATURES = icc['params']['download_features']['TRAINED_FEATURES']

TRAINED_FEATURES_PATH = os.path.join('output/features', TRAINED_FEATURES)

OUTPUT_PATH = icc['outputs']['FEATURES_EXTRACTED']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']
TRAINED_FEATURES = icc['params']['download_features']['TRAINED_FEATURES']
VALIDATION_FEATURES = icc['outputs']['FEATURES_VALIDATION']
FEATURES_EXTRACTED = icc['outputs']['FEATURES_EXTRACTED']


RETRAIN_OUTPUT_PATH = icc['dirs']['RETRAINED_MODEL']
MODEL_OUTPUT_NAME = 'lgbm_retrained_untuned'

query_df = query_to_df(TRACKS_QUERY)

# Download our latest features csv
if not os.path.exists(TRAINED_FEATURES_PATH):
    download_artifact(
        artifact_type=ArtifactType.FEATURES, filename=TRAINED_FEATURES
    )

trained_features = pd.read_csv(f'./output/features/{TRAINED_FEATURES}')

download_tracks(
    query_df,
    output_dir=OUTPUT_DIR,
    metadata_path=METADATA_PATH,
    bucket_name=BUCKET_NAME,
    skip_tracks_already_trained=True,  # Skip tracks that are already in the trained features
    trained_features=trained_features,
)

process_tracks(query_df)
# TODO - Save these samples in a GCS bucket after the processing

# Feature extraction using existing features csv
trained_features = pd.read_csv(TRAINED_FEATURES_PATH)
feature_extraction(retrain=True, trained_features=trained_features)
upload_artifact(ArtifactType.FEATURES, TRAINED_FEATURES_PATH)

# Train / test split
features_extracted = pd.read_csv(FEATURES_EXTRACTED)
# TODO - Put this split inside our split function
train_features = features_extracted[features_extracted['dataset'] != 'test']
validation_features = features_extracted[
    features_extracted['dataset'] == 'test'
]

X_train = train_features.drop(
    ['filename', 'frame', 'track_id', 'label', 'dataset'], axis=1
).to_numpy()
y_train = train_features['label'].to_numpy()

X_test = validation_features.drop(
    ['filename', 'frame', 'track_id', 'label', 'dataset'], axis=1
).to_numpy()
y_test = validation_features['label'].to_numpy()

lgbm = LightGBMClassifier()
lgbm.fit(X_train, y_train)

lgbm.save_model(
    path=RETRAIN_OUTPUT_PATH,
    model_name=MODEL_OUTPUT_NAME,
)
upload_artifact(ArtifactType.MODEL, RETRAIN_OUTPUT_PATH)

# Show metrics
print_classification_report(y_test, lgbm.predict(X_test, threshold=0.7))
plot_confusion_matrix_tracklevel(
    lgbm,
    lgbm.predict(X_test, threshold=0.7),
    y_test,
    validation_features,
    threshold=0.7,
)
