import os
import pandas as pd
from inda_mir.modeling.evaluation import (
    plot_confusion_matrix_tracklevel,
    print_classification_report,
)
from inda_mir.modeling.models.lgbm import LightGBMClassifier
from scripts.feature_extraction import feature_extraction
from scripts.download_tracks import download_tracks
from scripts.process_tracks import process_tracks

from scripts.util.gcloud import query_to_df
from scripts.util.config import instrument_classification_config as icc

from collections import Counter

OUTPUT_DIR = icc['dirs']['RAW_TRACKS']
METADATA_PATH = icc['metadata']['RAW_TRACKS']
BUCKET_NAME = icc['params']['download_tracks']['BUCKET_NAME']
QUERY = icc['params']['download_tracks']['QUERY']
TRAINED_FEATURES = icc['params']['download_features']['TRAINED_FEATURES']

TRAINED_FEATURES_PATH = os.path.join('output/features', TRAINED_FEATURES)

OUTPUT_PATH = icc['outputs']['FEATURES_EXTRACTED']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']
TRAINED_FEATURES = icc['params']['download_features']['TRAINED_FEATURES']
VALIDATION_FEATURES = icc['outputs']['FEATURES_VALIDATION']
FEATURES_EXTRACTED = icc['outputs']['FEATURES_EXTRACTED']


RETRAIN_OUTPUT_PATH = icc['dirs']['RETRAINED_MODEL']
MODEL_OUTPUT_NAME = 'lgbm_retrained_untuned'

query_df = query_to_df(QUERY)

# Download our latest features csv
if not os.path.exists(TRAINED_FEATURES_PATH):
    os.system(
        f'python scripts/gcs_interface.py -o download -t features -f {TRAINED_FEATURES}'
    )

trained_features = pd.read_csv(f'./output/features/{TRAINED_FEATURES}')
validation_features = pd.read_csv(VALIDATION_FEATURES)

download_tracks(
    query_df,
    output_dir=OUTPUT_DIR,
    metadata_path=METADATA_PATH,
    bucket_name=BUCKET_NAME,
    skip_tracks_already_trained=True,  # Skip tracks that are already in the trained features
    trained_features=trained_features,
)

process_tracks(query_df)

# Feature extraction using existing features csv
trained_features = pd.read_csv(TRAINED_FEATURES_PATH)
feature_extraction(retrain=True, trained_features=trained_features)

# Train / test split
features_extracted = pd.read_csv(FEATURES_EXTRACTED)

X_train = features_extracted.drop(
    ['filename', 'frame', 'track_id', 'label'], axis=1
).to_numpy()
y_train = features_extracted['label'].to_numpy()

X_test = validation_features.drop(
    ['filename', 'frame', 'track_id', 'label'], axis=1
).to_numpy()
y_test = validation_features['label'].to_numpy()

print(Counter(y_train))

lgbm = LightGBMClassifier()
lgbm.fit(X_train, y_train)

lgbm.save_model(
    path=RETRAIN_OUTPUT_PATH,
    model_name=MODEL_OUTPUT_NAME,
)

# Show metrics
print_classification_report(y_test, lgbm.predict(X_test, threshold=0.7))
plot_confusion_matrix_tracklevel(
    lgbm,
    lgbm.predict(X_test, threshold=0.7),
    y_test,
    validation_features,
    threshold=0.7,
)
