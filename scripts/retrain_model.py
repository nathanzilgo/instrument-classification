import os
import pandas as pd
from inda_mir.modeling.models.lgbm import LightGBMClassifier
from scripts.feature_extraction import feature_extraction
from scripts.train_test_split import train_test_split
from scripts.download_tracks import download_tracks
from scripts.process_tracks import process_tracks

from scripts.util.gcloud import query_to_df
from scripts.util.config import instrument_classification_config as icc

from inda_mir.loaders import load_data_partition
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

RETRAIN_OUTPUT_PATH = icc['outputs']['RETRAINED_MODEL']
MODEL_OUTPUT_NAME = 'lgbm_retrained_untuned'

query_df = query_to_df(QUERY)

# Download our latest features csv
if not os.path.exists(TRAINED_FEATURES_PATH):
    os.system(
        f'python scripts/gcs_interface.py -o download -t features -f {TRAINED_FEATURES}'
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

# Feature extraction using existing features csv
trained_features = pd.read_csv(TRAINED_FEATURES_PATH)
feature_extraction(retrain=True, trained_features=trained_features)

# Train / test split
train_test_split()

data_partition_path = './output/train_test_split/random_split.data'
data = load_data_partition(data_partition_path)
X_train, y_train = data.get_numpy_train_data()
X_test, y_test = data.get_numpy_test_data()

print(Counter(y_train))

lgbm = LightGBMClassifier()
lgbm.fit(X_train, y_train)

lgbm.save_model(
    path=RETRAIN_OUTPUT_PATH,
    model_name=MODEL_OUTPUT_NAME,
)
