import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_ID: str = 'loyal-parser-316218'
    TRACKS_OUTPUT_DIR: str = './output/raw_tracks'
    TRACKS_METADATA_PATH: str = './output/metadata/tracks.csv'
    BUCKET_NAME: str = 'inda-storage-uploads'
    TRACKS_QUERY: str = 'SELECT * FROM track_classification.track_labels_datasets LIMIT 10;'
    TRAINED_FEATURES: str = 'essentia_train_features.csv'
    TRAINED_FEATURES_PATH: str = os.path.join('output/features', TRAINED_FEATURES)
    FEATURES_EXTRACTED: str = 'essentia_extracted_features.csv'
    RETRAIN_OUTPUT_PATH: str = './output/models'
    MODEL_OUTPUT_NAME: str = 'lgbm_retrained_untuned'
    UP_SUBSCRIPTION_ID: str = 'instrument-classifier-model-retrain-scale-up-sub'
    FINISH_TOPIC_ID: str = 'instrument-classifier-model-retrain-notify-finish'
    MAX_MESSAGES: int = 1

settings = Settings()