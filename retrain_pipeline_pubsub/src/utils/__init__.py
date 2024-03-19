import random
import pandas as pd
from google.cloud import bigquery
from retrain_pipeline_pubsub.src.settings import settings


def is_unbalanced(train_features: pd.DataFrame) -> bool:
    grouped_train_features = train_features.groupby('label')

    max_value = grouped_train_features.size().max()
    min_value = grouped_train_features.size().min()
    dataset_length = len(train_features)

    imbalance = (max_value - min_value) / (dataset_length)

    return imbalance > settings.UNBALANCE_THRESHOLD


def sort_dataset(query_df: pd.DataFrame) -> pd.DataFrame:
    dataset = [
        'train' if random.randint(1, 10) <= 7 else 'test'
        for _ in range(len(query_df))
    ]

    return query_df.assign(dataset=dataset)


def save_extracted_tracks(query_df: pd.DataFrame) -> None:
    client = bigquery.Client()
    query = settings.SAVE_EXTRACTED_TRACKS_QUERY

    for i in range(len(query_df)):
        query += f'({query_df["track_id"][i]}, {query_df["audio_url"][i]}, {query_df["label"][i]}, {query_df["dataset"][i]})'

        if i < len(query_df) - 1:
            query += ', '
        else:
            query += ';'

    query_job = client.query(query)
    query_job.result()


def split_train_test(features_extracted: pd.DataFrame):
    train_features = features_extracted[
        features_extracted['dataset'] != 'test'
    ]
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

    return (X_train, y_train, X_test, y_test)
