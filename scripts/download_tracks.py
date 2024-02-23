import os

import pandas as pd
from tqdm import tqdm

from scripts.util.gcloud import download_blob, query_to_df
from scripts.util.config import instrument_classification_config as icc

OUTPUT_DIR = icc['dirs']['RAW_TRACKS']
METADATA_PATH = icc['metadata']['RAW_TRACKS']
BUCKET_NAME = icc['params']['download_tracks']['BUCKET_NAME']
QUERY = icc['params']['download_tracks']['TRAIN']

query_df = query_to_df(QUERY)


def download_tracks(
    query_df: pd.DataFrame = query_df,
    output_dir: str = OUTPUT_DIR,
    metadata_path: str = METADATA_PATH,
    bucket_name: str = BUCKET_NAME,
    skip_tracks_already_trained: bool = False,
    trained_features: pd.DataFrame = None,
) -> None:
    """
    Download tracks from a bucket, based on the given query dataframe, saving them to the specified output directory.
    The output csv is stored in the metadata directory, containing info only about the tracks that were effectively downloaded.

    Args:
        query_df (pd.DataFrame): The dataframe containing the query for tracks to be downloaded.
        output_dir (str): The directory where the downloaded tracks will be saved.
        metadata_path (str): The path to the metadata file where track information will be stored.
        bucket_name (str): The name of the bucket where the audio files are stored.
        skip_tracks_already_trained (bool): Whether to skip tracks that have already been trained.
        trained_features (pd.DataFrame): A dataframe containing trained features, used to check for already trained tracks.

    Returns:
        None
    """
    downloaded_paths = []
    for row in tqdm(
        query_df.itertuples(),
        total=query_df.shape[0],
        desc='Downloading tracks',
    ):
        if skip_tracks_already_trained:
            if row.track_id in trained_features['track_id'].values:
                query_df = query_df[query_df['track_id'] != row.track_id]
                continue

        audio_extension = row.audio_url.split('.')[1]
        destination_file_name = os.path.join(
            output_dir, f'{row.track_id}.{audio_extension}'
        )

        downloaded_paths.append(destination_file_name)

        if not os.path.exists(destination_file_name):
            download_blob(bucket_name, row.audio_url, destination_file_name)

    query_df['path'] = downloaded_paths
    query_df.drop(['audio_url'], axis=1).to_csv(metadata_path, index=False)


if __name__ == '__main__':
    download_tracks()
