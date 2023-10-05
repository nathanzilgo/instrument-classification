import os

from scripts.util.gcloud import download_blob, query_to_df
from scripts.util.config import instrument_classification_config as icc

OUTPUT_DIR = icc['dirs']['RAW_TRACKS']
METADATA_PATH = icc['metadata']['RAW_TRACKS']
BUCKET_NAME = icc['params']['download_tracks']['BUCKET_NAME']
QUERY = icc['params']['download_tracks']['QUERY']

query_df = query_to_df(QUERY)

downloaded_paths = []
for row in query_df.itertuples():
    audio_extension = row.audio_url.split('.')[1]
    destination_file_name = os.path.join(
        OUTPUT_DIR, f'{row.track_id}.{audio_extension}'
    )

    if os.path.exists(destination_file_name):
        continue

    downloaded_paths.append(destination_file_name)
    download_blob(BUCKET_NAME, row.audio_url, destination_file_name)

query_df['path'] = downloaded_paths
query_df.drop(['audio_url'], axis=1).to_csv(METADATA_PATH, index=False)
