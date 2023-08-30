import os
from typing import NamedTuple
from urllib.request import urlretrieve
import pandas as pd

from google.cloud import bigquery

from inda_mir.audio_processing import SampleOperation, SilenceFilter
from inda_mir.utils.logger import logger


Track = NamedTuple('Track', id=str, audio_url=str, label=str)

OUTPUT_DIR = './output-inda'
OUTPUT_DIR_RAW = os.path.join(OUTPUT_DIR, 'raw')
OUTPUT_DIR_SILENCE = os.path.join(OUTPUT_DIR, 'silenced')
OUTPUT_DIR_SAMPLE = os.path.join(OUTPUT_DIR, 'sampled')
OUTPUT_DIR_METADATA = os.path.join(OUTPUT_DIR, 'metadata')
OUTPUT_FORMAT = 'ogg'
SAMPLE_DURATION = 10000
AUDIO_BASE_PATH = 'https://uploads.storage.inda.band/'

os.makedirs(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR_RAW)
os.makedirs(OUTPUT_DIR_SAMPLE)
os.makedirs(OUTPUT_DIR_SILENCE)
os.makedirs(OUTPUT_DIR_METADATA)

client = bigquery.Client()
query_job = client.query(
    'SELECT * from track_classification.track_labels_filtered;'
)  # Make an API request.

tracks = []
metadata = []

for row in query_job:
    track = Track._make(row)
    try:
        logger.info(f'Processing track {track.id}')

        if 'http' in track.audio_url:
            audio_url = track.audio_url
        else:
            audio_url = AUDIO_BASE_PATH + track.audio_url

        audio_url_format = track.audio_url.split('.')[-1]
        filename = track.id + '.' + audio_url_format
        downloaded_path, _ = urlretrieve(
            audio_url, os.path.join(OUTPUT_DIR_RAW, filename)
        )

        output_filename = os.path.join(OUTPUT_DIR_SILENCE, track.id)
        output_silence_path = SilenceFilter.apply(
            downloaded_path, audio_url_format, OUTPUT_FORMAT, output_filename
        )

        track_sample_dir = os.path.join(OUTPUT_DIR_SAMPLE, track.id)
        output_filename = os.path.join(track_sample_dir, track.id)
        os.makedirs(track_sample_dir)

        SampleOperation.apply(
            output_silence_path,
            OUTPUT_FORMAT,
            OUTPUT_FORMAT,
            SAMPLE_DURATION,
            output_filename,
        )

        for sample_path in os.scandir(track_sample_dir):
            metadata.append(
                {
                    'track_id': track.id,
                    'sample_path': os.path.abspath(sample_path.path),
                    'label': track.label,
                }
            )

        logger.info(
            f'Success on track {track.id}, samples saved at {track_sample_dir}'
        )
    except Exception as e:
        logger.error(f'Error on track {track.id}, {e}')

df = pd.DataFrame.from_records(metadata, index=range(0, len(metadata)))
df.to_csv(
    f'{OUTPUT_DIR_METADATA}/metadata.csv',
    index=False,
    lineterminator='\n',
)
