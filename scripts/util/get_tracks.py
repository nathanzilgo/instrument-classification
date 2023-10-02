from typing import NamedTuple
from urllib.request import urlretrieve

import pandas as pd
import os

from inda_mir.utils.logger import logger


OUTPUT_METADATA_PATH = './output-inda/metadata/metadata_raw.csv'
REMOTE_QUERY_PATH = './output-inda/remote_tracks_query.csv'

RemoteTrack = NamedTuple('Track', id=str, audio_url=str, label=str)
LocalTrack = NamedTuple('Track', id=str, raw_path=str, label=str)


def get_local(row, OUTPUT_DIR_SILENCE) -> str:
    local_metadata: pd.DataFrame = pd.read_csv(OUTPUT_METADATA_PATH)
    local_metadata = local_metadata.drop_duplicates(subset='id', keep='first')

    track = LocalTrack._make(row)
    try:
        logger.info(f'Fetching {track.id} from metadata_raw.csv')

        original_format = track.raw_path.split('.')[-1]
        output_filename = f'{track.id}.ogg'
        return (
            track.raw_path,
            os.path.join(OUTPUT_DIR_SILENCE, output_filename),
            original_format,
            track,
        )

    except Exception as e:
        logger.error(
            f'Error finding raw file by metadata path, {e.with_traceback()}'
        )


def get_remote(
    track, AUDIO_BASE_PATH: str, OUTPUT_DIR_SILENCE: str, OUTPUT_DIR_RAW: str
) -> str:
    track = RemoteTrack._make(track)
    if 'http' in track.audio_url:
        audio_url = track.audio_url
    else:
        audio_url = AUDIO_BASE_PATH + track.audio_url

    original_format = track.audio_url.split('.')[-1]
    output_filename = track.id + '.' + original_format
    downloaded_path, _ = urlretrieve(
        audio_url, os.path.join(OUTPUT_DIR_RAW, output_filename)
    )

    return (
        downloaded_path,
        os.path.join(OUTPUT_DIR_SILENCE, track.id + '.ogg'),
        original_format,
        track,
    )
