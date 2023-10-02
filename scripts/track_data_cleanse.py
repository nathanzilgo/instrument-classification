import os
import pandas as pd
from typing import NamedTuple

from inda_mir.audio_processing import SampleOperation, SilenceFilter
from inda_mir.utils.logger import logger
from scripts.util.create_output_metadata import create_metadata
from scripts.util.get_tracks import get_local, get_remote


Track = NamedTuple('Track', id=str, audio_url=str, label=str)

OUTPUT_DIR = './output-inda'
OUTPUT_DIR_RAW = os.path.join(OUTPUT_DIR, 'raw')
OUTPUT_DIR_SILENCE = os.path.join(OUTPUT_DIR, 'silenced')
OUTPUT_DIR_SAMPLE = os.path.join(OUTPUT_DIR, 'sampled')
OUTPUT_DIR_METADATA = os.path.join(OUTPUT_DIR, 'metadata')
OUTPUT_FORMAT = 'ogg'
SAMPLE_DURATION = 10000
AUDIO_BASE_PATH = 'https://uploads.storage.inda.band/'
REMOTE_QUERY_PATH = './output-inda/metadata/remote_tracks_query.csv'
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_DIR_METADATA, 'metadata_raw.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)
os.makedirs(OUTPUT_DIR_SAMPLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_SILENCE, exist_ok=True)
os.makedirs(OUTPUT_DIR_METADATA, exist_ok=True)

remote = True

metadata_raw = []
metadata_silenced = []
metadata_sampled = []


def make_track(row, remote=True):
    if remote:
        return get_remote(
            row,
            AUDIO_BASE_PATH,
            OUTPUT_DIR_SILENCE,
            OUTPUT_DIR_RAW,
        )
    else:
        return get_local(row, OUTPUT_DIR_SILENCE)


def make_raw():
    if os.path.exists(OUTPUT_METADATA_PATH):
        # Read from existing downloaded tracks
        remote = False
        tracks_df = pd.read_csv(OUTPUT_METADATA_PATH)
    else:
        # Download tracks from bigquery job
        remote = True
        try:
            tracks_df = pd.read_csv(REMOTE_QUERY_PATH)
        except Exception as e:
            raise Exception(
                f'No remote tracks found, run the bigquery job first. (make query) {e.with_traceback()}'
            )

    for i, row in enumerate(tracks_df.itertuples(index=False)):
        logger.info(f'Making raw track: {i+1}/{len(tracks_df)}')
        downloaded_path, output_filename, audio_url_format, track = make_track(
            row, remote
        )
        metadata_raw.append(
            {
                'id': track.id,
                'raw_path': os.path.abspath(downloaded_path),
                'label': track.label,
            }
        )

    create_metadata(metadata_raw, 'raw')


def track_cleanse(
    MIN_SILENCE_LEN=200,
    SILENCE_THRESHOLD=-30,
    KEEP_SILENCE=20,
    OUTPUT_DIR=OUTPUT_DIR,
):
    # Create raw tracks if not exists
    if len(os.listdir(OUTPUT_DIR_RAW)) == 0 or not os.path.exists(
        OUTPUT_METADATA_PATH
    ):
        make_raw()

    # Read from existing downloaded tracks
    raw_tracks = pd.read_csv(OUTPUT_METADATA_PATH)

    try:
        for i, row in enumerate(raw_tracks.itertuples(index=False)):
            try:
                (
                    downloaded_path,
                    output_filename,
                    audio_url_format,
                    track,
                ) = make_track(row, False)

                logger.info(f'Applying silence filter to {track.id}')

                output_silence_path = SilenceFilter.apply(
                    audio_path=downloaded_path,
                    input_format=audio_url_format,
                    output_format=OUTPUT_FORMAT,
                    output_path=output_filename,
                    min_silence_len=MIN_SILENCE_LEN,
                    silence_thresh=SILENCE_THRESHOLD,
                    keep_silence=KEEP_SILENCE,
                )

                track_sample_dir = os.path.join(OUTPUT_DIR_SAMPLE, track.id)
                output_filename = os.path.join(track_sample_dir, track.id)
                os.makedirs(track_sample_dir)

                logger.info(f'Applying Sample operations to {track.id}')

                SampleOperation.apply(
                    audio_path=output_silence_path,
                    input_format=OUTPUT_FORMAT,
                    output_format=OUTPUT_FORMAT,
                    sample_duration=SAMPLE_DURATION,
                    output_path=output_filename,
                )

                for sample_path in os.scandir(track_sample_dir):
                    metadata_sampled.append(
                        {
                            'id': track.id,
                            'sample_path': os.path.abspath(sample_path.path),
                            'label': track.label,
                        }
                    )

                metadata_silenced.append(
                    {
                        'id': track.id,
                        'sample_path': os.path.abspath(sample_path.path),
                        'label': track.label,
                        'silenced_path': os.path.abspath(output_silence_path),
                        'min_silence_len': MIN_SILENCE_LEN,
                        'threshold': SILENCE_THRESHOLD,
                        'keep_silence': KEEP_SILENCE,
                    }
                )

                logger.info(
                    f'Success on track {track.id}, samples saved at {track_sample_dir}'
                )
                logger.info(f'Processed {i+1} out of {len(raw_tracks)} tracks')
            except Exception as e:
                logger.error(
                    f'Error on track {track.id}, {e.with_traceback()}'
                )

    except Exception as e:
        logger.error(f'Error on track {row.id}, {e.with_traceback()}')

    create_metadata(metadata_silenced, 'silenced')
    create_metadata(metadata_sampled, 'sampled')
    logger.info(f'Success generating metadata files at {OUTPUT_DIR_METADATA}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Cleanse with Silence Filter Params'
    )
    parser.add_argument(
        '--len', type=int, help='MIN_SILENCE_LEN, Default: 100'
    )
    parser.add_argument(
        '--thr', type=int, help='SILENCE_THRESHOLD, Default: -45'
    )
    parser.add_argument('--keep', type=int, help='KEEP_SILENCE, Default: 30')

    args: argparse.Namespace = parser.parse_args()
    if args.len and args.thr and args.keep:
        track_cleanse(args.len, args.thr, args.keep)
    else:
        track_cleanse()
