import os
import pandas as pd
import tqdm

from inda_mir.processing_pipelines import sample_and_filter_silence

from scripts.util.config import instrument_classification_config as icc

TRACKS_DIR = icc['dirs']['RAW_TRACKS']
OUTPUT_DIR = icc['dirs']['PROCESSED_SAMPLES']

TRACK_METADATA_PATH = icc['metadata']['RAW_TRACKS']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']

SAMPLE_DURATION = icc['params']['process_tracks']['SAMPLE_DURATION']
SILENCE_THRESHOLD = icc['params']['process_tracks']['SILENCE_THRESHOLD']
SILENCE_DURATION = icc['params']['process_tracks']['SILENCE_DURATION']
SILENCE_PERCENTAGE = icc['params']['process_tracks']['SILENCE_PERCENTAGE']
SAMPLE_PROPORTION = icc['params']['process_tracks']['SAMPLE_PROPORTION']

tracks = pd.read_csv(TRACK_METADATA_PATH)[['track_id', 'label']]


def process_tracks(
    tracks: pd.DataFrame = tracks,
    sample_duration: int = SAMPLE_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: int = SILENCE_DURATION,
    silence_percentage: float = SILENCE_PERCENTAGE,
    sample_proportion: float = SAMPLE_PROPORTION,
):
    metadata = []
    for track_path in tqdm.tqdm(os.listdir(TRACKS_DIR)):
        track_id, track_format = track_path.split('.')

        full_track_path = os.path.join(TRACKS_DIR, track_path)
        sample_dir_path = os.path.join(OUTPUT_DIR, track_id)

        silent_samples_path = sample_and_filter_silence(
            track_path=full_track_path,
            input_format=track_format,
            output_dir=sample_dir_path,
            output_basename=track_id,
            sample_duration=sample_duration * 1000,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            silence_percentage=silence_percentage,
            sample_proportion=sample_proportion,
        )

        metadata.extend(
            [
                (track_id, full_sample_path)
                for full_sample_path in silent_samples_path
            ]
        )
    samples = pd.DataFrame(metadata, columns=['track_id', 'sample_path'])
    pd.merge(samples, tracks, on='track_id').to_csv(
        SAMPLE_METADATA_PATH, index=False
    )


if __name__ == '__main__':
    process_tracks(
        tracks=tracks,
        sample_duration=SAMPLE_DURATION,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        silence_percentage=SILENCE_PERCENTAGE,
        sample_proportion=SAMPLE_PROPORTION,
    )
