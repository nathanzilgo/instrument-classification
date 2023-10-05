import os
import pandas as pd

from inda_mir.audio_processing import SampleOperation, FFmpegSilenceDetector

from scripts.util.config import instrument_classification_config as icc

TRACKS_DIR = icc['dirs']['RAW_TRACKS']
OUTPUT_DIR = icc['dirs']['PROCESSED_SAMPLES']

TRACK_METADATA_PATH = icc['metadata']['RAW_TRACKS']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']

SAMPLE_DURATION = icc['params']['process_tracks']['SAMPLE_DURATION']
SILENCE_THRESHOLD = icc['params']['process_tracks']['SILENCE_THRESHOLD']
SILENCE_DURATION = icc['params']['process_tracks']['SILENCE_DURATION']
SILENCE_PERCENTAGE = icc['params']['process_tracks']['SILENCE_PERCENTAGE']

tracks = pd.read_csv(TRACK_METADATA_PATH)[['track_id', 'label']]

metadata = []
for track_path in os.listdir(TRACKS_DIR):
    track_id, track_format = track_path.split('.')

    full_track_path = os.path.join(TRACKS_DIR, track_path)
    sample_dir_path = os.path.join(OUTPUT_DIR, track_id)

    if not os.path.exists(sample_dir_path):

        os.makedirs(sample_dir_path)

        SampleOperation.apply(
            audio_path=full_track_path,
            input_format=track_format,
            output_dir=sample_dir_path,
            output_basename=track_id,
            sample_duration=SAMPLE_DURATION * 1000,
        )

    for sample_path in os.listdir(sample_dir_path):
        full_sample_path = os.path.join(sample_dir_path, sample_path)
        if not FFmpegSilenceDetector.apply(
            full_sample_path,
            audio_duration=SAMPLE_DURATION,
            silence_threshold=SILENCE_THRESHOLD,
            min_silence_duration=SILENCE_DURATION,
            silence_percentage_threshold=SILENCE_PERCENTAGE,
        ):
            metadata.append((track_id, full_sample_path))

samples = pd.DataFrame(metadata, columns=['track_id', 'sample_path'])
pd.merge(samples, tracks, on='track_id').to_csv(
    SAMPLE_METADATA_PATH, index=False
)
