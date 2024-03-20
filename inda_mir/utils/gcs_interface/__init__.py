import os
from zipfile import ZipFile
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType
from scripts.util import zip_dir
from scripts.util.config import instrument_classification_config as icc
from scripts.util.gcloud import upload_blob, download_blob

TYPES_TO_REMOTE_DIR = {
    'raw': 'raw_tracks',
    'samples': 'samples',
    'features': 'features',
    'tts': 'train_test_splits',
    'model': 'models',
    'metadata': 'metadata',
    'metrics': 'metrics',
}

TYPES_TO_LOCAL_DIR = {
    'raw': icc['dirs']['RAW_TRACKS'],
    'samples': icc['dirs']['PROCESSED_SAMPLES'],
    'features': icc['dirs']['FEATURES'],
    'tts': icc['dirs']['TRAIN_TEST_SPLITS'],
    'metadata': icc['dirs']['METADATA'],
    'model': icc['dirs']['GRID_MODEL'],
    'metrics': icc['dirs']['METRICS'],
}

BUCKET_NAME = icc['gcs']['BUCKET_NAME']


def upload_artifact(artifact_type: ArtifactType, filename: str):
    if artifact_type.value not in ['raw', 'samples', 'metadata']:
        filepath = os.path.join(
            TYPES_TO_LOCAL_DIR[artifact_type.value], filename
        )
        destination_path = os.path.join(
            TYPES_TO_REMOTE_DIR[artifact_type.value], filename
        )
        if os.path.exists(filepath):
            upload_blob(BUCKET_NAME, filepath, destination_path)
    else:
        file = zip_dir(TYPES_TO_LOCAL_DIR[artifact_type.value])
        destination_path = os.path.join(
            TYPES_TO_REMOTE_DIR[artifact_type.value], filename
        )
        upload_blob(BUCKET_NAME, file.name, destination_path)


def download_artifact(artifact_type: ArtifactType, filename: str):
    source_path = os.path.join(
        TYPES_TO_REMOTE_DIR[artifact_type.value], filename
    )
    filepath = os.path.join(TYPES_TO_LOCAL_DIR[artifact_type.value], filename)
    download_blob(BUCKET_NAME, source_path, filepath)

    if artifact_type.value in ['raw', 'samples', 'metadata']:
        z = ZipFile(filepath)
        z.extractall(TYPES_TO_LOCAL_DIR[artifact_type.value])
        os.remove(filepath)
