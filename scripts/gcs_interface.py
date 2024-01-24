import os
import argparse

from zipfile import ZipFile

from scripts.util import zip_dir
from scripts.util.artifact_type import ArtifactType
from scripts.util.gcloud import upload_blob, download_blob
from scripts.util.config import instrument_classification_config as icc

TYPES_TO_REMOTE_DIR = {
    'raw': 'raw_tracks',
    'samples': 'samples',
    'features': 'features',
    'tts': 'train_test_splits',
    'model': 'models',
    'metadata': 'metadata',
}

TYPES_TO_LOCAL_DIR = {
    'raw': icc['dirs']['RAW_TRACKS'],
    'samples': icc['dirs']['PROCESSED_SAMPLES'],
    'features': icc['dirs']['FEATURES'],
    'tts': icc['dirs']['TRAIN_TEST_SPLITS'],
    'metadata': icc['dirs']['METADATA'],
    'model': './models',
}

BUCKET_NAME = icc['gcs']['BUCKET_NAME']


def upload_interface(artifact_type: ArtifactType, filename: str):
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


def download_interface(artifact_type: ArtifactType, filename: str):
    source_path = os.path.join(
        TYPES_TO_REMOTE_DIR[artifact_type.value], filename
    )
    filepath = os.path.join(TYPES_TO_LOCAL_DIR[artifact_type.value], filename)
    download_blob(BUCKET_NAME, source_path, filepath)

    if artifact_type.value in ['raw', 'samples', 'metadata']:
        z = ZipFile(filepath)
        z.extractall(TYPES_TO_LOCAL_DIR[artifact_type.value])
        os.remove(filepath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='GCSI', description='Google Cloud Storage Interface'
    )

    parser.add_argument(
        '-o',
        '--operation',
        dest='operation',
        choices=['upload', 'download'],
        default='upload',
    )
    parser.add_argument(
        '-t',
        '--type',
        dest='type',
        choices=['raw', 'samples', 'features', 'tts', 'model', 'metadata'],
        required=True,
    )
    parser.add_argument('-f', '--filename', dest='filename', required=False)

    args = parser.parse_args()

    if args.operation == 'upload':
        upload_interface(ArtifactType(args.type), args.filename)
    else:
        download_interface(ArtifactType(args.type), args.filename)
