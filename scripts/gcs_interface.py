import argparse

from inda_mir.utils.gcs_interface import download_artifact, upload_artifact
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType


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
        upload_artifact(ArtifactType(args.type), args.filename)
    else:
        download_artifact(ArtifactType(args.type), args.filename)
