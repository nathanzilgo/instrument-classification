import os
from tempfile import NamedTemporaryFile
from unittest import mock
from zipfile import ZipFile

from inda_mir.utils.gcs_interface import upload_artifact, download_artifact
from inda_mir.utils.gcs_interface.artifact_type import ArtifactType


@mock.patch('os.path.exists')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_samples_artifact(
    upload_blob_mock: mock.Mock, path_exists_mock: mock.Mock
) -> None:
    path_exists_mock.return_value = True

    upload_artifact(
        artifact_type=ArtifactType.FEATURES, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        './output/features/test_destination',
        'features/test_destination',
    )


@mock.patch('os.path.exists')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_models_artifact(
    upload_blob_mock: mock.Mock, path_exists_mock: mock.Mock
) -> None:
    path_exists_mock.return_value = True

    upload_artifact(
        artifact_type=ArtifactType.MODEL, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        './output/models/test_destination',
        'models/test_destination',
    )


@mock.patch('os.path.exists')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_tts_artifact(
    upload_blob_mock: mock.Mock, path_exists_mock: mock.Mock
) -> None:
    path_exists_mock.return_value = True

    upload_artifact(
        artifact_type=ArtifactType.TTS, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        './output/train_test_split/test_destination',
        'train_test_splits/test_destination',
    )


@mock.patch('inda_mir.utils.gcs_interface.zip_dir')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_samples_artifact(
    upload_blob_mock: mock.Mock, zip_dir_mock: mock.Mock
) -> None:
    zipped_file = NamedTemporaryFile()
    zipped_filename = zipped_file.name
    zip_dir_mock.return_value = zipped_file

    upload_artifact(
        artifact_type=ArtifactType.SAMPLES, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    zip_dir_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples', zipped_filename, 'samples/test_destination'
    )


@mock.patch('inda_mir.utils.gcs_interface.zip_dir')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_raw_artifact(
    upload_blob_mock: mock.Mock, zip_dir_mock: mock.Mock
) -> None:
    zipped_file = NamedTemporaryFile()
    zipped_filename = zipped_file.name
    zip_dir_mock.return_value = zipped_file

    upload_artifact(
        artifact_type=ArtifactType.RAW, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    zip_dir_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples', zipped_filename, 'raw_tracks/test_destination'
    )


@mock.patch('inda_mir.utils.gcs_interface.zip_dir')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_metadata_artifact(
    upload_blob_mock: mock.Mock, zip_dir_mock: mock.Mock
) -> None:
    zipped_file = NamedTemporaryFile()
    zipped_filename = zipped_file.name
    zip_dir_mock.return_value = zipped_file

    upload_artifact(
        artifact_type=ArtifactType.METADATA, filename='test_destination'
    )

    upload_blob_mock.assert_called_once()
    zip_dir_mock.assert_called_once()
    upload_blob_mock.assert_called_once_with(
        'inda-mir-samples', zipped_filename, 'metadata/test_destination'
    )


@mock.patch('os.path.exists')
@mock.patch('inda_mir.utils.gcs_interface.upload_blob')
def test_upload_samples_artifact_path_not_exists(
    upload_blob_mock: mock.Mock, path_exists_mock: mock.Mock
) -> None:
    path_exists_mock.return_value = False

    upload_artifact(
        artifact_type=ArtifactType.FEATURES, filename='test_destination'
    )

    upload_blob_mock.assert_not_called()


@mock.patch('zipfile.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_features_artifact(
    download_blob_mock: mock.Mock, zipfile_mock: mock.Mock
) -> None:

    download_artifact(
        artifact_type=ArtifactType.FEATURES, filename='test_source'
    )

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        'features/test_source',
        './output/features/test_source',
    )
    zipfile_mock.extractall.assert_not_called()


@mock.patch('zipfile.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_tts_artifact(
    download_blob_mock: mock.Mock, zipfile_mock: mock.Mock
) -> None:

    download_artifact(artifact_type=ArtifactType.TTS, filename='test_source')

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        'train_test_splits/test_source',
        './output/train_test_split/test_source',
    )
    zipfile_mock.extractall.assert_not_called()


@mock.patch('zipfile.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_model_artifact(
    download_blob_mock: mock.Mock, zipfile_mock: mock.Mock
) -> None:

    download_artifact(artifact_type=ArtifactType.MODEL, filename='test_source')

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        'models/test_source',
        './output/models/test_source',
    )
    zipfile_mock.extractall.assert_not_called()


@mock.patch('os.remove')
@mock.patch('inda_mir.utils.gcs_interface.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_raw_artifact(
    download_blob_mock: mock.Mock,
    zipfile_mock: mock.Mock,
    remove_mock: mock.Mock,
) -> None:
    filepath = './output/raw_tracks/test_source.zip'

    download_artifact(
        artifact_type=ArtifactType.RAW, filename='test_source.zip'
    )

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples', 'raw_tracks/test_source.zip', filepath
    )

    zipfile_mock.return_value.extractall.assert_called_once_with(
        './output/raw_tracks'
    )
    remove_mock.assert_called_once_with(filepath)


@mock.patch('os.remove')
@mock.patch('inda_mir.utils.gcs_interface.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_samples_artifact(
    download_blob_mock: mock.Mock,
    zipfile_mock: mock.Mock,
    remove_mock: mock.Mock,
) -> None:
    filepath = './output/samples/test_source.zip'

    download_artifact(
        artifact_type=ArtifactType.SAMPLES, filename='test_source.zip'
    )

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        'samples/test_source.zip',
        filepath,
    )

    zipfile_mock.return_value.extractall.assert_called_once_with(
        './output/samples'
    )
    remove_mock.assert_called_once_with(filepath)


@mock.patch('os.remove')
@mock.patch('inda_mir.utils.gcs_interface.ZipFile')
@mock.patch('inda_mir.utils.gcs_interface.download_blob')
def test_download_metadata_artifact(
    download_blob_mock: mock.Mock,
    zipfile_mock: mock.Mock,
    remove_mock: mock.Mock,
) -> None:
    filepath = './output/metadata/test_source.zip'

    download_artifact(
        artifact_type=ArtifactType.METADATA, filename='test_source.zip'
    )

    download_blob_mock.assert_called_once()
    download_blob_mock.assert_called_once_with(
        'inda-mir-samples',
        'metadata/test_source.zip',
        filepath,
    )

    zipfile_mock.return_value.extractall.assert_called_once_with(
        './output/metadata'
    )
    remove_mock.assert_called_once_with(filepath)
