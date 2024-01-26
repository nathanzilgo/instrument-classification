from typing import Tuple

from unittest import mock

import pandas as pd

from scripts.util.gcloud import (
    exists_blob,
    download_blob,
    upload_blob,
    query_to_df,
)


def _build_gcs_mock(
    mock_storage: mock.Mock, exists: bool = True
) -> Tuple[mock.Mock, mock.Mock, mock.Mock]:
    mock_storage.Blob.return_value.exists.return_value = exists
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    blob_mock = mock.Mock()

    mock_bucket.blob.return_value = blob_mock
    blob_mock.download_to_filename = mock.Mock()
    mock_gcs_client.bucket.return_value = mock_bucket

    return mock_gcs_client, mock_bucket, blob_mock


@mock.patch('scripts.util.gcloud.storage')
def test_exists_blob_true(mock_storage: mock.Mock) -> None:
    _build_gcs_mock(mock_storage)
    assert exists_blob('bucket', 'blob')


@mock.patch('scripts.util.gcloud.storage')
def test_exists_blob_false(mock_storage: mock.Mock) -> None:
    _build_gcs_mock(mock_storage, False)
    assert not exists_blob('bucket', 'blob')


@mock.patch('scripts.util.gcloud.storage')
def test_download_not_exists_blob(mock_storage: mock.Mock) -> None:
    mock_gcs_client, mock_bucket, blob_mock = _build_gcs_mock(
        mock_storage, False
    )

    download_blob('bucket', 'blob', 'dest')

    mock_gcs_client.bucket.assert_called_once_with('bucket')
    mock_bucket.blob.assert_not_called()
    blob_mock.download_to_filename.assert_not_called()


@mock.patch('scripts.util.gcloud.storage')
def test_download_exists_blob(mock_storage: mock.Mock) -> None:
    mock_gcs_client, mock_bucket, blob_mock = _build_gcs_mock(mock_storage)

    download_blob('bucket', 'blob', 'dest')

    mock_gcs_client.bucket.assert_called_with('bucket')
    assert len(mock_gcs_client.bucket.call_args_list) == 2
    mock_bucket.blob.assert_called_with('blob')
    blob_mock.download_to_filename.assert_called_once_with('dest')


@mock.patch('scripts.util.gcloud.storage')
def test_upload_blob(mock_storage: mock.Mock) -> None:
    mock_gcs_client, mock_bucket, blob_mock = _build_gcs_mock(mock_storage)

    upload_blob('bucket', 'blob', 'dest')

    mock_gcs_client.bucket.assert_called_with('bucket')
    assert len(mock_gcs_client.bucket.call_args_list) == 1
    mock_bucket.blob.assert_called_with('dest')
    blob_mock.upload_from_filename.assert_called_once_with('blob')


@mock.patch('scripts.util.gcloud.bigquery')
def test_query_to_df(mock_big_query: mock.Mock) -> None:
    mock_client = mock_big_query.Client.return_value
    mock_query_df = mock.Mock()
    mock_client.query.return_value = mock_query_df
    mock_query_df.result.return_value = None

    mock_data = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}
    mock_query_df.to_dataframe.return_value = pd.DataFrame(mock_data)

    result_df = query_to_df('SELECT * FROM your_table')

    mock_client.query.assert_called_once_with('SELECT * FROM your_table')

    expected_df = pd.DataFrame(mock_data)
    pd.testing.assert_frame_equal(result_df, expected_df)
