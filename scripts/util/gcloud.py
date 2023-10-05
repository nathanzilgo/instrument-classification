import pandas as pd

from google.cloud import bigquery
from google.cloud import storage


def exists_blob(bucket_name: str, blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return storage.Blob(bucket=bucket, name=blob_name).exists(storage_client)


def download_blob(
    bucket_name: str, source_blob_name: str, destination_file_name: str
):
    """
    Downloads a blob from the bucket.

    Args:
        bucket_name (str): The ID of your GCS bucket.
        source_blob_name (str): The ID of your GCS object.
        destination_file_name: The path to which the file should be downloaded.

    """
    if exists_blob(bucket_name, source_blob_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.

    Args:
        bucket_name (str): The ID of your GCS bucket.
        source_file_name (str): The path to your file to upload
        destination_blob_name: The ID of your GCS object
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def query_to_df(query: str) -> pd.DataFrame:
    client = bigquery.Client()
    query_job = client.query(query)  # Make an API request.

    # Wait for the query to complete
    query_job.result()
    query_df: pd.DataFrame = query_job.to_dataframe()
    return query_df
