import os
from inda_mir.utils.logger import logger
from zipfile import ZipFile
from google.cloud import storage   # type: ignore

OUTPUT_DIR = './output-inda'
OUTPUTS_DIR_NAMES = ['raw', 'silenced', 'sampled', 'features_output']
BUCKET_NAME = 'inda-mir-samples'
BLOB_NAME = 'outputs-inda-samples'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.blob(BLOB_NAME)

for directory in OUTPUTS_DIR_NAMES:
    current_dir = os.path.join(OUTPUT_DIR, directory)

    logger.info(f'Zipping {current_dir} directory!')
    try:
        with ZipFile(f'./{directory}.zip', 'w') as zipped_result:
            for file in os.listdir(current_dir):
                zipped_result.write(os.path.abspath(file))

        blob.upload_from_filename(f'./{directory}.zip')
        os.remove(f'./{directory}.zip')

    except:
        logger.error(f'Error on uploading {current_dir} to the bucket!')
