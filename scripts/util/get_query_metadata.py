import os
from google.cloud import bigquery

from inda_mir.utils.logger import logger


OUTPUT_DIR = './output-inda/metadata'
RESULT_FILE = 'remote_tracks_query.csv'
RESULT_PATH = os.path.join(OUTPUT_DIR, RESULT_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = bigquery.Client()
query_job = client.query(
    'SELECT * from track_classification.track_labels_filtered;'
)  # Make an API request.

# Wait for the query to complete
query_job.result()

# Export the query results to a CSV file
query_df = query_job.to_dataframe()
query_df.to_csv(RESULT_PATH, index=False)

logger.info(f'Query results saved to {RESULT_PATH}')
