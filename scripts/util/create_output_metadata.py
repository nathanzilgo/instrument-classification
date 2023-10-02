from typing import List
import pandas as pd

OUTPUT_DIR_METADATA = './output-inda/metadata'


def create_metadata(metadata: List, name: str):
    df = pd.DataFrame.from_records(metadata, index=range(0, len(metadata)))
    df.to_csv(
        f'{OUTPUT_DIR_METADATA}/metadata_{name}.csv',
        index=False,
    )
