import pandas as pd

from extraction.essentia_extractor import (
    EssentiaExtractor,
)
from settings import settings

OUTPUT_PATH = settings.OUTPUT_PATH
SAMPLE_METADATA_PATH = settings.SAMPLE_OUTPUT_PATH


def feature_extraction(
    metadata_path: str = SAMPLE_METADATA_PATH,
    output_path: str = OUTPUT_PATH,
    retrain: bool = False,
    trained_features: pd.DataFrame = None,
) -> None:
    samples = pd.read_csv(metadata_path)
    feature_extractor = EssentiaExtractor()
    feature_extractor.extract(samples['sample_path'], output_path)
    features = pd.read_csv(output_path)
    features = pd.merge(
        left=features,
        right=samples,
        left_on='filename',
        right_on='sample_path',
    )
    if retrain:
        features = pd.concat([trained_features, features])
    features.drop(['sample_path'], axis=1).to_csv(output_path, index=False)


if __name__ == '__main__':
    feature_extraction()
