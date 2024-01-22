import pandas as pd

from inda_mir.modeling.feature_extractor import (
    EssentiaExtractor,
)

from scripts.util.config import instrument_classification_config as icc

OUTPUT_PATH = icc['outputs']['FEATURES']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']


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
