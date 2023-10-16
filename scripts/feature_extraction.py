import os
import pandas as pd

from inda_mir.modeling.feature_extractor import (
    FreesoundExtractor,
    EssentiaExtractor,
)

from scripts.util.config import instrument_classification_config as icc

OUTPUT_PATH = icc['outputs']['FEATURES']
SAMPLE_METADATA_PATH = icc['metadata']['PROCESSED_SAMPLES']

feature_extractor = EssentiaExtractor()

samples = pd.read_csv(SAMPLE_METADATA_PATH)

feature_extractor.extract(samples['sample_path'], OUTPUT_PATH)
features = pd.read_csv(OUTPUT_PATH)
features = pd.merge(
    left=features, right=samples, left_on='filename', right_on='sample_path'
)
features.drop(['sample_path'], axis=1).to_csv(OUTPUT_PATH, index=False)
