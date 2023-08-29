import os
from inda_mir.modeling.feature_extractor import (
    FreesoundExtractor,
)
from inda_mir.utils.logger import logger


OUTPUT_DIR = './output-inda'
OUTPUT_DIR_SAMPLE = os.path.join(OUTPUT_DIR, 'sampled')
OUTPUT_DIR_FEATURE = os.path.join(OUTPUT_DIR, 'features_output')

feature_extractor = FreesoundExtractor()

samples = []
for sample_dir in os.scandir(OUTPUT_DIR_SAMPLE):
    for sample in os.scandir(sample_dir.name):
        samples.append(os.path.abspath(sample.path))

try:
    logger.info(f'Extracting samples!')

    features = feature_extractor.extract(samples, OUTPUT_DIR_FEATURE)

    logger.info(
        f'Features extracted successfully!\nSaved at {OUTPUT_DIR_FEATURE}'
    )
except:
    logger.error(f'Error on sample feature extraction')
