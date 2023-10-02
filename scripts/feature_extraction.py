import os
from inda_mir.modeling.feature_extractor import (
    FreesoundExtractor,
)
from inda_mir.utils.logger import logger


def extract(output_dir):
    OUTPUT_DIR = output_dir
    OUTPUT_DIR_SAMPLE = os.path.join(OUTPUT_DIR, 'sampled')
    OUTPUT_DIR_FEATURE = os.path.join(OUTPUT_DIR, 'features_output')

    if not os.path.exists(OUTPUT_DIR_SAMPLE):
        logger.error('Sample directory not found.')
        raise FileNotFoundError

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FEATURE, exist_ok=True)
    feature_extractor = FreesoundExtractor()

    samples = []
    for sample_dir in os.scandir(OUTPUT_DIR_SAMPLE):
        for sample in os.scandir(sample_dir.path):
            samples.append(os.path.abspath(sample.path))

    try:
        logger.info(f'Extracting samples! {len(samples)} samples found.')

        feature_extractor.extract(
            samples, os.path.join(OUTPUT_DIR_FEATURE, 'freesound_features.csv')
        )

        logger.info(
            f'Features extracted successfully!\nSaved at {OUTPUT_DIR_FEATURE}'
        )
    except Exception as e:
        logger.error(
            f'Error on sample feature extraction - {e}, {e.with_traceback()}'
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract freesound features')
    parser.add_argument(
        '-dir', type=str, help='Output directory', required=False
    )
    args = parser.parse_args()
    output_dir = args.dir
    if output_dir is None:
        output_dir = './output-inda'

    extract(output_dir)
