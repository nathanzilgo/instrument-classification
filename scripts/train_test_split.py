import pandas as pd

from inda_mir.modeling.train_test_split import RandomTrainTestSplit
from scripts.util.config import instrument_classification_config as icc

OUTFILE = icc['outputs']['TRAIN_TEST_SPLITS']
FEATURES_PATH = icc['outputs']['FEATURES_EXTRACTED']


def train_test_split(
    features_path=FEATURES_PATH, train_size=0.7, random_state=0
):
    features_df = pd.read_csv(features_path)

    r = RandomTrainTestSplit()
    d = r.split(features_df, train_size=train_size, random_state=random_state)
    d.save(OUTFILE)


if __name__ == '__main__':
    train_test_split(FEATURES_PATH, train_size=0.7, random_state=0)
