import pandas as pd

from inda_mir.modeling.train_test_split import RandomTrainTestSplit
from scripts.util.config import instrument_classification_config as icc

OUTFILE = icc['outputs']['TRAIN_TEST_SPLITS']

FEATURES_PATH = icc['outputs']['FEATURES']

features_df = pd.read_csv(FEATURES_PATH)

r = RandomTrainTestSplit()
d = r.split(features_df, train_size=0.7, random_state=0)

d.save(OUTFILE)
