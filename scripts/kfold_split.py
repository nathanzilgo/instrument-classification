import pandas as pd
import pickle

from inda_mir.modeling.train_test_split import StratifiedKFoldSplitter
from scripts.util.config import instrument_classification_config as icc

OUTFILE = icc['outputs']['TRAIN_TEST_SPLITS']

FEATURES_PATH = './output/features/freesound_features.csv'

features_df = pd.read_csv(FEATURES_PATH)

kfold = StratifiedKFoldSplitter()
r = kfold.split(features_df, k=10, random_state=0)

pickle.dump(r, open(OUTFILE + '.data', 'wb'))
