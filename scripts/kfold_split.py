import os

import pandas as pd
import pickle

from inda_mir.modeling.train_test_split import StratifiedKFoldSplitter

OUTDIR = './output-inda/train_test_split/'
OUTFILE = 'kfold_split'

os.makedirs(OUTDIR, exist_ok=True)

METADATA_PATH = './output-inda/metadata/metadata.csv'
FEATURES_PATH = './output-inda/features_output/freesound_features.csv'

metadata_df = pd.read_csv(METADATA_PATH)
features_df = pd.read_csv(FEATURES_PATH)

kfold = StratifiedKFoldSplitter()
r = kfold.split(metadata_df, features_df, k=10, random_state=0)

pickle.dump(r, open(os.path.join(OUTDIR, OUTFILE) + '.data', 'wb'))
