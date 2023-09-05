import os

import pandas as pd

from inda_mir.modeling.train_test_split import RandomTrainTestSplit

OUTDIR = './output-inda/train_test_split/'
OUTFILE = 'random_split'

os.makedirs(OUTDIR, exist_ok=True)

METADATA_PATH = './output-inda/metadata/metadata.csv'
FEATURES_PATH = './output-inda/features_output/freesound_features.csv'

metadata_df = pd.read_csv(METADATA_PATH)
features_df = pd.read_csv(FEATURES_PATH)

r = RandomTrainTestSplit()
d = r.split(metadata_df, features_df, train_size=0.7, random_state=0)

d.save(OUTDIR, OUTFILE)
