import os

import pandas as pd
import numpy as np

from inda_mir.modeling.train_test_split import RandomTrainTestSplit

OUTDIR = './output-inda/train_test_split/'
OUTFILE = os.path.join(OUTDIR, 'random_split.npz')

os.makedirs(OUTDIR, exist_ok=True)

METADATA_PATH = './output-inda/metadata/metadata.csv'
FEATURES_PATH = './output-inda/features_output/freesound_features.csv'

metadata_df = pd.read_csv(METADATA_PATH)
features_df = pd.read_csv(FEATURES_PATH)

r = RandomTrainTestSplit()
X_train, X_test, y_train, y_test, labels = r.split(
    metadata_df, features_df, train_size=0.7, random_state=0
)

np.savez(
    OUTFILE,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    labels=labels,
)
