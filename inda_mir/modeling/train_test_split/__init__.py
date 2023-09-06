from .random_splitter import RandomTrainTestSplit
from .kfold_splitter import StratifiedKFoldSplitter
from .dataset_interface import DatasetInterface

import pickle


def load_data(path) -> DatasetInterface:
    return pickle.load(open(path, 'rb'))
