import pickle

from typing import List

from inda_mir.modeling.models import BaseModel
from inda_mir.modeling.train_test_split import DatasetInterface


def load_model(path) -> BaseModel:
    return pickle.load(open(path, 'rb'))


def load_data_partition(path) -> DatasetInterface:
    return pickle.load(open(path, 'rb'))


def load_kfold_partition(path) -> List[DatasetInterface]:
    return pickle.load(open(path, 'rb'))
