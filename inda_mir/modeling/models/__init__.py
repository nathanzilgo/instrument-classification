import pickle

from .base_model import BaseModel
from .random_forest import RandomForestClassifier
from .svm import SVMClassifier
from .xgboost import XGBClassifier
from .lgbm import LightGBMClassifier


def load_model(path) -> BaseModel:
    return pickle.load(open(path, 'rb'))
