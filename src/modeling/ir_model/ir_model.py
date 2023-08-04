import numpy as np
from abc import ABC, abstractmethod


class IRModel(ABC):
    @abstractmethod
    def fit(self, X: np.array, Y: np.array, **options) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.array, **options) -> np.array:
        ...

    @abstractmethod
    def predict_proba(self, X: np.array, **options) -> np.array:
        ...
