import numpy as np
from abc import ABC, abstractmethod


class IRModel(ABC):
    @abstractmethod
    def fit(self, X: np.array, Y: np.array, **kwargs) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.array, **kwargs) -> np.array:
        ...

    @abstractmethod
    def predict_proba(self, X: np.array, **kwargs) -> np.array:
        ...
