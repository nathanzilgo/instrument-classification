import numpy as np
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self, truth: np.array, prediction: np.array, **options
    ) -> any:
        ...
