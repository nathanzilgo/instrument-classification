import numpy.typing as npt
import pickle
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> None:
        ...

    @abstractmethod
    def predict(self, X: npt.NDArray, **kwargs) -> npt.NDArray:
        ...

    @abstractmethod
    def predict_proba(self, X: npt.NDArray, **kwargs) -> npt.NDArray:
        ...

    @abstractmethod
    def _get_feature_importance(self, **kwargs) -> npt.NDArray:
        ...

    @abstractmethod
    def get_params(self, **kwargs) -> Dict[str, Any]:
        ...

    def get_feature_importance(
        self, feature_names: List[str] = None
    ) -> Dict[str, float]:
        feature_importances = self._get_feature_importance()

        if feature_names is None:
            feature_names = [
                f'feature {i}' for i in range(len(feature_importances))
            ]

        if len(feature_names) != len(feature_importances):
            raise IndexError(
                'Feature names array size differ from number of features'
            )

        return {
            feature_names[i]: feature_importances[i]
            for i in range(len(feature_importances))
        }

    def save_model(self, path, model_name) -> dict:
        pickle.dump(
            self.model, open(os.path.join(path, model_name) + '.pkl', 'wb')
        )
        json.dump(
            self.get_params(),
            open(os.path.join(path, model_name) + '.json', 'w'),
        )
