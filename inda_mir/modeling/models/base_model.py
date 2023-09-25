import numpy as np
import numpy.typing as npt
import pickle
import json
import os
from abc import ABC, abstractmethod
from sklearn import preprocessing
from typing import Dict, List, Any


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model = None
        self.le = preprocessing.LabelEncoder()

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> None:
        self.le = self.le.fit(y)
        self.classes_ = list(self.le.classes_) + ['other']
        self._fit(X, self.le.transform(y), **kwargs)

    @abstractmethod
    def _fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> None:
        ...

    def predict(
        self, X: npt.NDArray, threshold: float = 0.5, **kwargs
    ) -> npt.NDArray:
        probs = self.predict_proba(X, **kwargs)
        probs_above_threshold = probs >= threshold
        probs *= probs_above_threshold
        max_pos = np.argmax(probs, axis=1)
        max_value = np.max(probs, axis=1)
        mapping_function = (
            lambda x, y: self.le.inverse_transform([x])[0]
            if y != 0
            else 'other'
        )
        mapping_function = np.vectorize(mapping_function)
        return mapping_function(max_pos, max_value)

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
        pickle.dump(self, open(os.path.join(path, model_name) + '.pkl', 'wb'))
        json.dump(
            self.get_params(),
            open(os.path.join(path, model_name) + '.json', 'w'),
        )
