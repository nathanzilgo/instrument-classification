import numpy.typing as npt

import lightgbm as ltb

from inda_mir.modeling.models.base_model import BaseModel


class LightGBMClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ltb.LGBMClassifier(**kwargs)
        self.name = 'LightGBM'

    def _fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        self.model = self.model.fit(X, y)

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.predict_proba(X)

    def _get_feature_importance(self) -> npt.NDArray:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.model.get_params()
