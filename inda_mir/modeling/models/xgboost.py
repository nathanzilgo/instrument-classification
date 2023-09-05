import numpy.typing as npt

from sklearn import preprocessing
from xgboost import XGBClassifier as XGBC

from inda_mir.modeling.models.base_model import BaseModel


class XGBClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = XGBC(**kwargs)
        self.le = preprocessing.LabelEncoder()

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        self.le = self.le.fit(y)
        self.model = self.model.fit(X, self.le.transform(y))

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.predict_proba(X)

    def _get_feature_importance(self) -> npt.NDArray:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.model.get_params()
