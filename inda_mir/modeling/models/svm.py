import numpy.typing as npt

from typing import Dict, List

from sklearn.svm import SVC

from inda_mir.modeling.models.base_model import BaseModel


class SVMClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = SVC(**kwargs)

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        self.model = self.model.fit(X, y)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.predict(X)

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        return self.model.predict_proba(X)

    def get_feature_importance(
        self, feature_names: List[str] = None
    ) -> Dict[str, float]:
        feature_importances = None
        if feature_names is None:
            feature_importances = {
                f'feature {i}': self.model.coef_[i]
                for i in range(len(self.model.coef_))
            }
        elif len(feature_names) != len(self.model.coef_):
            raise IndexError(
                'Feature names array size differ from number of features'
            )
        else:
            feature_importances = {
                feature_names[i]: self.model.coef_[i]
                for i in range(len(self.model.coef_))
            }
        return feature_importances
