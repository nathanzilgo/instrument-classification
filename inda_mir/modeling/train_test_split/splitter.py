import pandas as pd
from abc import ABC, abstractmethod


class TrainTestSplitter(ABC):
    @abstractmethod
    def split(
        self,
        track_metadata: pd.DataFrame,
        track_features: pd.DataFrame,
        **kwargs
    ) -> (pd.DataFrame, pd.DataFrame):
        ...
