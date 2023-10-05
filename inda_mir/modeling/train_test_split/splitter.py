import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List
from .dataset_interface import DatasetInterface


class TrainTestSplitter(ABC):
    @abstractmethod
    def _split(
        self, track_data: pd.DataFrame, **kwargs
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]
    ]:
        ...

    def split(self, track_data: pd.DataFrame, **kwargs) -> DatasetInterface:
        return DatasetInterface(*self._split(track_data, **kwargs))
