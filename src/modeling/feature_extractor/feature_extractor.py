from abc import ABC, abstractmethod
from typing import Dict, List, Union
import pandas as pd


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        file_path: str,
        output_path: str,
        output_separator: str = ',',
        **kwargs
    ) -> any:
        ...

    def write_features(
        self,
        features: Dict[str, List[Union[int, float]]],
        file_path: str,
        output_path: str,
        output_separator: str = ',',
    ):
        df = pd.DataFrame.from_dict(features, orient='index').reset_index()
        df = df.melt(id_vars=['index'], var_name='frame')
        df = df.pivot_table(
            index='frame', columns='index', values='value', aggfunc='first'
        ).reset_index()
        df.insert(loc=0, column='filename', value=file_path)
        df.to_csv(output_path, sep=output_separator, index=False)
