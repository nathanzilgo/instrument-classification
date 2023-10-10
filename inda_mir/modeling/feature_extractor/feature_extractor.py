import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, List

from tqdm import tqdm

from inda_mir.utils.logger import logger


class FeatureExtractor(ABC):
    @abstractmethod
    def _extract(
        self, file_path: str, **kwargs
    ) -> Dict[str, List[int | float]]:
        ...

    def extract(
        self,
        files: List[str],
        output_path: str,
        output_separator: str = ',',
        agg: bool = False,
        **kwargs,
    ):

        file_features_df = []
        for file in tqdm(
            files, desc='Extracting Features: ', total=len(files)
        ):
            try:
                features = self._extract(file, **kwargs)
                df = self.features_to_df(features, file, agg)
                file_features_df.append(df)
            except Exception as e:
                logger.error(
                    f'Error at feature_extractor.py at file: {file} - {e}'
                )
                pass
        try:
            unified_df = pd.concat(file_features_df)
            unified_df.to_csv(output_path, sep=output_separator, index=False)
        except Exception as e:
            logger.error(f'Error at feature_extractor.py on unified_df - {e}')

    def features_to_df(
        self, features: Dict[str, List[int | float]], file_path: str, agg=False
    ) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(features, orient='index').reset_index()
        df = df.melt(id_vars=['index'], var_name='frame')
        df = df.pivot_table(
            index='frame', columns='index', values='value', aggfunc='first'
        ).reset_index(drop=agg)

        if agg:
            df = self.__agg_features_df(df)

        df.insert(loc=0, column='filename', value=file_path)
        return df

    def __agg_features_df(self, df: pd.DataFrame):
        df = (
            df.agg(['max', 'mean', 'median', 'min', 'std', 'var'])
            .stack()
            .reset_index()
        )
        df['feature'] = df['level_0'] + '_' + df['index']
        df = df.drop(['level_0', 'index'], axis=1)
        df['index'] = 0
        df.columns = ['value', 'feature', 'index']
        df = (
            df.pivot_table(
                index='index',
                columns='feature',
                values='value',
                aggfunc='first',
            )
            .rename_axis('frame')
            .reset_index()
        )
        return df
