import pandas as pd
import numpy as np
import pickle

from abc import ABC, abstractmethod
from typing import Dict, List
from numpy.typing import NDArray

from tqdm import tqdm

from utils.logger import logger


class FeatureExtractor(ABC):
    @abstractmethod
    def _extract(
        self, file_path: str, **kwargs
    ) -> Dict[str, List[int | float]]:
        ...

    def extract(
        self,
        files: List[str],
        output_path: str = '',
        output_separator: str = ',',
        save_df: bool = True,
        agg: bool = False,
        checkpoint: bool = False,
        **kwargs,
    ) -> NDArray:

        features = []
        files_extracted = []
        for file in tqdm(
            files, desc='Extracting Features: ', total=len(files)
        ):
            try:
                extracted_features = self._extract(file, **kwargs)
                features.append(extracted_features)
                files_extracted.append(file)
            except Exception as e:
                logger.error(
                    f'Error at feature_extractor.py at file: {file} - {e}'
                )
                pass

        if checkpoint:
            features_data = {'files': files_extracted, 'features': features}
            pickle.dump(features_data, open(output_path + '.checkpoint', 'wb'))

        if save_df:
            self.save_features_as_df(
                files_extracted, features, output_path, output_separator, agg
            )

        return self.features_to_array(features)

    def features_to_array(self, features: List[Dict[str, List[int | float]]]):

        if len(features) == 0:
            return [], np.array([])

        features_names = [k for k in features[0]]
        features_names.sort()
        features = [
            [features[i][v] for v in features_names]
            for i in range(len(features))
        ]
        return features_names, np.array(features)

    def save_features_as_df(
        self,
        files: List[str],
        features: List[Dict[str, List[int | float]]],
        output_path: str,
        output_separator: str = ',',
        agg: bool = False,
    ):

        if len(files) != len(features):
            raise ValueError(
                'The lengths of the files and features parameters should be equal.'
            )

        file_features_df = []
        for filename, extracted_features in zip(files, features):
            file_features_df.append(
                self._features_dict_to_df(
                    extracted_features, filename, agg=agg
                )
            )

        try:
            unified_df = pd.concat(file_features_df)
            unified_df.to_csv(output_path, sep=output_separator, index=False)
        except Exception as e:
            logger.error(f'Error at feature_extractor.py on unified_df - {e}')

    def _features_dict_to_df(
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
