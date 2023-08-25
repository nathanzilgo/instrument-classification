import json
import os
import tempfile

from typing import Dict, List

from flatten_json import flatten

from inda_mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)


class FreesoundExtractor(FeatureExtractor):
    """
    This class computes a feature extraction using the Essentia's Freesound Extractor.
    The details about these features can be seen in the provided link.

    @see https://essentia.upf.edu/freesound_extractor.html
    """

    def _extract(
        self,
        file_path: str,
        **kwargs,
    ) -> Dict[str, List[int | float]]:
        file = tempfile.NamedTemporaryFile()
        os.system(
            f'essentia_streaming_extractor_freesound {file_path} {file.name}'
        )

        features = json.load(file)
        features = flatten(features['lowlevel'])
        return features
