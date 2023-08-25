import audioflux as af

from typing import Dict, List

from inda_mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)
from inda_mir.utils.vggish import waveform_to_features


class VGGExtractor(FeatureExtractor):
    """
    This class computes the VGGish features. For further information check the link provided.

    @see https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md
    """

    def _extract(
        self,
        file_path: str,
        **kwargs,
    ) -> Dict[str, List[int | float]]:
        audio_arr, sr = af.read(file_path)
        _, features = waveform_to_features(audio_arr, sr)
        features = {
            f'f_{i:03d}': [features[j][i] for j in range(len(features))]
            for i in range(len(features[0]))
        }

        return features
