from src.modeling.feature_extractor.feature_extractor import FeatureExtractor
import audioflux as af
from src.utils.vggish import waveform_to_features


class VGGExtractor(FeatureExtractor):
    """
    This class computes the VGGish features. For further information check the link provided.

    @see https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md
    """

    def extract(
        self,
        file_path: str,
        output_path: str,
        output_separator: str = ',',
        **kwargs,
    ) -> any:
        audio_arr, sr = af.read(file_path)
        _, features = waveform_to_features(audio_arr, sr)
        features = {
            f'f_{i:03d}': [features[j][i] for j in range(len(features))]
            for i in range(len(features[0]))
        }
        super().write_features(
            features, file_path, output_path, output_separator
        )
