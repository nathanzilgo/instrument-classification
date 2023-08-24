import audioflux as af
import numpy as np

from audioflux.type import SpectralFilterBankScaleType
from typing import Dict, List

from src.modeling.feature_extractor.feature_extractor import FeatureExtractor


class SpectralExtractor(FeatureExtractor):
    """
    This class computes 25 spectral features using the AudioFlux library.
    The details about these features can be seen in the provided link.
    Additional parameters to calculate a feature must be passed as a kwargs:
    ```py
        # Passing the is_positive argument to be used in spectral flux calculation
        af.extract(..., flux={is_positive: True})
    ```

    @see https://audioflux.top/feature/spectral.html
    """

    FEATURES = [
        'band_width',
        'broadband',
        'centroid',
        'crest',
        'decrease',
        'eef',
        'eer',
        'energy',
        'entropy',
        'flatness',
        'flux',
        'hfc',
        'kurtosis',
        'max',
        'mean',
        'mkl',
        'novelty',
        'rms',
        'rolloff',
        'sd',
        'sf',
        'skewness',
        'slope',
        'spread',
        'var',
    ]

    def _extract(
        self, file_path: str, slide_length: int = 1024, **kwargs
    ) -> Dict[str, List[int | float]]:
        audio_arr, sr = af.read(file_path)

        radix2_exp = int(np.log2(slide_length) + 1)

        extractor = af.FeatureExtractor(
            transforms=['bft'],
            samplate=sr,
            slide_length=slide_length,
            radix2_exp=radix2_exp,
            scale_type=SpectralFilterBankScaleType.MEL,
        )
        spec = extractor.spectrogram(audio_arr, is_continue=True)

        extracted_features = {}

        for feature in self.FEATURES:
            spectral_result = extractor.spectral(
                spec, spectral=feature, spectral_kw=kwargs.get(feature)
            )['bft'][0]
            if isinstance(spectral_result[0], np.ndarray):
                spectral_result = spectral_result[0]
            extracted_features[feature] = spectral_result

        return extracted_features
