import audioflux as af
import numpy as np

from audioflux.type import SpectralFilterBankScaleType
from typing import Dict, List

from inda_mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)


class MFCCExtractor(FeatureExtractor):
    """
    This class computes MFCC using the AudioFlux library.
    The details about these features can be seen in the provided link.

    @see https://audioflux.top/feature/xxcc.html
    """

    def _extract(
        self,
        file_path: str,
        slide_length: int = 1024,
        num_coefficients: int = 13,
        **kwargs,
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

        xxcc_result = extractor.xxcc(spec, cc_num=num_coefficients)['bft'][0]
        xxcc_result = {
            f'mfcc_{i:02d}': xxcc_result[i] for i in range(num_coefficients)
        }

        return xxcc_result
