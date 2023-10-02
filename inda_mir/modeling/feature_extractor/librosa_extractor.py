import numpy as np
import librosa

from typing import Dict, List


from inda_mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)

"""
Top 20 Features for LGBM:
spectral_contrast_valleys
spectral_rms
mfcc_mean
mfcc_max
zerocrossingrate_max
pitch_instantaneous_confidence_mean
mfcc_stdev
loudness_ebu128_short_term_min
gfcc_mean
melbands96_median
spectral_contrast_coeffs_mean
spectral_contrast_coeffs_stdev
erbbands_spread_dvar
average_loudness
spectral_centroid_mean
melbans96_mean
spectral_contrast_valleys_stdev
mfcc_min
zerocrossingrate_median
"""
# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
HOP_LENGTH = 512


class LibrosaExtractor(FeatureExtractor):
    """
    This class computes a feature extraction using the Librosa's Extractor.
    The details about these features can be seen in the provided link.

    @see https://librosa.org/doc/latest/tutorial.html
    """

    def _extract(
        self, file_path: str, **kwargs
    ) -> Dict[str, List[int | float]]:

        y, sr = librosa.load(file_path, sr=None)

        features = {}
        mfcc_mean = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)
        if isinstance(mfcc_mean, np.ndarray):
            features['mfcc_mean'] = mfcc_mean
        mfcc_max = librosa.feature.mfcc(y=y, sr=sr).max(axis=1)
        if isinstance(mfcc_max, np.ndarray):
            features['mfcc_max'] = mfcc_max
        spectral_centroid_mean = librosa.feature.spectral_centroid(
            y=y, sr=sr
        ).mean(axis=1)
        if isinstance(spectral_centroid_mean, np.ndarray):
            features['spectral_centroid_mean'] = spectral_centroid_mean
        spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(
            y=y, sr=sr
        ).mean(axis=1)
        if isinstance(spectral_bandwidth_mean, np.ndarray):
            features['spectral_bandwidth_mean'] = spectral_bandwidth_mean
        spectral_contrast_mean = librosa.feature.spectral_contrast(
            y=y, sr=sr
        ).mean(axis=1)
        if isinstance(spectral_contrast_mean, np.ndarray):
            features['spectral_contrast_mean'] = spectral_contrast_mean
        zerocrossingrate_mean = librosa.feature.zero_crossing_rate(y=y).mean(
            axis=1
        )
        if isinstance(zerocrossingrate_mean, np.ndarray):
            features['zerocrossingrate_mean'] = zerocrossingrate_mean
        mfcc_min = librosa.feature.mfcc(y=y, sr=sr).min(axis=1)
        if isinstance(mfcc_min, np.ndarray):
            features['mfcc_min'] = mfcc_min
        zerocrossingrate_max = librosa.feature.zero_crossing_rate(y=y).max(
            axis=1
        )
        if isinstance(zerocrossingrate_max, np.ndarray):
            features['zerocrossingrate_max'] = zerocrossingrate_max
        spectral_rms = librosa.feature.rms(y=y)
        if isinstance(spectral_rms[0], np.ndarray):
            features['spectral_rms'] = spectral_rms[0]
        mel_features = librosa.feature.melspectrogram(y=y, sr=sr)
        if isinstance(mel_features[0], np.ndarray):
            features['mel'] = mel_features[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        if isinstance(chroma_stft[0], np.ndarray):
            features['chroma_stft'] = chroma_stft[0]
        spectral_rolloff_mean = librosa.feature.spectral_rolloff(
            y=y, sr=sr
        ).mean(axis=1)
        if isinstance(spectral_rolloff_mean, np.ndarray):
            features['spectral_rolloff_mean'] = spectral_rolloff_mean
        spectral_centroid_std = librosa.feature.spectral_centroid(y=y, sr=sr)
        if isinstance(spectral_centroid_std[0], np.ndarray):
            features['spectral_centroid_std'] = spectral_centroid_std[0]
        spectral_bandwidth_std = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        if isinstance(spectral_bandwidth_std[0], np.ndarray):
            features['spectral_bandwidth_std'] = spectral_bandwidth_std[0]
        spectral_contrast_std = librosa.feature.spectral_contrast(y=y, sr=sr)
        if isinstance(spectral_contrast_std[0], np.ndarray):
            features['spectral_contrast_std'] = spectral_contrast_std[0]
        spectral_rolloff_std = librosa.feature.spectral_rolloff(y=y, sr=sr)
        if isinstance(spectral_rolloff_std[0], np.ndarray):
            features['spectral_rolloff_std'] = spectral_rolloff_std[0]
        mfcc_std = librosa.feature.mfcc(y=y, sr=sr)
        if isinstance(mfcc_std[0], np.ndarray):
            features['mfcc_std'] = mfcc_std[0]

        return features
