from typing import Dict, List
import numpy as np
from mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


class TorchAudioExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.n_fft = 400
        self.win_length = 200
        self.hop_length = 1024
        self.n_mels = 128
        self.n_mfcc = 40

    def _extract(
        self, file_path: str, **kwargs
    ) -> Dict[str, List[int | float]]:
        waveform, sample_rate = torchaudio.load(file_path)

        return self.compute_features(waveform, sample_rate, **kwargs)

    def compute_features(
        self, waveform, sample_rate, **kwargs
    ) -> Dict[str, List[int | float]]:
        pool: Dict = dict()

        pitch = F.detect_pitch_frequency(waveform, sample_rate)
        self.aggregate('pitch', pitch.tolist(), pool)

        spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=None,
        )

        spec = spectrogram(waveform)

        for band in range(spec.shape[1]):
            self.aggregate(
                f'spec_{band}', spec[:, band, :].squeeze().tolist(), pool
            )

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            mel_scale='htk',
        )
        mel = mel_spectrogram(waveform)

        for band in range(mel.shape[1]):
            self.aggregate(
                f'mel_{band}', mel[:, band, :].squeeze().tolist(), pool
            )

        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'mel_scale': 'htk',
            },
        )

        mfcc = mfcc_transform(waveform)

        for band in range(mfcc.shape[1]):
            self.aggregate(
                f'mfcc_{band}', mfcc[:, band, :].squeeze().tolist(), pool
            )

        lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            speckwargs={
                'n_fft': self.n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
            },
        )

        lfcc = lfcc_transform(waveform)
        for band in range(lfcc.shape[1]):
            self.aggregate(
                f'lfcc_{band}', lfcc[:, band, :].squeeze().tolist(), pool
            )

        centroid_transform = T.SpectralCentroid(sample_rate)
        spectral_centroid = centroid_transform(waveform)
        self.aggregate('spectral_centroid', spectral_centroid.tolist(), pool)

        loudness_transform = T.Loudness(sample_rate)
        loudness = loudness_transform(waveform)
        self.aggregate('loudness', loudness.tolist(), pool)

        return pool

    def compute_features_hubert(self, waveform, sample_rate, **kwargs):
        bundle = torchaudio.pipelines.HUBERT_BASE
        model = bundle.get_model()
        waveform = F.resample(waveform, sample_rate, bundle.sample_rate)

        features, _ = model.extract_features(waveform)
        self.pool.append(features)

    def aggregate(self, feature_name, feature_values, pool):

        aggregated = {}
        # Calculate dmean
        aggregated[feature_name + '_dmean'] = float(
            abs(np.mean(feature_values))
        )
        # Calculate dvar
        aggregated[feature_name + '_dvar'] = np.var(feature_values)
        # Calculate min and max
        aggregated[feature_name + '_min'] = float(abs(np.min(feature_values)))
        aggregated[feature_name + '_max'] = float(abs(np.max(feature_values)))
        # Calculate statistics
        aggregated[feature_name + '_mean'] = float(
            abs(np.mean(feature_values))
        )
        aggregated[feature_name + '_std'] = np.std(feature_values)
        aggregated[feature_name + '_median'] = float(
            abs(np.median(feature_values))
        )

        for key, value in aggregated.items():
            if key not in pool:
                pool[key] = value
