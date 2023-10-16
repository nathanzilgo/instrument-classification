import json
import statistics
import tempfile
import math

from typing import Dict, List
from flatten_json import flatten
import essentia.standard as essentia
from essentia import INFO, Pool, isSilent

from inda_mir.modeling.feature_extractor.feature_extractor import (
    FeatureExtractor,
)

from inda_mir.utils import pow2db, squeezeRange


class EssentiaExtractor(FeatureExtractor):
    """
    This class computes a feature extraction using the Essentia's Freesound Extractor for a file.
    The details about these features can be seen in the provided link.

    @see https://essentia.upf.edu/freesound_extractor.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.namespace = 'lowlevel'

    @staticmethod
    def pow2db(value):
        SILENCE_CUTOFF = 1e-10
        DB_SILENCE_CUTOFF = -100
        return (
            DB_SILENCE_CUTOFF
            if value < SILENCE_CUTOFF
            else 10.0 * math.log10(value)
        )

    @staticmethod
    def squeeze_range(x, x1, x2):
        return 0.5 + 0.5 * math.tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1))

    def compute_average_loudness(self, pool):

        level_array = pool[self.namespace + 'loudness']
        pool.remove(self.namespace + 'loudness')

        EPSILON = 10e-5
        maxValue = max(level_array)
        if maxValue <= EPSILON:
            maxValue = EPSILON

        THRESHOLD = 0.0001
        for i in range(len(level_array)):
            level_array[i] /= maxValue
            if level_array[i] <= THRESHOLD:
                level_array[i] = THRESHOLD

        levelAverage = EssentiaExtractor.pow2db(statistics.mean(level_array))

        x1, x2 = -5.0, -2.0
        levelAverageSqueezed = EssentiaExtractor.squeeze_range(
            levelAverage, x1, x2
        )
        pool.set(
            self.namespace + '.' + 'average_loudness', levelAverageSqueezed
        )

    def compute_audio_metadata(self, audio, pool, options):
        demuxer = essentia.StereoDemuxer()
        muxer = essentia.StereoMuxer()
        resampleR = essentia.Resample(
            inputSampleRate=options['sampleRate'],
            outputSampleRate=options['analysisSampleRate'],
        )
        resampleL = essentia.Resample(
            inputSampleRate=options['sampleRate'],
            outputSampleRate=options['analysisSampleRate'],
        )
        trimmer = essentia.StereoTrimmer(
            sampleRate=options['analysisSampleRate']
        )
        loudness = essentia.LoudnessEBUR128()

        (left, right) = demuxer(audio)
        left = resampleL(left)
        right = resampleR(right)
        m_audio = muxer(left, right)
        t_audio = trimmer(m_audio)
        (
            momentaryLoudness,
            shortTermLoudness,
            integratedLoudness,
            loudnessRange,
        ) = loudness(t_audio)
        pool.set(
            self.namespace + '.' + 'loudness_ebu128_short_term_min',
            min(shortTermLoudness),
        )

    def compute(self, audio, pool, options):

        sampleRate = options['sampleRate']
        frameSize = options['frameSize']
        hopSize = options['hopSize']
        windowType = options['windowType']

        frames = essentia.FrameGenerator(
            audio=audio, frameSize=frameSize, hopSize=hopSize
        )
        window = essentia.Windowing(
            size=frameSize, zeroPadding=0, type=windowType
        )
        spectrum = essentia.Spectrum(size=frameSize)

        zerocrossingrate = essentia.ZeroCrossingRate()
        mfcc = essentia.MFCC()
        melbands96 = essentia.MelBands(numberBands=96)
        square = essentia.UnaryOperator(type='square')
        centroid = essentia.Centroid(range=sampleRate * 0.5)

        gfcc = essentia.GFCC(
            highFrequencyBound=9795, lowFrequencyBound=26, numberBands=18
        )
        erbs_cm = essentia.CentralMoments(range=18 - 1)
        erbs_ds = essentia.DistributionShape()

        sc = essentia.SpectralContrast(
            frameSize=frameSize,
            sampleRate=sampleRate,
            numberBands=6,
            lowFrequencyBound=20,
            highFrequencyBound=11000,
            neighbourRatio=0.4,
            staticDistribution=0.15,
        )

        pitch = essentia.PitchYin(frameSize=frameSize)

        rms = essentia.RMS()

        ln = essentia.Loudness()

        bark = essentia.BarkBands(numberBands=27)

        for frame in frames:

            if options['skipSilence'] and isSilent(frame):
                continue

            pool.add(
                self.namespace + '.' + 'zerocrossingrate',
                zerocrossingrate(frame),
            )

            frame_windowed = window(frame)
            frame_spectrum = spectrum(frame_windowed)

            (frame_melbands, frame_mfcc) = mfcc(frame_spectrum)
            pool.add(self.namespace + '.' + 'mfcc', frame_mfcc)

            frame_melbands96 = melbands96(frame_spectrum)
            pool.add(self.namespace + '.' + 'melbands96', frame_melbands96)

            square_spectrum = square(frame_spectrum)
            frame_centroid = centroid(square_spectrum)
            pool.add(
                self.namespace + '.' + 'spectral_centroid', frame_centroid
            )

            (frame_erb_bands, frame_gfcc) = gfcc(frame_spectrum)
            pool.add(self.namespace + '.' + 'gfcc', frame_gfcc)
            centralMoments = erbs_cm(frame_erb_bands)
            (erb_spread, erb_skewness, erb_kurtosis) = erbs_ds(centralMoments)
            pool.add(self.namespace + '.' + 'erbbands_spread', erb_spread)

            (spectral_contrast_coeffs, spectral_contrast_valleys) = sc(
                frame_spectrum
            )
            pool.add(
                self.namespace + '.' + 'spectral_contrast_coeffs',
                spectral_contrast_coeffs,
            )
            pool.add(
                self.namespace + '.' + 'spectral_contrast_valleys',
                spectral_contrast_valleys,
            )

            (frame_pitch, pitchConfidence) = pitch(frame)
            pool.add(
                self.namespace + '.' + 'pitch_instantaneous_confidence',
                pitchConfidence,
            )

            frame_rms = rms(frame_spectrum)
            pool.add(self.namespace + '.' + 'spectral_rms', frame_rms)

            loudness = ln(frame)
            pool.add(self.namespace + 'loudness', loudness)

            barkbands = bark(frame_spectrum)
            pool.add(self.namespace + '.' + 'barkbands', barkbands)

    def aggregate(self, pool):
        exceptions = {
            f'{self.namespace}.spectral_contrast_calleys_coeffs': ['stdev'],
            f'{self.namespace}.melbands96': ['median'],
            f'{self.namespace}.zerocrossingrate': ['max', 'median'],
            f'{self.namespace}.mfcc': [
                'min',
                'max',
                'stdev',
                'median',
                'mean',
            ],
            f'{self.namespace}.spectral_centroid': ['mean'],
            f'{self.namespace}.gfcc': ['mean', 'dvar'],
            f'{self.namespace}.erbbands_spread': ['dvar'],
            f'{self.namespace}.spectral_contrast_coeffs': ['stdev', 'mean'],
            f'{self.namespace}.spectral_contrast_valleys': [
                'dvar',
                'stdev',
                'median',
            ],
            f'{self.namespace}.pitch_instantaneous_confidence': [
                'mean',
                'dvar',
            ],
            f'{self.namespace}.spectral_rms': ['dvar'],
            f'{self.namespace}.barkbands': ['median'],
        }

        defaultStats = ['min', 'max', 'stdev', 'median', 'mean', 'dvar']

        return essentia.PoolAggregator(
            defaultStats=defaultStats, exceptions=exceptions
        )(pool)

    def _extract(
        self,
        file_path: str,
        sample_rate: float = 44100,
        frame_size: int = 2028,
        hop_size: int = 1024,
        window_type: str = 'blackmanharris62',
        skip_silence: bool = True,
        **kwargs,
    ) -> Dict[str, List[int | float]]:

        loader = essentia.AudioLoader(filename=file_path, computeMD5=True)
        (
            audio,
            loader_sampleRate,
            numberChannels,
            md5,
            bit_rate,
            codec,
        ) = loader()
        ops = {
            'sampleRate': loader_sampleRate,
            'analysisSampleRate': sample_rate,
        }

        pool = Pool()

        self.compute_audio_metadata(audio, pool, ops)

        loader = essentia.EasyLoader(filename=file_path)
        audio = loader()

        ops = {
            'sampleRate': sample_rate,
            'frameSize': frame_size,
            'hopSize': hop_size,
            'windowType': window_type,
            'skipSilence': skip_silence,
        }

        self.compute(audio, pool, ops)
        self.compute_average_loudness(pool)
        aggPool = self.aggregate(pool)

        file = tempfile.NamedTemporaryFile()
        output = essentia.YamlOutput(filename=file.name, format='json')
        output(aggPool)

        with open(file.name, 'r') as json_file:
            features = json.load(json_file)

        features = flatten(features['lowlevel'])

        return features
