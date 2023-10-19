from unittest import mock
import essentia.standard as std
import essentia

import numpy as np
import re

from inda_mir.modeling.feature_extractor import EssentiaExtractor

FEATURES = [
    'mfcc',
    'spectral_centroid',
    'spectral_contrast_valleys',
    'spectral_rms',
    'barkbands',
    'zerocrossingrate',
    'pitch_instantaneous_confidence',
    'spectral_contrast_coeffs',
    'erbbands_spread',
    'melbands96',
    'gfcc',
]

FEATURES_WITH_LOUDNESS = FEATURES + [
    'loudness',
    'average_loudness',
    'loudness_ebu128',
]

NAMESPACE = 'lowlevel'


def test_compute_average_loudness():
    e = EssentiaExtractor()
    pool = essentia.Pool()
    pool.set('lowlevelloudness', [0.1, 0.2, 0.3])
    e.compute_average_loudness(pool)

    assert round(pool['lowlevel.average_loudness'], 2) == 0.91
    assert not pool.containsKey('lowlevelloudness')
    assert len(pool.descriptorNames()) == 1


def test_compute_average_loudness_below_epsilon():
    e = EssentiaExtractor()
    pool = essentia.Pool()
    pool.set('lowlevelloudness', [10e-5, 10e-5, 10e-5])
    e.compute_average_loudness(pool)

    assert round(pool['lowlevel.average_loudness'], 2) == 0.99
    assert not pool.containsKey('lowlevelloudness')
    assert len(pool.descriptorNames()) == 1


def test_compute_average_loudness_below_threshold():
    e = EssentiaExtractor()
    pool = essentia.Pool()
    pool.set('lowlevelloudness', [0.000000001, 0.2, 0.3])
    e.compute_average_loudness(pool)

    assert round(pool['lowlevel.average_loudness'], 2) == 0.78
    assert not pool.containsKey('lowlevelloudness')
    assert len(pool.descriptorNames()) == 1


@mock.patch('essentia.standard.AudioLoader')
def test_compute_audio_metadata(audio_loader_mock: mock.Mock):
    e = EssentiaExtractor()
    pool = essentia.Pool()

    audio_array = np.array(
        [[-0.0072446, 0], [-0.00688219, 0.0], [-0.00650775, 0.0]]
    )
    audio_loader_mock.return_value = mock.Mock(
        return_value=(audio_array, 44100, None, None, None, None)
    )

    loader = std.AudioLoader(filename='', computeMD5=True)
    (
        audio,
        loader_sampleRate,
        numberChannels,
        md5,
        bit_rate,
        codec,
    ) = loader()

    e.compute_audio_metadata(
        audio, pool, {'sampleRate': 44100, 'analysisSampleRate': 44100}
    )

    assert pool.containsKey('lowlevel.loudness_ebu128_short_term_min')
    assert (
        pool['lowlevel.loudness_ebu128_short_term_min'] == -87.34703826904297
    )
    assert len(pool.descriptorNames()) == 1


def test_compute():
    e = EssentiaExtractor()
    pool = essentia.Pool()

    audio_mocked_mono = np.array(
        [
            -0.00667055,
            -0.00721706,
            -0.0061507,
            -0.00742259,
            -0.00680307,
            -0.00769188,
        ]
    )

    options = {
        'sampleRate': 44100,
        'frameSize': 2028,
        'hopSize': 1024,
        'windowType': 'blackmanharris62',
        'skipSilence': True,
    }

    e.compute(audio_mocked_mono, pool, options)

    for feature in FEATURES:
        assert pool.containsKey(f'{NAMESPACE}.{feature}')

    assert len(pool.descriptorNames()) == 12


def test_aggregate():
    e = EssentiaExtractor()
    pool = essentia.Pool()

    pool.add('lowlevel.mfcc', [0.1, 0.2, 0.3])
    pool.add('lowlevel.melbands96', [0.4, 0.5, 0.6])
    pool.add('lowlevel.spectral_contrast', [0.7, 0.8, 0.9])

    result_pool = e.aggregate(
        pool,
        exceptions={
            'lowlevel.mfcc': ['max'],
            'lowlevel.melbands96': ['dmean'],
            'lowlevel.spectral_contrast': ['dvar'],
        },
    )

    assert result_pool.containsKey('lowlevel.melbands96.dmean')
    assert result_pool.containsKey('lowlevel.spectral_contrast.dvar')
    assert result_pool.containsKey('lowlevel.mfcc.max')
    assert not result_pool.containsKey('lowlevel.mfcc.min')
    assert len(result_pool.descriptorNames()) == 3


@mock.patch('essentia.standard.EasyLoader')
@mock.patch('essentia.standard.AudioLoader')
def test_extract(audio_loader_mock: mock.Mock, easy_loader_mock: mock.Mock):
    e = EssentiaExtractor()
    pool = essentia.Pool()

    audio_mocked_mono = np.array(
        [
            -0.00667055,
            -0.00721706,
            -0.0061507,
            -0.00742259,
            -0.00680307,
            -0.00769188,
        ]
    )

    audio_mocked_stereo = np.array(
        np.array([[-0.0072446, 0], [-0.00688219, 0.0], [-0.00650775, 0.0]])
    )

    audio_loader_mock.return_value = mock.Mock(
        return_value=(audio_mocked_stereo, 44100, None, None, None, None)
    )

    easy_loader_mock.return_value = mock.Mock(return_value=audio_mocked_mono)

    features = e._extract(file_path='', pool=pool, sample_rate=44100)

    input_list = features.keys()

    for feature in FEATURES_WITH_LOUDNESS:
        assert any(re.search(feature, string) for string in input_list)

    easy_loader_mock.assert_called_once()
    audio_loader_mock.assert_called_once()
