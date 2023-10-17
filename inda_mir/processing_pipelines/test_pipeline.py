from unittest import mock

from typing import List

from inda_mir.processing_pipelines import sample_and_filter_silence


def _build_os_mock(
    os_mock: mock.Mock, exists: bool = True, list_dir: List[str] = []
):

    path_mock = mock.Mock()
    path_mock.exists.return_value = exists
    path_mock.join = lambda *x: x[-1]

    os_mock.path = path_mock
    os_mock.makedirs = lambda x: f'Created dir {x}'
    os_mock.listdir.return_value = list_dir


def _build_operations_mock(
    sample_op: mock.Mock, silence_op: mock.Mock, silent: List[bool] = []
):

    sample_op.return_value = f'Created samples'
    silence_op.side_effect = silent


@mock.patch('inda_mir.audio_processing.FFmpegSilenceDetector.apply')
@mock.patch('inda_mir.audio_processing.SampleOperation.apply')
@mock.patch('inda_mir.processing_pipelines.os')
def test_pipeline_non_existing_samples_not_silent(
    os_mock: mock.Mock, sample_op: mock.Mock, silence_op: mock.Mock
):

    samples = ['/path/to/sample1', '/path/to/sample2']

    _build_os_mock(os_mock, exists=False, list_dir=samples)
    _build_operations_mock(sample_op, silence_op, silent=[False, False])

    non_silent_samples = sample_and_filter_silence(
        '/path/to/track', 'ogg', '/output/', 'basename', 0, 0, 0, 0
    )
    assert sample_op.called
    assert non_silent_samples == samples


@mock.patch('inda_mir.audio_processing.FFmpegSilenceDetector.apply')
@mock.patch('inda_mir.audio_processing.SampleOperation.apply')
@mock.patch('inda_mir.processing_pipelines.os')
def test_pipeline_existing_samples_not_silent(
    os_mock: mock.Mock, sample_op: mock.Mock, silence_op: mock.Mock
):

    samples = ['/path/to/sample1', '/path/to/sample2']

    _build_os_mock(os_mock, list_dir=samples)
    _build_operations_mock(sample_op, silence_op, silent=[False, False])

    non_silent_samples = sample_and_filter_silence(
        '/path/to/track', 'ogg', '/output/', 'basename', 0, 0, 0, 0
    )
    assert not sample_op.called
    assert silence_op.call_count == 2
    assert non_silent_samples == samples


@mock.patch('inda_mir.audio_processing.FFmpegSilenceDetector.apply')
@mock.patch('inda_mir.audio_processing.SampleOperation.apply')
@mock.patch('inda_mir.processing_pipelines.os')
def test_pipeline_existing_samples_silent(
    os_mock: mock.Mock, sample_op: mock.Mock, silence_op: mock.Mock
):

    samples = ['/path/to/sample1', '/path/to/sample2']

    _build_os_mock(os_mock, list_dir=samples)
    _build_operations_mock(sample_op, silence_op, silent=[True, True])

    non_silent_samples = sample_and_filter_silence(
        '/path/to/track', 'ogg', '/output/', 'basename', 0, 0, 0, 0
    )
    assert not sample_op.called
    assert silence_op.call_count == 2
    assert non_silent_samples == []


@mock.patch('inda_mir.audio_processing.FFmpegSilenceDetector.apply')
@mock.patch('inda_mir.audio_processing.SampleOperation.apply')
@mock.patch('inda_mir.processing_pipelines.os')
def test_pipeline_existing_samples_half_silent(
    os_mock: mock.Mock, sample_op: mock.Mock, silence_op: mock.Mock
):

    samples = ['/path/to/sample1', '/path/to/sample2']

    _build_os_mock(os_mock, list_dir=samples)
    _build_operations_mock(sample_op, silence_op, silent=[True, False])

    non_silent_samples = sample_and_filter_silence(
        '/path/to/track', 'ogg', '/output/', 'basename', 0, 0, 0, 0
    )
    assert not sample_op.called
    assert silence_op.call_count == 2
    assert non_silent_samples == ['/path/to/sample2']
