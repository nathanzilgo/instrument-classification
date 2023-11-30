import math

from unittest import mock

from inda_mir.audio_processing.sample_operation import (
    SampleOperation,
    AudioSegment,
)


def _build_audio_segment_mock_for_segments(
    audio_segment_mock: mock.Mock, track_duration: int
):
    audio_segment_mock.from_file.return_value = [
        0 for _ in range(track_duration)
    ]


def _build_audio_segment_mock_for_write(
    audio_segment_mock: mock.Mock, track_duration: int
):
    audio_segment_instance_mock = mock.Mock()
    audio_segment_mock.from_file.return_value = audio_segment_instance_mock
    audio_segment_instance_mock.__len__ = mock.Mock(
        return_value=track_duration
    )

    segment = AudioSegment.empty()
    audio_segment_instance_mock.__getitem__ = mock.Mock(return_value=segment)


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_segment_size_equals_sample_size(audio_segment_mock: mock.Mock):

    track_duration, sample_duration = 6, 2

    _build_audio_segment_mock_for_segments(audio_segment_mock, track_duration)

    s = SampleOperation()
    segments = s._get_segments(
        '/path/to/file',
        'ogg',
        sample_duration=sample_duration,
        sample_proportion=1.0,
        keep_trace=True,
    )
    for segment in segments:
        assert len(segment) == sample_duration


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_track_duration_is_multiple_with_keep_trace(
    audio_segment_mock: mock.Mock,
):

    track_duration, sample_duration = 6, 2
    expected_size = track_duration / sample_duration

    _build_audio_segment_mock_for_segments(audio_segment_mock, track_duration)

    s = SampleOperation()
    segments = s._get_segments(
        '/path/to/file',
        'ogg',
        sample_duration=sample_duration,
        sample_proportion=1.0,
        keep_trace=True,
    )
    assert len(segments) == expected_size


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_track_duration_is_multiple_without_keep_trace(
    audio_segment_mock: mock.Mock,
):

    track_duration, sample_duration = 6, 2
    expected_size = track_duration / sample_duration

    _build_audio_segment_mock_for_segments(audio_segment_mock, track_duration)

    s = SampleOperation()
    segments = s._get_segments(
        '/path/to/file',
        'ogg',
        sample_duration=sample_duration,
        sample_proportion=1.0,
        keep_trace=False,
    )
    assert len(segments) == expected_size


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_track_duration_not_multiple_with_keep_trace(
    audio_segment_mock: mock.Mock,
):

    track_duration, sample_duration = 5, 2
    expected_size = math.ceil(track_duration / sample_duration)

    _build_audio_segment_mock_for_segments(audio_segment_mock, track_duration)

    s = SampleOperation()
    segments = s._get_segments(
        '/path/to/file',
        'ogg',
        sample_duration=sample_duration,
        sample_proportion=1.0,
        keep_trace=True,
    )
    assert len(segments) == expected_size


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_track_duration_not_multiple_without_keep_trace(
    audio_segment_mock: mock.Mock,
):

    track_duration, sample_duration = 5, 2
    expected_size = track_duration // sample_duration

    _build_audio_segment_mock_for_segments(audio_segment_mock, track_duration)

    s = SampleOperation()
    segments = s._get_segments(
        '/path/to/file',
        'ogg',
        sample_duration=sample_duration,
        sample_proportion=1.0,
        keep_trace=False,
    )
    assert len(segments) == expected_size


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_writes_files(audio_segment_mock: mock.Mock):

    track_duration, sample_duration = 6, 2
    expected_size = track_duration // sample_duration

    _build_audio_segment_mock_for_write(audio_segment_mock, track_duration)

    s = SampleOperation()

    m = mock.mock_open()
    with mock.patch.object(AudioSegment, 'export', m):
        s.apply(
            '/path/to/file',
            sample_proportion=1.0,
            input_format='ogg',
            sample_duration=sample_duration,
            keep_trace=True,
            output_dir='/path/to/outdir',
        )

    assert len(m.mock_calls) == expected_size
    for i in range(expected_size):
        assert (
            mock.call(f'/path/to/outdir/track_{i}.ogg', format='ogg')
            in m.mock_calls
        )


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_writes_files_with_basename(audio_segment_mock: mock.Mock):

    track_duration, sample_duration = 6, 2
    expected_size = track_duration // sample_duration

    basename = 'test'

    _build_audio_segment_mock_for_write(audio_segment_mock, track_duration)

    s = SampleOperation()

    m = mock.mock_open()
    with mock.patch.object(AudioSegment, 'export', m):
        s.apply(
            audio_path='/path/to/file',
            sample_proportion=1.0,
            input_format='ogg',
            sample_duration=sample_duration,
            keep_trace=True,
            output_dir='/path/to/outdir',
            output_basename=basename,
        )

    assert len(m.mock_calls) == expected_size
    for i in range(expected_size):
        assert (
            mock.call(f'/path/to/outdir/{basename}_{i}.ogg', format='ogg')
            in m.mock_calls
        )


@mock.patch('inda_mir.audio_processing.sample_operation.AudioSegment')
def test_writes_files_with_format(audio_segment_mock: mock.Mock):

    track_duration, sample_duration = 6, 2
    expected_size = track_duration // sample_duration

    output_format = 'test'

    _build_audio_segment_mock_for_write(audio_segment_mock, track_duration)

    s = SampleOperation()

    m = mock.mock_open()
    with mock.patch.object(AudioSegment, 'export', m):
        s.apply(
            audio_path='/path/to/file',
            sample_proportion=1.0,
            input_format='ogg',
            sample_duration=sample_duration,
            keep_trace=True,
            output_dir='/path/to/outdir',
            output_format=output_format,
        )

    assert len(m.mock_calls) == expected_size
    for i in range(expected_size):
        assert (
            mock.call(
                f'/path/to/outdir/track_{i}.{output_format}',
                format=output_format,
            )
            in m.mock_calls
        )
