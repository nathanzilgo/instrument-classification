from unittest import mock

from inda_mir.audio_processing.ffmpeg_silence_detector import (
    FFmpegSilenceDetector,
)


def _build_ffmpeg_instance_mock(
    ffmpeg_mock: mock.Mock, run_output: tuple[bytes, bytes]
) -> None:
    ffmpeg_instance_mock = mock.Mock()
    ffmpeg_mock.input.return_value = ffmpeg_instance_mock
    ffmpeg_instance_mock.output.return_value = ffmpeg_instance_mock
    ffmpeg_instance_mock.filter.return_value = ffmpeg_instance_mock
    ffmpeg_instance_mock.run.return_value = run_output


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_the_subprocess_output_is_empty(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(ffmpeg_mock, (b'', b''))

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_no_silence_data(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(ffmpeg_mock, (b'', b'\n \n \n'))

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_silence_duration_below_threshold(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 1.11656\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_true_when_silence_duration_equal_threshold(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 3.0\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is True


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_true_when_silence_duration_above_threshold(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 6.6536\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is True


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_silence_duration_below_threshold_multiple_occurrence(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 1.10'
            + b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 1.25\n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_true_when_silence_duration_above_threshold_multiple_occurrence(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 2.45'
            + b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 1.25\n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file')

    assert result is True


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_silence_duration_below_custom_threshold(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 7.6536\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply(
        '/path/to/file', silence_percentage_threshold=0.9
    )

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_true_when_silence_duration_above_custom_threshold(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 7.6536\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply(
        '/path/to/file', silence_percentage_threshold=0.7
    )

    assert result is True


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_false_when_silence_duration_below_threshold_custom_duration(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 8.987\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file', audio_duration=30)

    assert result is False


@mock.patch('inda_mir.audio_processing.ffmpeg_silence_detector.ffmpeg')
def test_return_true_when_silence_duration_above_threshold_custom_duration(
    ffmpeg_mock: mock.Mock,
) -> None:
    _build_ffmpeg_instance_mock(
        ffmpeg_mock,
        (
            b'',
            b'\n[silencedetect @ 0x129f16bb0] silence_end: 10 | silence_duration: 9.289\n \n',
        ),
    )

    result = FFmpegSilenceDetector.apply('/path/to/file', audio_duration=30)

    assert result is True
