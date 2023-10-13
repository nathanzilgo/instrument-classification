from unittest import mock

from typing import List

from inda_mir.audio_processing.silence_filter import (
    AudioSegment,
    SilenceFilter,
)


def _build_audio_segment_mock(
    audio_segment_mock: mock.Mock,
    split_on_silence_mock: mock.Mock,
    split_return: List[AudioSegment],
):
    audio_segment_mock.from_file.return_value = []
    split_on_silence_mock.return_value = split_return


@mock.patch('inda_mir.audio_processing.silence_filter.split_on_silence')
@mock.patch('inda_mir.audio_processing.silence_filter.AudioSegment')
def test_audio_with_empty_split(
    audio_segment_mock: mock.Mock, split_on_silence_mock: mock.Mock
):

    _build_audio_segment_mock(audio_segment_mock, split_on_silence_mock, [])

    s = SilenceFilter()
    filename = s.apply('/path/to/file', 'ogg')

    assert filename == ''


@mock.patch('inda_mir.audio_processing.silence_filter.split_on_silence')
@mock.patch('inda_mir.audio_processing.silence_filter.AudioSegment')
def test_audio_with_nonempty_split(
    audio_segment_mock: mock.Mock, split_on_silence_mock: mock.Mock
):

    _build_audio_segment_mock(
        audio_segment_mock, split_on_silence_mock, [AudioSegment.empty()]
    )

    s = SilenceFilter()
    m = mock.mock_open()
    with mock.patch.object(AudioSegment, 'export', m):
        filename = s.apply('/path/to/file', 'ogg', output_path='/path/to/file')

    assert filename == '/path/to/file_rm_silence.ogg'
    assert mock.call(filename, format='ogg') in m.mock_calls
