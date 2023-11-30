import os

from typing import List

from inda_mir.audio_processing.audio_operation import AudioOperation
from inda_mir.utils.logger import logger
from pydub import AudioSegment  # type: ignore


class SampleOperation(AudioOperation):
    def apply(
        self,
        audio_path: str,
        sample_proportion: float,
        input_format: str = None,
        output_format: str = 'ogg',
        sample_duration: int = 10000,
        output_dir: str = './output',
        output_basename: str = 'track',
        keep_trace: bool = False,
    ) -> None:
        """
        Apply an audio slice operation to a specified audio file.

        Args:
            audio_path (str): The path to the input audio file.
            input_format (str, optional): The format of the input audio file. Defaults to 'ogg'.
            output_format (str, optional): The format of the output audio files. Defaults to 'ogg'.
            sample_duration (int, optional): The duration of each sample in milliseconds. Defaults to 10000.
            output_path (str, optional): The path to the directory where the output audio files will be saved. Defaults to './output'.
            keep_trace (bool, optional): Whether to keep the trace of the audio remaining after the operation. Defaults to False.

        Returns:
            None
        """

        segments = self._get_segments(
            audio_path, input_format, sample_duration, keep_trace, sample_proportion
        )
        self._write_segments(
            segments, output_dir, output_basename, output_format
        )

    def _get_segments(
        self,
        audio_path: str,
        input_format: str,
        sample_duration: int,
        keep_trace: bool,
        sample_proportion: float,
    ) -> List[AudioSegment]:

        track = AudioSegment.from_file(audio_path, format=input_format)

        segments = []
        for i in range(0, int(sample_proportion * len(track)), sample_duration):
            if len(track) >= i + sample_duration or keep_trace:
                segments.append(track[i : i + sample_duration])
            else:
                logger.info(f'Skipping {len(track) - i} trace at {i}')
        return segments

    def _write_segments(
        self,
        segments: List[AudioSegment],
        output_dir: str,
        output_basename: str,
        output_format: str,
    ):
        for i in range(0, len(segments)):
            segments[i].export(
                os.path.join(
                    output_dir, f'{output_basename}_{i}.{output_format}'
                ),
                format=output_format,
            )
