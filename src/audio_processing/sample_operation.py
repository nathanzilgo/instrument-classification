from src.audio_processing.audio_operation import AudioOperation
from pydub import AudioSegment  # type: ignore
import os


class SampleOperation(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        input_format: str = None,
        output_format: str = 'ogg',
        sample_duration: int = 10000,
        output_path: str = './output',
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

        track = AudioSegment.from_file(audio_path, format=input_format)
        os.makedirs('./output', exist_ok=True)

        for i in range(0, len(track), sample_duration):
            if len(track) > i + sample_duration or keep_trace:
                track[i : i + sample_duration].export(
                    f'{output_path}_{i}.{output_format}',
                    format=output_format,
                )
            else:
                print(f'Skipping {len(track) - i} trace at {i}')
