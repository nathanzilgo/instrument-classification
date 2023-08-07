from src.audio_processing.audio_operation import AudioOperation
from pydub import AudioSegment  # type: ignore
import os


class SampleOperation(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        input_format: str = 'ogg',
        output_format: str = 'ogg',
        sample_duration: int = 10000,
        output_path: str = './output',
        keep_trace: bool = False,
    ) -> None:

        track = AudioSegment.from_file(audio_path, format=input_format)
        os.makedirs(output_path, exist_ok=True)
        for i in range(0, len(track), sample_duration):
            if len(track) > i + sample_duration or keep_trace:
                track[i : i + sample_duration].export(
                    f'{output_path}/out_{i}.{output_format}',
                    format=output_format,
                )
            else:
                print(f'Skipping {len(track) - i} trace at {i}')
