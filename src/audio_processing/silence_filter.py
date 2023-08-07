from src.audio_processing.audio_operation import AudioOperation
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


class SilenceFilter(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        input_format: str = 'ogg',
        output_format: str = 'ogg',
        output_path: str = './output',
        min_silence_len: int = 100,
        silence_thresh: int = -45,
        keep_silence: int = 30,
    ) -> None:

        track = AudioSegment.from_file(audio_path, format=input_format)
        audio_chunks = split_on_silence(
            track,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        combined = sum(audio_chunks)
        os.makedirs(output_path, exist_ok=True)
        combined.export(
            f'{output_path}/out_rm_silence.{output_format}',
            format=output_format,
        )
