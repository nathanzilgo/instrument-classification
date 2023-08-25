from inda_mir.audio_processing.audio_operation import AudioOperation
from pydub import AudioSegment
from pydub.silence import split_on_silence


class SilenceFilter(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        input_format: str = None,
        output_format: str = 'ogg',
        output_path: str = './output',
        min_silence_len: int = 100,
        silence_thresh: int = -45,
        keep_silence: int = 30,
    ) -> str:
        """
        Applies a silence removal process to the audio file at the specified path.

        Args:
            audio_path (str): The path to the audio file.
            input_format (str, optional): The format of the input audio file. Defaults to 'ogg'.
            output_format (str, optional): The format of the output audio file. Defaults to 'ogg'.
            output_path (str, optional): The path where the output audio file will be saved. Defaults to './output'.
            min_silence_len (int, optional): The minimum duration of silence to consider for removal. Defaults to 100.
            silence_thresh (int, optional): The threshold below which audio is considered silence. Defaults to -45.
            keep_silence (int, optional): The duration of silence to keep between audio chunks. Defaults to 30.

        Returns:
            str: The path to the output audio file.
        """
        track = AudioSegment.from_file(audio_path, format=input_format)
        audio_chunks = split_on_silence(
            track,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        combined = sum(audio_chunks)

        if combined is None or combined == 0:
            return ''

        return combined.export(
            f'{output_path}_rm_silence.{output_format}',
            format=output_format,
        )
