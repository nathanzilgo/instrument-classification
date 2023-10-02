from inda_mir.audio_processing.audio_operation import AudioOperation


class FFmpegSilenceRemoval(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        input_format: str = None,
        output_format: str = 'ogg',
        output_path: str = './output',
        **kwargs
    ) -> str:
        ...
