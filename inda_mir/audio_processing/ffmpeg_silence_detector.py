import ffmpeg
import subprocess
import re

from inda_mir.audio_processing.audio_operation import AudioOperation
from inda_mir.utils.logger import logger


class FFmpegSilenceDetector(AudioOperation):
    @staticmethod
    def apply(
        audio_path: str,
        audio_duration: float = 10.0,
        min_silence_duration: int = 1,
        silence_threshold: int = -45,
        silence_percentage_threshold: float = 0.3,
        **kwargs
    ) -> bool:

        p = subprocess.Popen(
            (
                ffmpeg.input(audio_path)
                .filter(
                    'silencedetect',
                    n='{}dB'.format(silence_threshold),
                    d=min_silence_duration,
                )
                .output('-', format='null')
                .compile()
            )
            + [
                '-nostats'
            ],  # FIXME: use .nostats() once it's implemented in ffmpeg-python.
            stderr=subprocess.PIPE,
        )

        output = p.communicate()[1].decode('utf-8')
        if p.returncode != 0:
            logger.error(output)
            raise Exception

        lines = output.splitlines()

        silence_duration_re = re.compile(
            r' silence_duration: (?P<duration>[0-9]+(\.?[0-9]*))$'
        )

        silence_chunks_duration = []
        for line in lines:
            silence_duration_match = silence_duration_re.search(line)
            if silence_duration_match:
                silence_chunks_duration.append(
                    float(silence_duration_match.group('duration'))
                )

        total_silence_duration = sum(silence_chunks_duration)
        return (
            total_silence_duration / audio_duration
            >= silence_percentage_threshold
        )
