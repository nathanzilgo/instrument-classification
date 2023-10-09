import os

from inda_mir.audio_processing import SampleOperation, FFmpegSilenceDetector


def sample_and_filter_silence(
    track_path,
    input_format,
    output_dir,
    output_basename,
    sample_duration,
    silence_threshold,
    silence_duration,
    silence_percentage,
):

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

        SampleOperation.apply(
            audio_path=track_path,
            input_format=input_format,
            output_dir=output_dir,
            output_basename=output_basename,
            sample_duration=sample_duration,
        )

    not_silent_samples = []
    for sample_path in os.listdir(output_dir):
        full_sample_path = os.path.join(output_dir, sample_path)
        if not FFmpegSilenceDetector.apply(
            full_sample_path,
            audio_duration=sample_duration,
            silence_threshold=silence_threshold,
            min_silence_duration=silence_duration,
            silence_percentage_threshold=silence_percentage,
        ):
            not_silent_samples.append(full_sample_path)

    return not_silent_samples
