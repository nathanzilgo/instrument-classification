import os

from mir.audio_processing import SampleOperation, FFmpegSilenceDetector


def sample_and_filter_silence(
    track_path,
    input_format,
    output_dir,
    output_basename,
    sample_duration,
    silence_threshold,
    silence_duration,
    silence_percentage,
    sample_proportion=1.0,
    keep_trace=False,
):

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

        SampleOperation().apply(
            audio_path=track_path,
            input_format=input_format,
            output_dir=output_dir,
            output_basename=output_basename,
            sample_duration=sample_duration,
            keep_trace=keep_trace,
            sample_proportion=sample_proportion,
        )

    not_silent_samples = []
    for sample_path in os.listdir(output_dir):
        full_sample_path = os.path.join(output_dir, sample_path)

        is_silence = False
        try:
            is_silence = FFmpegSilenceDetector.apply(
                full_sample_path,
                audio_duration=sample_duration,
                silence_threshold=silence_threshold,
                min_silence_duration=silence_duration,
                silence_percentage_threshold=silence_percentage,
            )
        except:
            continue

        if not is_silence:
            not_silent_samples.append(full_sample_path)

    return not_silent_samples
