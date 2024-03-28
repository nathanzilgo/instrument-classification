from collections import Counter
import os

from settings import settings
from mir.modeling.feature_extractor import EssentiaExtractor
from mir.processing_pipelines import sample_and_filter_silence
from . import load_classifier_model
from .types.classifier_instrument import ClassifiedInstrument
from log import logging


class InstrumentClassifier:
    def __init__(self) -> None:
        self.extractor = EssentiaExtractor()

        model_path = os.path.join(
            settings.MODEL_OUTPUT_PATH, settings.MODEL_NAME
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError('Classification model is not installed.')

        self.model = load_classifier_model()

    def classify(self, filename: str) -> ClassifiedInstrument:
        track_id, input_format = filename.split('/')[-1].split('.')

        logging.info(f'Extracting {track_id} features...')
        output_path = os.path.join(settings.SAMPLE_OUTPUT_PATH, track_id)

        filtered_track = sample_and_filter_silence(
            track_path=filename,
            input_format=input_format,
            output_dir=output_path,
            output_basename=track_id,
            sample_duration=settings.SAMPLE_DURATION,
            silence_threshold=settings.SILENCE_THRESHOLD,
            silence_duration=settings.SILENCE_DURATION,
            silence_percentage=settings.SILENCE_PERCENTAGE,
            sample_proportion=settings.SAMPLE_PROPORTION,
            keep_trace=True,
        )

        if len(filtered_track) == 0:
            label = 'silence'
            logging.info(f'Classified "{track_id}" as "{label}"!')
            return ClassifiedInstrument(source_id=track_id, instrument=label)

        _, extracted_features = self.extractor.extract(
            filtered_track, save_df=False
        )
        logging.info(f'Features from {track_id} extracted with success!')

        predicted_labels = self.model.predict(
            extracted_features, threshold=settings.PREDICT_THRESHOLD
        )

        mode = Counter(predicted_labels)

        logging.info(f'Classifying "{track_id}"...')
        label = str(mode.most_common(1)[0][0])
        logging.info(f'Classified "{track_id}" as "{label}"!')

        return ClassifiedInstrument(source_id=track_id, instrument=label)