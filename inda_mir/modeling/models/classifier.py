from collections import Counter
import os

from inda_mir.modeling.models.lgbm import LightGBMClassifier
from settings import settings
from inda_mir.modeling.feature_extractor import EssentiaExtractor
from inda_mir.processing_pipelines import sample_and_filter_silence

from inda_mir.utils.logger import logger
from pydantic import BaseModel


class ClassifiedInstrument(BaseModel):
    source_id: str
    instrument: str
    probability: float | None

class InstrumentClassifier:
    def __init__(self, model: LightGBMClassifier) -> None:
        self.extractor = EssentiaExtractor()
        self.model = model
        # model_path = os.path.join(
        #     settings.MODEL_OUTPUT_PATH, settings.MODEL_NAME
        # )
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError('Classification model is not installed.')

        # self.model = load_classifier_model()

    def classify(self, filename: str) -> ClassifiedInstrument:
        print(filename)
        print(filename.split('/')[-1].split('.'))
        track_id, input_format = filename.split('/')[-1].split('.')

        if not os.path.exists(filename):
            raise FileNotFoundError(f'File {filename} not found.')

        logger.info(f'Extracting {track_id} features...')
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
            logger.info(f'Classified "{track_id}" as "{label}"!')
            return ClassifiedInstrument(source_id=track_id, instrument=label)

        _, extracted_features = self.extractor.extract(
            filtered_track, save_df=False
        )
        logger.info(f'Features from {track_id} extracted with success!')

        predicted_labels = self.model.predict(
            extracted_features, threshold=settings.PREDICT_THRESHOLD
        )

        mode = Counter(predicted_labels)

        logger.info(f'Classifying "{track_id}"...')
        label = str(mode.most_common(1)[0][0])
        logger.info(f'Classified "{track_id}" as "{label}"!')

        return ClassifiedInstrument(source_id=track_id, instrument=label)
