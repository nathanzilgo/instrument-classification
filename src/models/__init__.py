from .base_model import BaseModel
from .lgbm import LightGBMClassifier

def load_classifier_model() -> BaseModel:
    logging.info('Loading model...')
    model_path = os.path.join(settings.MODEL_OUTPUT_PATH, settings.MODEL_NAME)
    model = load_model(model_path)
    logging.info('Model loaded!')

    return model

def load_model(path) -> BaseModel:
    return pickle.load(open(path, 'rb'))
