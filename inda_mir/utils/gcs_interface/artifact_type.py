from enum import Enum


class ArtifactType(Enum):
    RAW = 'raw'
    SAMPLES = 'samples'
    FEATURES = 'features'
    TTS = 'tts'
    METADATA = 'metadata'
    MODEL = 'model'
    METRICS = 'metrics'
