from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str, output_path: str, **options) -> any:
        ...
