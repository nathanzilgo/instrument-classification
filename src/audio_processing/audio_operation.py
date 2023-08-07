from abc import ABC, abstractmethod


class AudioOperation(ABC):
    @abstractmethod
    def apply(self, audio_path: str, **options):
        pass
