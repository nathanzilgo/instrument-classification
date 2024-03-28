from pydantic import BaseSettings, BaseModel


class Environment(BaseModel):
    env: str

    def isMisd(self) -> bool:
        return self.env == 'misd'

    def isIrmas(self) -> bool:
        return self.env == 'irmas'


class Settings(BaseSettings):
    ENV: str = 'local'
    OUTPUT_PATH: str = './output'
    TRACKS_OUTPUT_PATH: str = f'{OUTPUT_PATH}/tracks'
    MODEL_OUTPUT_PATH: str = f'{OUTPUT_PATH}/model'
    SAMPLE_OUTPUT_PATH: str = f'{OUTPUT_PATH}/samples'
    SAMPLE_DURATION: int = 10000
    SILENCE_THRESHOLD: int = -45
    SILENCE_DURATION: int = 1
    SILENCE_PERCENTAGE: float = 0.3
    PREDICT_THRESHOLD: float = 0.7
    MAX_MESSAGES: int = 1
    SAMPLE_PROPORTION: float = 0.75

    IRMAS_PATH: str = './IRMAS'
    IRMAS_TRAIN_PATH: str = f'{IRMAS_PATH}/IRMAS-TrainingData'
    IRMAS_TEST_PATH: str = f'{IRMAS_PATH}/test'

    TRACKS_DIR: str = './tracks'

    class Config:
        env_file = '.env'

    @property
    def env(self) -> Environment:
        return Environment(env=self.ENV)


settings = Settings()
