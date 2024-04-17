from pydantic import BaseSettings, BaseModel


class Environment(BaseModel):
    env: str

    def isLocal(self) -> bool:
        return self.env == 'local'

    def isProduction(self) -> bool:
        return self.env == 'production'


class Settings(BaseSettings):
    ENV: str = 'local'
    OUTPUT_PATH: str = './output'
    TRACKS_OUTPUT_PATH: str = f'{OUTPUT_PATH}/tracks'
    MODEL_OUTPUT_PATH: str = f'{OUTPUT_PATH}/models'
    SAMPLE_OUTPUT_PATH: str = f'{OUTPUT_PATH}/samples'
    SAMPLE_DURATION: int = 10000
    SILENCE_THRESHOLD: int = -45
    SILENCE_DURATION: int = 1
    SILENCE_PERCENTAGE: float = 0.3
    PREDICT_THRESHOLD: float = 0.5
    MAX_MESSAGES: int = 1
    SAMPLE_PROPORTION: float = 1.0

    IRMAS_PATH: str = './IRMAS'
    IRMAS_TRAIN_PATH: str = f'{IRMAS_PATH}/IRMAS-TrainingData'
    IRMAS_TEST_PATH: str = f'{IRMAS_PATH}/test'

    TRACKS_DIR: str = './tracks'

    MODEL_NAME: str = 'lgbm_retrained_00_46_21_27_03_2024.pkl'

    class Config:
        env_file = '.env'

    @property
    def env(self) -> Environment:
        return Environment(env=self.ENV)


settings = Settings()
