from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379"
    BATCH_TIMEOUT_MS: int = 15  # Sliding batch window threshold
    MAX_BATCH_SIZE: int = 4     # Tensor matrix horizontal request limit
    PRIMARY_ENGINE_FAIL_SIMULATION: bool = False

settings = Settings()