from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    data_dir: str = "../data"
    host: str = "0.0.0.0"
    port: int = 8080
    model_config = {"env_prefix": "INTERPRET_"}


settings = Settings()
