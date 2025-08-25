"""Configuration management using pydantic-settings.

Handles environment variables and application configuration.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Database Configuration
    persistence_path: str = "data/vector_db.json"

    # Cohere API Configuration
    cohere_api_key: str | None = None
    cohere_model: str = "embed-english-v3.0"
    cohere_input_type: str = "search_document"

    # Index Configuration
    default_index_type: str = "linear"
    lsh_num_hashes: int = 10
    lsh_num_buckets: int = 100

    # Logging Configuration
    log_level: str = "info"
    log_file: str | None = None
    enable_file_logging: bool = False
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    log_max_size_mb: int = 10
    log_backup_count: int = 5

    # CORS Configuration
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    # Health Check Configuration
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3

    class Config:
        """Pydantic model configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
