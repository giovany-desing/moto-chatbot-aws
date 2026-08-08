"""
Configuración central de la aplicación.
Carga todos los valores desde variables de entorno usando Pydantic Settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AWS
    AWS_REGION: str = "us-east-1"
    AWS_ACCOUNT_ID: str = ""

    # Proyecto
    PROJECT_NAME: str = "moto-chatbot"
    ENVIRONMENT: str = "dev"

    # RDS PostgreSQL
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str = "moto_chatbot"
    DB_USER: str
    DB_PASSWORD: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379

    # Bedrock
    BEDROCK_REGION: str = "us-east-1"
    LLM_MODEL: str = "us.amazon.nova-lite-v1:0"
    EMBEDDING_MODEL: str = "amazon.titan-embed-text-v2:0"
    EMBEDDING_DIMENSIONS: int = 1024

    # S3
    S3_BUCKET_NAME: str

    # SQS
    SQS_QUEUE_URL: str

    # RAG
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_MIN_SIZE: int = 100
    TOP_K_RESULTS: int = 5
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_SECONDS: int = 3600
    MEMORY_TTL_SECONDS: int = 1800
    MEMORY_MAX_MESSAGES: int = 6

    # Contextual Retrieval
    CONTEXTUAL_RETRIEVAL_ENABLED: bool = False
    CONTEXT_MAX_DOC_CHARS: int = 6000

    # Reranking
    RERANKING_ENABLED: bool = True
    RERANK_MODEL: str = "amazon.rerank-v1:0"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
