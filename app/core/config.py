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

    # Seguridad
    API_KEY: str

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

    # RAG -- chunking recursivo basado en TOKENS (no caracteres), via
    # tiktoken cl100k_base como aproximacion estandar del tamano real
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    CHUNK_MIN_SIZE_TOKENS: int = 20
    TOP_K_RESULTS: int = 5
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_SECONDS: int = 3600
    MEMORY_TTL_SECONDS: int = 1800
    MEMORY_MAX_MESSAGES: int = 6

    # Contextual Retrieval
    CONTEXTUAL_RETRIEVAL_ENABLED: bool = True
    CONTEXT_MAX_DOC_CHARS: int = 2000

    # Reranking: proveedor conmutable (local = BGE reranker, bedrock = casos atípicos)
    RERANKING_ENABLED: bool = True
    RERANK_PROVIDER: str = "local"
    RERANK_MODEL: str = "amazon.rerank-v1:0"
    LOCAL_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # LLM de chat: proveedor conmutable (groq = producción, bedrock = casos atípicos)
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_REASONING_EFFORT: str = "low"  # low/medium/high -- low reduce tokens y evita respuestas vacias con max_tokens ajustado

    # Embeddings: proveedor conmutable (local = BGE-M3, bedrock = Titan)
    EMBEDDING_PROVIDER: str = "local"
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-m3"

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
