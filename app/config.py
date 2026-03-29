"""Application configuration module.

Uses pydantic-settings to load configuration from environment variables and .env files.
"""

import logging
import sys

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # DeepSeek configuration
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"

    # Qwen Embedding configuration (阿里云通义千问)
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode"
    embedding_model: str = "text-embedding-v3"

    # MySQL configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "ai_assistant"

    # Milvus configuration
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "document_chunks"

    # RAG configuration
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Conversation configuration
    max_context_turns: int = 20

    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("deepseek_api_key")
    @classmethod
    def validate_deepseek_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "deepseek_api_key is required. "
                "Set it via the DEEPSEEK_API_KEY environment variable or in the .env file."
            )
        return v.strip()


def validate_settings() -> Settings:
    """Load and validate settings at startup.

    Returns the Settings instance if valid, otherwise raises RuntimeError.
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:
        logger.error("Configuration validation failed: %s", exc)
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr, flush=True)
        raise RuntimeError(f"Configuration error: {exc}") from exc
