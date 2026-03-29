"""Property-based tests for the configuration module.

# Feature: enterprise-ai-assistant, Property 16: 配置加载正确性
Validates that the Settings model correctly parses environment variables.
Requirement: 8.1
"""

import os
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from app.config import Settings

# --- Strategies ---

# Strategy for valid string config values (non-empty, stripped)
_non_empty_str = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())

_url_str = st.sampled_from([
    "https://api.deepseek.com",
    "http://localhost:8080",
    "https://custom.endpoint.io/v1",
])

_positive_int = st.integers(min_value=1, max_value=10000)
_threshold_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(
    api_key=_non_empty_str,
    base_url=_url_str,
    chat_model=_non_empty_str,
    embedding_model=_non_empty_str,
    milvus_uri=_url_str,
    milvus_collection=_non_empty_str,
    rag_top_k=_positive_int,
    rag_threshold=_threshold_float,
    chunk_size=_positive_int,
    chunk_overlap=_positive_int,
    max_turns=_positive_int,
    port=st.integers(min_value=1, max_value=65535),
)
@settings(max_examples=100)
def test_settings_parses_env_vars_correctly(
    api_key: str,
    base_url: str,
    chat_model: str,
    embedding_model: str,
    milvus_uri: str,
    milvus_collection: str,
    rag_top_k: int,
    rag_threshold: float,
    chunk_size: int,
    chunk_overlap: int,
    max_turns: int,
    port: int,
) -> None:
    """Property 16: For any set of environment variable values, the Settings
    model should parse and load the corresponding configuration correctly."""

    env = {
        "DEEPSEEK_API_KEY": api_key,
        "DEEPSEEK_BASE_URL": base_url,
        "DEEPSEEK_CHAT_MODEL": chat_model,
        "DEEPSEEK_EMBEDDING_MODEL": embedding_model,
        "MILVUS_URI": milvus_uri,
        "MILVUS_COLLECTION": milvus_collection,
        "RAG_TOP_K": str(rag_top_k),
        "RAG_SIMILARITY_THRESHOLD": str(rag_threshold),
        "CHUNK_SIZE": str(chunk_size),
        "CHUNK_OVERLAP": str(chunk_overlap),
        "MAX_CONTEXT_TURNS": str(max_turns),
        "PORT": str(port),
    }

    with patch.dict(os.environ, env, clear=False):
        s = Settings(**{k.lower(): v for k, v in env.items()})

        assert s.deepseek_api_key == api_key.strip()
        assert s.deepseek_base_url == base_url
        assert s.deepseek_chat_model == chat_model
        assert s.deepseek_embedding_model == embedding_model
        assert s.milvus_uri == milvus_uri
        assert s.milvus_collection == milvus_collection
        assert s.rag_top_k == rag_top_k
        assert s.rag_similarity_threshold == rag_threshold
        assert s.chunk_size == chunk_size
        assert s.chunk_overlap == chunk_overlap
        assert s.max_context_turns == max_turns
        assert s.port == port


@given(api_key=_non_empty_str)
@settings(max_examples=100)
def test_settings_defaults_applied_when_only_required_set(api_key: str) -> None:
    """Property 16 (supplement): When only the required env var is provided,
    all other fields should fall back to their documented defaults."""

    s = Settings(deepseek_api_key=api_key)

    assert s.deepseek_api_key == api_key.strip()
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_chat_model == "deepseek-chat"
    assert s.deepseek_embedding_model == "deepseek-embedding"
    assert s.milvus_uri == "http://localhost:19530"
    assert s.milvus_collection == "document_chunks"
    assert s.rag_top_k == 5
    assert s.rag_similarity_threshold == 0.7
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.max_context_turns == 20
    assert s.host == "0.0.0.0"
    assert s.port == 8000
