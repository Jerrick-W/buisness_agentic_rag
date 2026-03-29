"""Milvus vector store module.

Wraps pymilvus operations: collection management, insert, search, delete, stats.
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from app.config import Settings

logger = logging.getLogger(__name__)

# Default embedding dimension (DeepSeek embedding)
_DEFAULT_DIM = 1024


class SearchResult:
    """A single vector search result."""

    def __init__(self, chunk_id: str, doc_id: str, doc_name: str, content: str, chunk_index: int, score: float):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.content = content
        self.chunk_index = chunk_index
        self.score = score


class MilvusVectorStore:
    """Async-friendly wrapper around Milvus for document chunk storage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()  # type: ignore[call-arg]
        self._uri = self._settings.milvus_uri
        self._collection = self._settings.milvus_collection
        self._client: MilvusClient | None = None
        self._available = True  # False if connection failed

    def _get_client(self) -> MilvusClient | None:
        if not self._available:
            return None
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self._uri, timeout=5)
                self._ensure_collection()
                logger.info("Connected to Milvus at %s", self._uri)
            except Exception as exc:
                logger.warning("Milvus not available at %s: %s", self._uri, exc)
                self._available = False
                self._client = None
                return None
        return self._client

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        client = self._client
        assert client is not None

        if client.has_collection(self._collection):
            # Check if dimension matches, drop and recreate if not
            try:
                info = client.describe_collection(self._collection)
                for field in info.get("fields", []):
                    if field.get("name") == "embedding":
                        existing_dim = field.get("params", {}).get("dim", 0)
                        if existing_dim and int(existing_dim) != _DEFAULT_DIM:
                            logger.warning(
                                "Collection dim mismatch (existing=%s, expected=%s), recreating",
                                existing_dim, _DEFAULT_DIM,
                            )
                            client.drop_collection(self._collection)
                            break
                else:
                    client.load_collection(collection_name=self._collection)
                    return
            except Exception:
                client.load_collection(collection_name=self._collection)
                return

        schema = CollectionSchema(fields=[
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DEFAULT_DIM),
        ])

        client.create_collection(
            collection_name=self._collection,
            schema=schema,
        )

        # Create IVF_FLAT index on embedding field
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        client.create_index(
            collection_name=self._collection,
            index_params=index_params,
        )
        client.load_collection(collection_name=self._collection)
        logger.info("Created Milvus collection '%s' with IVF_FLAT index", self._collection)

    async def insert(
        self,
        doc_id: str,
        doc_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Insert document chunks with their embeddings."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Milvus is not available")
        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            data.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "content": chunk,
                "chunk_index": i,
                "embedding": emb,
            })
        if data:
            client.insert(collection_name=self._collection, data=data)
            logger.info("Inserted %d chunks for doc '%s'", len(data), doc_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar chunks by embedding vector."""
        try:
            client = self._get_client()
            if client is None:
                return []
            results = client.search(
                collection_name=self._collection,
                data=[query_embedding],
                limit=top_k,
                output_fields=["chunk_id", "doc_id", "doc_name", "content", "chunk_index"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            )
        except Exception as exc:
            logger.warning("Milvus search failed: %s", exc)
            return []

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                search_results.append(SearchResult(
                    chunk_id=entity.get("chunk_id", ""),
                    doc_id=entity.get("doc_id", ""),
                    doc_name=entity.get("doc_name", ""),
                    content=entity.get("content", ""),
                    chunk_index=entity.get("chunk_index", 0),
                    score=hit.get("distance", 0.0),
                ))
        return search_results

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Milvus is not available")
        client.delete(
            collection_name=self._collection,
            filter=f'doc_id == "{doc_id}"',
        )
        logger.info("Deleted all chunks for doc '%s'", doc_id)

    async def get_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        client = self._get_client()
        if client is None:
            return {"collection_name": self._collection, "row_count": 0, "vector_dimension": _DEFAULT_DIM}
        stats = client.get_collection_stats(self._collection)
        return {
            "collection_name": self._collection,
            "row_count": stats.get("row_count", 0),
            "vector_dimension": _DEFAULT_DIM,
        }

    def close(self) -> None:
        """Close the Milvus client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
