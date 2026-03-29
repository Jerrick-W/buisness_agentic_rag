"""RAG pipeline module.

Coordinates vector retrieval and prompt construction for
retrieval-augmented generation.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.clients.deepseek_client import DeepSeekClient
from app.models import DocumentSource, Message
from app.storage.vector_store import MilvusVectorStore, SearchResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        deepseek_client: DeepSeekClient,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._deepseek = deepseek_client
        self._settings = settings or Settings()  # type: ignore[call-arg]

    async def retrieve(self, query: str) -> list[SearchResult]:
        """Retrieve relevant document chunks for a query."""
        try:
            emb = await self._deepseek.create_embedding(query)
            if isinstance(emb, dict):
                logger.error("Embedding generation failed: %s", emb)
                return []
            logger.info("[RAG] Embedding generated, dim=%d", len(emb))

            results = await self._vector_store.search(
                query_embedding=emb,
                top_k=self._settings.rag_top_k,
            )
            logger.info("[RAG] Milvus returned %d raw results", len(results))
            for r in results:
                logger.info("[RAG]   chunk=%s doc=%s score=%.4f content=%.60s...", r.chunk_id, r.doc_name, r.score, r.content)

            # Filter by similarity threshold
            threshold = self._settings.rag_similarity_threshold
            filtered = [r for r in results if r.score >= threshold]
            logger.info("[RAG] After threshold filter (>=%.2f): %d results", threshold, len(filtered))

            filtered.sort(key=lambda r: r.score, reverse=True)
            return filtered
        except Exception as exc:
            logger.warning("RAG retrieve failed (non-fatal): %s", exc)
            return []

    def build_prompt(
        self,
        query: str,
        context: list[Message],
        chunks: list[SearchResult],
    ) -> list[dict[str, str]]:
        """Build the message list for the LLM, injecting retrieved context."""
        messages: list[dict[str, str]] = []

        # System prompt
        if chunks:
            sources_text = "\n\n".join(
                f"[来源: {c.doc_name}, 片段 {c.chunk_index}]\n{c.content}"
                for c in chunks
            )
            system_content = (
                "你是一个企业级智能助手。请基于以下知识库内容回答用户问题，"
                "并在回复中标注引用的文档来源。如果知识库内容不足以回答，"
                "请基于自身知识补充，并说明哪些内容来自知识库。\n\n"
                f"--- 知识库内容 ---\n{sources_text}\n--- 知识库内容结束 ---"
            )
        else:
            system_content = (
                "你是一个企业级智能助手。当前未检索到相关知识库内容，"
                "请基于自身知识回答用户问题，并告知用户未检索到相关知识库内容。"
            )

        messages.append({"role": "system", "content": system_content})

        # Conversation history
        for msg in context:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Current query
        messages.append({"role": "user", "content": query})
        return messages

    @staticmethod
    def results_to_sources(chunks: list[SearchResult]) -> list[DocumentSource]:
        """Convert search results to DocumentSource models."""
        return [
            DocumentSource(
                doc_id=c.doc_id,
                doc_name=c.doc_name,
                chunk_id=c.chunk_id,
                chunk_text=c.content,
                similarity_score=c.score,
            )
            for c in chunks
        ]
