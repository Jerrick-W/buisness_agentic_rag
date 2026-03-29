"""Document processor module.

Handles document upload, text extraction, chunking, embedding generation,
vector index management, and metadata persistence to MySQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select, delete as sa_delete

from app.config import Settings
from app.storage.database import DocumentRecord, get_session
from app.clients.deepseek_client import DeepSeekClient
from app.models import DocumentMetadata
from app.storage.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "txt", "md"}


def _record_to_meta(r: DocumentRecord) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=r.doc_id,
        filename=r.filename,
        file_type=r.file_type,
        file_size=r.file_size,
        upload_time=r.upload_time,
        chunk_count=r.chunk_count,
        status=r.status,
    )


class DocumentProcessor:
    """Processes uploaded documents: extract, chunk, embed, index."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        deepseek_client: DeepSeekClient,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._deepseek = deepseek_client
        self._settings = settings or Settings()  # type: ignore[call-arg]

    @staticmethod
    def is_supported(filename: str) -> bool:
        ext = Path(filename).suffix.lstrip(".").lower()
        return ext in SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: str, file_type: str) -> str:
        ext = file_type.lower().lstrip(".")
        if ext == "pdf":
            return self._extract_pdf(file_path)
        elif ext in ("txt", "md"):
            return Path(file_path).read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        text_parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    def chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        return splitter.split_text(text)

    async def process_upload(
        self,
        filename: str,
        file_path: str,
        file_size: int,
    ) -> DocumentMetadata:
        ext = Path(filename).suffix.lstrip(".").lower()
        doc_id = str(uuid.uuid4())

        # Handle same-name replacement
        existing = await self._find_by_filename(filename)
        if existing:
            await self.delete_document(existing.doc_id)

        # Create DB record
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            file_type=ext,
            file_size=file_size,
            upload_time=datetime.now(timezone.utc),
            chunk_count=0,
            status="processing",
        )
        async with get_session() as session:
            session.add(record)
            await session.commit()

        try:
            text = self.extract_text(file_path, ext)
            chunks = self.chunk_text(text)

            embeddings: list[list[float]] = []
            for chunk in chunks:
                emb = await self._deepseek.create_embedding(chunk)
                if isinstance(emb, dict):
                    raise RuntimeError(f"Embedding failed: {emb}")
                embeddings.append(emb)

            await self._vector_store.insert(doc_id, filename, chunks, embeddings)

            # Update record
            async with get_session() as session:
                record = await session.get(DocumentRecord, doc_id)
                if record:
                    record.chunk_count = len(chunks)
                    record.status = "ready"
                    await session.commit()

            logger.info("Document '%s' processed: %d chunks", filename, len(chunks))

        except Exception as exc:
            async with get_session() as session:
                record = await session.get(DocumentRecord, doc_id)
                if record:
                    record.status = "error"
                    record.error_message = str(exc)[:500]
                    await session.commit()
            logger.error("Failed to process document '%s': %s", filename, exc)
            raise

        return DocumentMetadata(
            doc_id=doc_id, filename=filename, file_type=ext,
            file_size=file_size, upload_time=datetime.now(timezone.utc),
            chunk_count=len(chunks), status="ready",
        )

    async def delete_document(self, doc_id: str) -> None:
        await self._vector_store.delete_by_doc_id(doc_id)
        async with get_session() as session:
            await session.execute(sa_delete(DocumentRecord).where(DocumentRecord.doc_id == doc_id))
            await session.commit()
        logger.info("Document '%s' deleted", doc_id)

    async def get_document(self, doc_id: str) -> DocumentMetadata | None:
        async with get_session() as session:
            record = await session.get(DocumentRecord, doc_id)
            return _record_to_meta(record) if record else None

    async def list_documents(self) -> list[DocumentMetadata]:
        async with get_session() as session:
            result = await session.execute(select(DocumentRecord).order_by(DocumentRecord.upload_time.desc()))
            return [_record_to_meta(r) for r in result.scalars().all()]

    async def _find_by_filename(self, filename: str) -> DocumentMetadata | None:
        async with get_session() as session:
            result = await session.execute(
                select(DocumentRecord).where(DocumentRecord.filename == filename)
            )
            record = result.scalar_one_or_none()
            return _record_to_meta(record) if record else None
