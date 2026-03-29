"""Core data models for the Enterprise AI Assistant."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Role of a message participant."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocumentSource(BaseModel):
    """Reference to a source document chunk used in a response."""

    doc_id: str
    doc_name: str
    chunk_id: str
    chunk_text: str
    similarity_score: float


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime
    sources: list[DocumentSource] | None = None


class Session(BaseModel):
    """A conversation session."""

    session_id: str
    created_at: datetime
    messages: list[Message] = []


class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document."""

    doc_id: str
    filename: str
    file_type: str
    file_size: int
    upload_time: datetime
    chunk_count: int
    status: str  # "processing" | "ready" | "error"


class DocumentChunk(BaseModel):
    """A chunk of text from a document, optionally with its embedding."""

    chunk_id: str
    doc_id: str
    content: str
    embedding: list[float] | None = None
    chunk_index: int


class KnowledgeBaseStats(BaseModel):
    """Statistics about the knowledge base."""

    total_documents: int
    total_chunks: int
    vector_dimension: int
    collection_name: str


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error_type: str
    message: str
    detail: str | None = None
