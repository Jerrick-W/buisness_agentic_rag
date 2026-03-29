"""FastAPI application entry point.

Registers all API routes, mounts static files, and initializes services.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Configure logging so startup errors are visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import validate_settings, Settings
from app.core.conversation import ConversationManager
from app.storage.database import init_db, close_db
from app.clients.deepseek_client import DeepSeekClient
from app.services.document_processor import DocumentProcessor
from app.models import ErrorResponse, MessageRole
from app.core.rag import RAGPipeline
from app.core.streaming import StreamingEngine
from app.storage.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global service instances (initialized in lifespan)
# ---------------------------------------------------------------------------
settings: Settings | None = None
conversation_mgr: ConversationManager | None = None
deepseek_client: DeepSeekClient | None = None
vector_store: MilvusVectorStore | None = None
doc_processor: DocumentProcessor | None = None
rag_pipeline: RAGPipeline | None = None
streaming_engine: StreamingEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, conversation_mgr, deepseek_client
    global vector_store, doc_processor, rag_pipeline, streaming_engine

    try:
        settings = validate_settings()
        await init_db(settings)
        conversation_mgr = ConversationManager(max_context_turns=settings.max_context_turns)
        deepseek_client = DeepSeekClient(settings=settings)
        vector_store = MilvusVectorStore(settings=settings)
        doc_processor = DocumentProcessor(vector_store, deepseek_client, settings)
        rag_pipeline = RAGPipeline(vector_store, deepseek_client, settings)
        streaming_engine = StreamingEngine(deepseek_client)

        logger.info("Services initialized — server ready")
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Failed to initialize services: %s", exc, exc_info=True)
        print(f"[FATAL] Startup error: {exc}", flush=True)
        raise

    yield

    if vector_store:
        vector_store.close()
    await close_db()


app = FastAPI(title="Enterprise AI Assistant", lifespan=lifespan)

# Mount static files
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_type="http_error",
            message=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type="internal_error",
            message="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session():
    assert conversation_mgr is not None
    sid = await conversation_mgr.create_session()
    return SessionResponse(session_id=sid)


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    assert conversation_mgr is not None
    if not conversation_mgr.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    history = await conversation_mgr.get_history(session_id)
    return [msg.model_dump() for msg in history]


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint."""
    assert conversation_mgr is not None
    assert rag_pipeline is not None
    assert streaming_engine is not None

    if not conversation_mgr.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    # Get context window
    context = await conversation_mgr.get_context_window(req.session_id)

    # RAG retrieval
    chunks = await rag_pipeline.retrieve(req.message)
    sources = rag_pipeline.results_to_sources(chunks)

    # Build prompt and get response
    messages = rag_pipeline.build_prompt(req.message, context, chunks)
    result = await streaming_engine.non_stream_chat(messages)

    if isinstance(result, dict):
        raise HTTPException(status_code=502, detail=result.get("description", "LLM error"))

    # Save to history
    await conversation_mgr.add_user_and_assistant(
        req.session_id, req.message, result, sources if sources else None
    )

    return {
        "response": result,
        "sources": [s.model_dump() for s in sources] if sources else [],
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint (SSE)."""
    assert conversation_mgr is not None
    assert rag_pipeline is not None
    assert streaming_engine is not None

    if not conversation_mgr.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    context = await conversation_mgr.get_context_window(req.session_id)
    logger.info("[stream] Got context, %d messages", len(context))

    chunks = await rag_pipeline.retrieve(req.message)
    logger.info("[stream] RAG retrieved %d chunks", len(chunks))

    sources = rag_pipeline.results_to_sources(chunks)
    messages = rag_pipeline.build_prompt(req.message, context, chunks)
    logger.info("[stream] Prompt built, starting stream")

    async def event_generator():
        full_response = []
        async for event in streaming_engine.stream_chat(messages, sources if sources else None):
            full_response.append(event)
            yield event

        # Extract the full text from token events
        import json
        response_text = ""
        for evt in full_response:
            try:
                data_str = evt.get("data", "") if isinstance(evt, dict) else evt
                data = json.loads(data_str)
                if data.get("type") == "token":
                    response_text += data.get("content", "")
            except (json.JSONDecodeError, KeyError, AttributeError):
                pass

        if response_text:
            await conversation_mgr.add_user_and_assistant(
                req.session_id, req.message, response_text,
                sources if sources else None,
            )

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for indexing."""
    assert doc_processor is not None

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        meta = await doc_processor.process_upload(file.filename, tmp_path, len(content))
        return meta.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/documents")
async def list_documents():
    assert doc_processor is not None
    docs = await doc_processor.list_documents()
    return [d.model_dump() for d in docs]


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    assert doc_processor is not None
    doc = await doc_processor.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc.model_dump()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    assert doc_processor is not None
    doc = await doc_processor.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    await doc_processor.delete_document(doc_id)
    return {"message": f"Document '{doc_id}' deleted"}


# ---------------------------------------------------------------------------
# Knowledge base stats
# ---------------------------------------------------------------------------

@app.get("/api/knowledge-base/stats")
async def knowledge_base_stats():
    assert vector_store is not None
    assert doc_processor is not None
    stats = await vector_store.get_stats()
    docs = await doc_processor.list_documents()
    return {
        "total_documents": len(docs),
        "total_chunks": sum(d.chunk_count for d in docs),
        "vector_dimension": stats.get("vector_dimension", 0),
        "collection_name": stats.get("collection_name", ""),
    }


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Enterprise AI Assistant</h1><p>Frontend not found.</p>")
