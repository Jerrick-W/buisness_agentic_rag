"""MySQL database module.

Provides async SQLAlchemy engine, session, and ORM models for document metadata.
Compatible with MySQL 5.7.
"""

from __future__ import annotations

import logging
from datetime import datetime

from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """Document metadata table."""

    __tablename__ = "documents"

    doc_id = Column(String(64), primary_key=True)
    filename = Column(String(256), nullable=False, index=True)
    file_type = Column(String(16), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="processing")
    error_message = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def _build_url(settings: Settings) -> str:
    """Build MySQL async connection URL with proper escaping."""
    password = quote_plus(settings.mysql_password)
    user = quote_plus(settings.mysql_user)
    return (
        f"mysql+aiomysql://{user}:{password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
        "?charset=utf8mb4"
    )


async def init_db(settings: Settings) -> None:
    """Initialize the database engine and create tables if needed."""
    global _engine, _session_factory

    url = _build_url(settings)
    _engine = create_async_engine(url, echo=False, pool_size=5, max_overflow=10)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    # Create tables (MySQL 5.7 compatible — no JSON columns)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("MySQL database initialized at %s:%s/%s", settings.mysql_host, settings.mysql_port, settings.mysql_database)


def get_session() -> AsyncSession:
    """Get a new async session."""
    assert _session_factory is not None, "Database not initialized. Call init_db() first."
    return _session_factory()


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
