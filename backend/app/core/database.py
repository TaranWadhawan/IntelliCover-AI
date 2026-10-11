"""
Async SQLAlchemy database engine and session factory.

Uses asyncpg driver for high-performance async PostgreSQL access.
Session dependency is injected per-request via FastAPI's DI system.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── SQLAlchemy engine ─────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG,  # logs SQL in debug mode
    future=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Declarative Base for ORM models ──────────────────────────────────────────
class Base(DeclarativeBase):
    """Parent class for all SQLAlchemy ORM models."""
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session.

    Each HTTP request gets its own session, which is always closed in the
    finally block — even on exception.

    Usage:
        @router.get("/example")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle helpers ─────────────────────────────────────────────────────────
async def create_all_tables() -> None:
    """
    Create all tables from ORM metadata.
    Should only be used in development; use Alembic migrations in production.
    """
    async with engine.begin() as conn:
        from app.models import db_models  # noqa: F401 – registers all models
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def close_db_connection() -> None:
    """Dispose the engine connection pool on shutdown."""
    await engine.dispose()
    logger.info("database_connection_pool_closed")
