"""
IntelliCover AI – FastAPI Application Entrypoint
==================================================
Wires together:
  - Logging setup
  - Database lifecycle (connect on startup, dispose on shutdown)
  - CORS middleware
  - Global exception handlers
  - All API routers
  - OpenAPI documentation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.database import close_db_connection, create_all_tables
from app.core.logging_config import get_logger, setup_logging

# ── Logging must be set up BEFORE anything else logs ─────────────────────────
setup_logging()
logger = get_logger(__name__)


# ── Application Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    FastAPI recommends the lifespan context manager over @app.on_event decorators.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    if settings.DEBUG:
        # In development, auto-create tables (use Alembic in production)
        logger.warning(
            "auto_creating_tables",
            message="Auto-creating tables from ORM metadata. Use Alembic migrations in production.",
        )
        await create_all_tables()

    logger.info("application_startup_complete")
    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("application_shutting_down")
    await close_db_connection()
    logger.info("application_shutdown_complete")


# ── FastAPI App Instance ──────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "IntelliCover AI is an AI-powered career assistant that analyzes your "
        "resume against a job description, provides an ATS alignment score, "
        "actionable improvement suggestions, and generates a personalized cover letter."
    ),
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handlers ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.
    Returns a generic 500 response to the client while logging the full traceback.
    """
    import traceback

    logger.error(
        "unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Our team has been notified.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(api_v1_router)


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Health check endpoint")
async def health_check() -> dict:
    """
    Simple health check endpoint. Used by load balancers and monitoring tools.
    Returns 200 if the application is running.
    """
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


# ── Development server ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
