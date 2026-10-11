# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile – FastAPI API Service
# ──────────────────────────────────────────────────────────────────────────────
# Lightweight image for the API server only (no ML libraries).
# Build: docker build -f Dockerfile -t intellicover-api .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install only the API-relevant dependencies (exclude heavy ML libs in prod API)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/ ./app/

# Create directories the app needs
RUN mkdir -p logs uploads && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start with Uvicorn (4 workers for production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
