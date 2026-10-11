"""
Celery Application Configuration
==================================
Defines and configures the Celery application instance.

This module is the entrypoint for the Celery worker process.
It's imported by both:
  1. The FastAPI app (to dispatch tasks)
  2. The Celery worker CLI (to execute tasks)

Start the worker with:
  celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "intellicover",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"],  # auto-discovers task modules
)

# ── Celery configuration ──────────────────────────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task behavior
    task_track_started=True,
    task_acks_late=True,  # only ack after task completes (safer on crash)
    worker_prefetch_multiplier=1,  # one task at a time per worker (memory-heavy)
    # Result expiry (keep task results for 24 hours)
    result_expires=60 * 60 * 24,
    # Retry settings
    task_max_retries=3,
    task_default_retry_delay=60,  # 1 minute between retries
    # Beat (periodic tasks — for future use like cleanup jobs)
    beat_schedule={},
)
