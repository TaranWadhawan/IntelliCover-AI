"""
Celery Tasks – The Orchestration Layer
========================================
Contains all Celery task definitions. Each task is the "glue" that
sequences calls to the NLP pipeline modules in the correct order.

Two tasks:
  1. run_analysis_task    – Full NLP pipeline (parse → score → suggest)
  2. run_cover_letter_task– Cover letter generation

Both tasks update the database with intermediate and final results,
and handle exceptions by writing to the error_message field.

IMPORTANT: These tasks run in a SEPARATE process (the Celery worker).
They do NOT share the FastAPI event loop. They use synchronous SQLAlchemy.
"""

import traceback
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging
from app.models.db_models import AnalysisJob, CoverLetter, JobStatus
from app.nlp.jd_scraper import parse_job_description, scrape_job_description
from app.nlp.llm_service import LLMService
from app.nlp.resume_parser import parse_resume
from app.nlp.scorer import run_scoring_pipeline
from app.nlp.spacy_loader import get_nlp_model
from app.services.storage_service import get_local_file_path
from app.worker.celery_app import celery_app

# ── Setup ─────────────────────────────────────────────────────────────────────

# Workers need their own logging setup (separate process from FastAPI)
setup_logging()
logger = get_logger(__name__)

# Synchronous SQLAlchemy engine for Celery workers
# (asyncpg doesn't work in sync Celery context)
_sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
sync_engine = create_engine(_sync_db_url, pool_pre_ping=True)
SyncSession = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)


def _get_sync_db() -> Session:
    return SyncSession()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_job_status(
    db: Session,
    job_id: str,
    status: str,
    **kwargs,
) -> None:
    """Update an AnalysisJob record with new status and optional field values."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
    if not job:
        logger.error("job_not_found_in_worker", job_id=job_id)
        return

    job.status = status
    for key, value in kwargs.items():
        setattr(job, key, value)

    if status in (JobStatus.SUCCESS, JobStatus.FAILURE):
        job.completed_at = datetime.now(timezone.utc)

    db.commit()
    logger.info("job_status_updated", job_id=job_id, status=status)


def _update_cover_letter_status(
    db: Session,
    cover_letter_id: str,
    status: str,
    **kwargs,
) -> None:
    """Update a CoverLetter record."""
    cl = db.query(CoverLetter).filter(
        CoverLetter.id == uuid.UUID(cover_letter_id)
    ).first()
    if not cl:
        return

    cl.status = status
    for key, value in kwargs.items():
        setattr(cl, key, value)

    if status in (JobStatus.SUCCESS, JobStatus.FAILURE):
        cl.completed_at = datetime.now(timezone.utc)

    db.commit()


# ── Task 1: Full Analysis Pipeline ────────────────────────────────────────────

@celery_app.task(
    name="app.worker.tasks.run_analysis_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_analysis_task(self, job_id: str) -> dict:
    """
    Execute the full NLP analysis pipeline for a given job_id.

    Pipeline stages:
      1. Fetch job record + update status to IN_PROGRESS
      2. Download/resolve resume file path
      3. Parse resume PDF → structured JSON
      4. Scrape job description URL → raw text
      5. Parse JD raw text → structured JSON
      6. Persist parsed data to DB
      7. Run scoring pipeline (hard skill + semantic + education)
      8. Generate AI suggestions (LLM)
      9. Persist final results → set status to SUCCESS

    On any exception: set status to FAILURE and write error message.
    """
    db: Session = _get_sync_db()
    logger.info("analysis_task_started", job_id=job_id, celery_task_id=self.request.id)

    try:
        # ── Stage 1: Mark as IN_PROGRESS ─────────────────────────────────────
        _update_job_status(
            db, job_id,
            status=JobStatus.IN_PROGRESS,
            celery_task_id=self.request.id,
        )

        job = db.query(AnalysisJob).filter(
            AnalysisJob.id == uuid.UUID(job_id)
        ).first()
        if not job:
            raise ValueError(f"Job {job_id} not found in database.")

        # ── Stage 2: Resolve resume file path ────────────────────────────────
        resume_path = get_local_file_path(job.resume_storage_key)
        logger.info("resume_path_resolved", job_id=job_id, path=resume_path)

        # ── Stage 3: Parse Resume ─────────────────────────────────────────────
        nlp = get_nlp_model()
        parsed_resume = parse_resume(resume_path, nlp)
        logger.info("resume_parsed", job_id=job_id, skills=len(parsed_resume.skills))

        # ── Stage 4: Scrape Job Description ──────────────────────────────────
        import asyncio
        jd_raw_text = asyncio.run(
            scrape_job_description(str(job.job_description_url))
        )

        # ── Stage 5: Parse Job Description ───────────────────────────────────
        parsed_jd = parse_job_description(jd_raw_text, nlp)
        logger.info(
            "jd_parsed",
            job_id=job_id,
            job_title=parsed_jd.job_title,
            required_skills=len(parsed_jd.required_skills),
        )

        # ── Stage 6: Persist parsed data to DB ───────────────────────────────
        _update_job_status(
            db, job_id,
            status=JobStatus.IN_PROGRESS,
            parsed_resume=parsed_resume.model_dump(),
            parsed_jd=parsed_jd.model_dump(),
        )

        # ── Stage 7: Run Scoring Pipeline ────────────────────────────────────
        scoring, gap = run_scoring_pipeline(parsed_resume, parsed_jd)
        logger.info(
            "scoring_complete",
            job_id=job_id,
            ats_score=scoring.total_ats_score,
        )

        # ── Stage 8: Generate AI Suggestions ─────────────────────────────────
        llm_service = LLMService.get_instance()
        suggestions = llm_service.generate_suggestions(
            parsed_resume,
            parsed_jd,
            gap,
            jd_full_text=jd_raw_text,
        )
        logger.info("suggestions_generated", job_id=job_id, count=len(suggestions))

        # ── Stage 9: Persist final results ───────────────────────────────────
        _update_job_status(
            db, job_id,
            status=JobStatus.SUCCESS,
            ats_score=float(scoring.total_ats_score),
            hard_skill_score=float(scoring.hard_skill_score),
            semantic_score=float(scoring.semantic_score),
            education_score=float(scoring.education_score),
            suggestions=suggestions,
        )

        logger.info("analysis_task_complete", job_id=job_id)
        return {"status": "SUCCESS", "job_id": job_id, "ats_score": scoring.total_ats_score}

    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {str(exc)}"
        tb = traceback.format_exc()
        logger.error(
            "analysis_task_failed",
            job_id=job_id,
            error=error_detail,
            traceback=tb,
        )
        _update_job_status(
            db, job_id,
            status=JobStatus.FAILURE,
            error_message=error_detail,
        )
        # Re-raise so Celery marks it as failed and can retry
        raise

    finally:
        db.close()


# ── Task 2: Cover Letter Generation ──────────────────────────────────────────

@celery_app.task(
    name="app.worker.tasks.run_cover_letter_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_cover_letter_task(self, cover_letter_id: str, job_id: str) -> dict:
    """
    Generate a personalized cover letter for a completed analysis job.

    Requires:
      - The associated AnalysisJob must have status=SUCCESS.
      - Parsed resume and JD data must be stored in the job record.

    Pipeline:
      1. Fetch job record → extract parsed_resume and parsed_jd
      2. Call LLMService.generate_cover_letter()
      3. Persist generated text → set status to SUCCESS
    """
    db: Session = _get_sync_db()
    logger.info(
        "cover_letter_task_started",
        cover_letter_id=cover_letter_id,
        job_id=job_id,
    )

    try:
        # ── Stage 1: Fetch job data ───────────────────────────────────────────
        _update_cover_letter_status(
            db, cover_letter_id,
            status=JobStatus.IN_PROGRESS,
            celery_task_id=self.request.id,
        )

        job = db.query(AnalysisJob).filter(
            AnalysisJob.id == uuid.UUID(job_id)
        ).first()

        if not job:
            raise ValueError(f"Job {job_id} not found.")
        if job.status != JobStatus.SUCCESS:
            raise ValueError(
                f"Cannot generate cover letter: job {job_id} is in status '{job.status}'."
            )
        if not job.parsed_resume or not job.parsed_jd:
            raise ValueError(f"Job {job_id} has no parsed data.")

        from app.schemas.schemas import ParsedJDSchema, ParsedResumeSchema

        parsed_resume = ParsedResumeSchema(**job.parsed_resume)
        parsed_jd = ParsedJDSchema(**job.parsed_jd)

        # ── Stage 2: Generate cover letter ───────────────────────────────────
        llm_service = LLMService.get_instance()
        cover_letter_text = llm_service.generate_cover_letter(
            parsed_resume,
            parsed_jd,
            jd_full_text=parsed_jd.raw_text,
        )

        # ── Stage 3: Persist ──────────────────────────────────────────────────
        _update_cover_letter_status(
            db, cover_letter_id,
            status=JobStatus.SUCCESS,
            generated_text=cover_letter_text,
        )

        logger.info("cover_letter_task_complete", cover_letter_id=cover_letter_id)
        return {"status": "SUCCESS", "cover_letter_id": cover_letter_id}

    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {str(exc)}"
        logger.error(
            "cover_letter_task_failed",
            cover_letter_id=cover_letter_id,
            error=error_detail,
            traceback=traceback.format_exc(),
        )
        _update_cover_letter_status(
            db, cover_letter_id,
            status=JobStatus.FAILURE,
            error_message=error_detail,
        )
        raise

    finally:
        db.close()
