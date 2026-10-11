"""
Jobs API Routes  –  /api/v1/jobs
==================================
Implements the full async job lifecycle API:

  POST   /analyze                          – Submit resume + URL, start pipeline
  GET    /{job_id}/status                  – Poll for job results
  POST   /{job_id}/generate-cover-letter   – Start cover letter generation
  GET    /{job_id}/cover-letter/status     – Poll for cover letter result
"""

import uuid
from typing import Annotated, Optional

from pydantic import BaseModel

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging_config import get_logger
from app.models.db_models import AnalysisJob, CoverLetter, JobStatus, User
from app.schemas.schemas import (
    AnalysisJobCreateResponse,
    AnalysisJobStatusResponse,
    CoverLetterCreateResponse,
    CoverLetterStatusResponse,
    ParsedJDSchema,
    ParsedResumeSchema,
    ScoringBreakdown,
)
from app.services.auth_dependency import get_current_user
from app.services.storage_service import generate_storage_key, save_uploaded_file

router = APIRouter(prefix="/jobs", tags=["Analysis Jobs"])
logger = get_logger(__name__)

_ALLOWED_MIME_TYPES = {"application/pdf"}
_MAX_FILE_SIZE_MB = 10


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a resume and job URL to start an analysis",
)
async def submit_analysis(
    url: Annotated[str, Form(description="Public URL of the job description page")],
    resume: Annotated[UploadFile, File(description="Resume PDF file")],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJobCreateResponse:
    """
    Accepts a resume PDF and a job description URL.

    Actions:
      1. Validates file type and size.
      2. Saves the resume to storage, generates a storage key.
      3. Creates an AnalysisJob record (status=PENDING).
      4. Dispatches the Celery analysis task.

    Returns 202 immediately with the job_id.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if resume.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only PDF files are accepted. Got: {resume.content_type}",
        )

    # Check file size without fully loading into memory
    if resume.size and resume.size > _MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Resume file exceeds the {_MAX_FILE_SIZE_MB}MB limit.",
        )

    # ── Store file ────────────────────────────────────────────────────────────
    storage_key = generate_storage_key(str(current_user.id), resume.filename or "resume.pdf")
    await save_uploaded_file(resume, storage_key)

    # ── Create job record ─────────────────────────────────────────────────────
    job = AnalysisJob(
        user_id=current_user.id,
        resume_storage_key=storage_key,
        job_description_url=str(url),
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.flush()  # Populate job.id before calling Celery

    logger.info(
        "analysis_job_created",
        job_id=str(job.id),
        user_id=str(current_user.id),
        url=str(url),
    )

    # ── Dispatch Celery task ──────────────────────────────────────────────────
    from app.worker.tasks import run_analysis_task

    celery_task = run_analysis_task.delay(str(job.id))
    job.celery_task_id = celery_task.id

    logger.info(
        "analysis_task_dispatched",
        job_id=str(job.id),
        celery_task_id=celery_task.id,
    )

    return AnalysisJobCreateResponse(
        job_id=job.id,
        status=JobStatus.PENDING,
        message="Analysis job queued successfully. Poll /status for results.",
    )


# ── POST /{job_id}/retry ──────────────────────────────────────────────────────

class RetryAnalysisRequest(BaseModel):
    job_description_text: str


@router.post(
    "/{job_id}/retry",
    response_model=AnalysisJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a failed analysis with raw Job Description text",
)
async def retry_analysis(
    job_id: uuid.UUID,
    payload: RetryAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJobCreateResponse:
    """
    If JD scraping previously failed, user can submit the raw text here.
    Resets the job status to PENDING and re-runs the Celery worker.
    """
    result = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    # Reset job values for retry
    # We use a special prefix "text:" to indicate to the scraper to just use this text directly
    job.job_description_url = f"text:{payload.job_description_text}"
    job.status = JobStatus.PENDING
    job.error_message = None
    job.parsed_jd = None
    job.ats_score = None
    job.suggestions = None
    
    await db.flush()

    logger.info("analysis_job_retried_with_text", job_id=str(job.id))

    from app.worker.tasks import run_analysis_task
    celery_task = run_analysis_task.delay(str(job.id))
    job.celery_task_id = celery_task.id

    return AnalysisJobCreateResponse(
        job_id=job.id,
        status=JobStatus.PENDING,
        message="Job retried with raw text. Poll /status for results.",
    )


# ── GET /{job_id}/status ──────────────────────────────────────────────────────

@router.get(
    "/{job_id}/status",
    response_model=AnalysisJobStatusResponse,
    summary="Poll for the status and results of an analysis job",
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJobStatusResponse:
    """
    Returns the current status of the job.

    - PENDING / IN_PROGRESS: Check back in a few seconds.
    - SUCCESS: Full results included in the response.
    - FAILURE: error_message field explains what went wrong.
    """
    result = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found.",
        )

    response = AnalysisJobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        ats_score=float(job.ats_score) if job.ats_score else None,
        suggestions=job.suggestions,
        error_message=job.error_message,
    )

    # Include structured data when SUCCESS
    if job.status == JobStatus.SUCCESS and job.parsed_resume and job.parsed_jd:
        response.parsed_resume = ParsedResumeSchema(**job.parsed_resume)
        response.parsed_jd = ParsedJDSchema(**job.parsed_jd)

        if job.ats_score is not None:
            response.scoring_breakdown = ScoringBreakdown(
                hard_skill_score=float(job.hard_skill_score or 0),
                semantic_score=float(job.semantic_score or 0),
                education_score=float(job.education_score or 0),
                total_ats_score=float(job.ats_score),
                matched_skills=job.parsed_resume.get("skills", []),
                missing_skills=[],
            )

    return response


# ── POST /{job_id}/generate-cover-letter ─────────────────────────────────────

@router.post(
    "/{job_id}/generate-cover-letter",
    response_model=CoverLetterCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start cover letter generation for a completed analysis job",
)
async def generate_cover_letter(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CoverLetterCreateResponse:
    """
    Creates a CoverLetter record and dispatches the generation Celery task.

    Requires the corresponding AnalysisJob to have status=SUCCESS.
    Returns 409 if a cover letter already exists for this job.
    """
    result = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    if job.status != JobStatus.SUCCESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot generate a cover letter: analysis job is in status '{job.status}'. "
                "Only completed (SUCCESS) jobs can generate a cover letter."
            ),
        )

    # Check if cover letter already exists
    existing = await db.execute(
        select(CoverLetter).where(CoverLetter.job_id == job_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A cover letter already exists for this job. Fetch its status instead.",
        )

    # Create the cover letter record
    cover_letter = CoverLetter(
        job_id=job.id,
        generated_text="",  # will be populated by worker
        status=JobStatus.PENDING,
    )
    db.add(cover_letter)
    await db.flush()

    # Dispatch Celery task
    from app.worker.tasks import run_cover_letter_task

    celery_task = run_cover_letter_task.delay(str(cover_letter.id), str(job.id))
    cover_letter.celery_task_id = celery_task.id

    logger.info(
        "cover_letter_task_dispatched",
        cover_letter_id=str(cover_letter.id),
        job_id=str(job.id),
    )

    return CoverLetterCreateResponse(
        cover_letter_id=cover_letter.id,
        job_id=job.id,
        status=JobStatus.PENDING,
    )


# ── GET /{job_id}/cover-letter/status ────────────────────────────────────────

@router.get(
    "/{job_id}/cover-letter/status",
    response_model=CoverLetterStatusResponse,
    summary="Poll for the status of the cover letter generation",
)
async def get_cover_letter_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CoverLetterStatusResponse:
    """Poll for the status of a cover letter generation task."""
    # Verify job belongs to user
    job_result = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
    )
    if not job_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    cl_result = await db.execute(
        select(CoverLetter).where(CoverLetter.job_id == job_id)
    )
    cover_letter = cl_result.scalar_one_or_none()

    if not cover_letter:
        raise HTTPException(
            status_code=404,
            detail="No cover letter found for this job. Submit a generation request first.",
        )

    return CoverLetterStatusResponse(
        cover_letter_id=cover_letter.id,
        job_id=job_id,
        status=cover_letter.status,
        generated_text=cover_letter.generated_text if cover_letter.status == JobStatus.SUCCESS else None,
        error_message=cover_letter.error_message,
        created_at=cover_letter.created_at,
        completed_at=cover_letter.completed_at,
    )
