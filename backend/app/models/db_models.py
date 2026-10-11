"""
SQLAlchemy ORM models — the authoritative database schema.

Maps directly to the PostgreSQL tables defined in the project spec.
All UUIDs are generated server-side by Python (not the DB) to ensure
they are available immediately after insert, before a DB roundtrip.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── Users ─────────────────────────────────────────────────────────────────────
class User(Base):
    """
    Represents an authenticated user account.
    One user → many AnalysisJob records.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=_now_utc, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=_now_utc,
        onupdate=_now_utc,
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    jobs: Mapped[List["AnalysisJob"]] = relationship(
        "AnalysisJob",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


# ── Analysis Jobs ─────────────────────────────────────────────────────────────
class JobStatus:
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class AnalysisJob(Base):
    """
    Central record for every analysis task.

    Tracks:
      - Input: uploaded resume path and job URL.
      - Processing: Celery task ID and status.
      - Output: structured parsed data, ATS score, AI suggestions.
    """

    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    resume_storage_key: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # S3 key or local path
    job_description_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # ── Parsed Structured Data ─────────────────────────────────────────────────
    parsed_resume: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    parsed_jd: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # ── Analysis Results ──────────────────────────────────────────────────────
    ats_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2), nullable=True
    )  # e.g. 85.50
    hard_skill_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2), nullable=True
    )
    semantic_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2), nullable=True
    )
    education_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2), nullable=True
    )
    suggestions: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=_now_utc, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="jobs")
    cover_letter: Mapped[Optional["CoverLetter"]] = relationship(
        "CoverLetter",
        back_populates="job",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<AnalysisJob id={self.id} status={self.status}>"


# ── Cover Letters ─────────────────────────────────────────────────────────────
class CoverLetter(Base):
    """
    Stores the AI-generated cover letter for a successful analysis job.
    One AnalysisJob → at most one CoverLetter.
    """

    __tablename__ = "cover_letters"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_cover_letters_job_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # ── Content ───────────────────────────────────────────────────────────────
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=JobStatus.PENDING
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    final_pdf_storage_key: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=_now_utc, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    job: Mapped["AnalysisJob"] = relationship(
        "AnalysisJob", back_populates="cover_letter"
    )

    def __repr__(self) -> str:
        return f"<CoverLetter id={self.id} job_id={self.job_id}>"
