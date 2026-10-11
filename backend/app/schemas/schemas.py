"""
Pydantic schemas for request/response validation.

Kept separate from ORM models to avoid tight coupling:
  - ORM models  → database representation
  - Schemas     → API input/output contracts
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator


# ── Shared Base ───────────────────────────────────────────────────────────────
class TimestampMixin(BaseModel):
    created_at: datetime
    updated_at: Optional[datetime] = None


# ── Auth ──────────────────────────────────────────────────────────────────────
class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=255)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Analysis Job ──────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """
    Submitted via multipart/form-data. The `resume` file is handled
    separately by FastAPI's UploadFile; this schema represents the
    text fields in the same form submission.
    """

    url: HttpUrl = Field(..., description="Public URL of the job description page")


class ParsedResumeSchema(BaseModel):
    """Structured representation of an extracted resume."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    raw_text: Optional[str] = None


class ParsedJDSchema(BaseModel):
    """Structured representation of an extracted job description."""

    job_title: Optional[str] = None
    company: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


class AnalysisJobCreateResponse(BaseModel):
    """Returned immediately after the user submits an analysis request."""

    job_id: uuid.UUID
    status: str
    message: str = "Analysis job queued successfully."


class ScoringBreakdown(BaseModel):
    hard_skill_score: float = Field(..., ge=0, le=100)
    semantic_score: float = Field(..., ge=0, le=100)
    education_score: float = Field(..., ge=0, le=100)
    total_ats_score: float = Field(..., ge=0, le=100)
    missing_skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)


class AnalysisJobStatusResponse(BaseModel):
    """Full job status including results when complete."""

    job_id: uuid.UUID
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    # Populated when status == SUCCESS
    ats_score: Optional[float] = None
    scoring_breakdown: Optional[ScoringBreakdown] = None
    suggestions: Optional[List[str]] = None
    parsed_resume: Optional[ParsedResumeSchema] = None
    parsed_jd: Optional[ParsedJDSchema] = None

    # Populated when status == FAILURE
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


# ── Cover Letter ──────────────────────────────────────────────────────────────
class CoverLetterCreateResponse(BaseModel):
    cover_letter_id: uuid.UUID
    job_id: uuid.UUID
    status: str
    message: str = "Cover letter generation queued."


class CoverLetterStatusResponse(BaseModel):
    cover_letter_id: uuid.UUID
    job_id: uuid.UUID
    status: str
    generated_text: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── NLP Internal Schemas (used between modules, not exposed in API) ────────────
class GapAnalysis(BaseModel):
    """Gap analysis result between resume and JD."""

    missing_skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    weakly_matched_responsibilities: List[str] = Field(default_factory=list)
    strong_matched_responsibilities: List[str] = Field(default_factory=list)


# ── Common Responses ──────────────────────────────────────────────────────────
class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
