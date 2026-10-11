"""
Unit Tests – Schemas (schemas.py)
=====================================
Tests Pydantic schema validation for all major request/response models.
No database or external services required.
"""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.schemas import (
    AnalysisJobCreateResponse,
    GapAnalysis,
    ParsedJDSchema,
    ParsedResumeSchema,
    ScoringBreakdown,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)


# ── UserRegisterRequest ───────────────────────────────────────────────────────


class TestUserRegisterRequest:
    def test_valid_payload_passes(self):
        u = UserRegisterRequest(email="user@example.com", password="strongpassword")
        assert u.email == "user@example.com"

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(email="not_an_email", password="password123")

    def test_password_too_short_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(email="user@example.com", password="short")

    def test_password_too_long_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(email="user@example.com", password="x" * 129)

    def test_optional_full_name_defaults_to_none(self):
        u = UserRegisterRequest(email="user@example.com", password="validpass123")
        assert u.full_name is None

    def test_full_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(
                email="user@example.com",
                password="validpass123",
                full_name="A" * 256,
            )

    def test_full_name_exactly_at_limit_passes(self):
        u = UserRegisterRequest(
            email="user@example.com",
            password="validpass123",
            full_name="A" * 255,
        )
        assert len(u.full_name) == 255


# ── UserLoginRequest ──────────────────────────────────────────────────────────


class TestUserLoginRequest:
    def test_valid_login_payload(self):
        u = UserLoginRequest(email="user@example.com", password="anypassword")
        assert u.email == "user@example.com"

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            UserLoginRequest(email="bad-email", password="anypassword")


# ── TokenResponse ────────────────────────────────────────────────────────────


class TestTokenResponse:
    def test_valid_token(self):
        t = TokenResponse(access_token="sometoken", expires_in=3600)
        assert t.token_type == "bearer"

    def test_custom_token_type(self):
        t = TokenResponse(access_token="tok", token_type="bearer", expires_in=100)
        assert t.token_type == "bearer"


# ── ParsedResumeSchema ────────────────────────────────────────────────────────


class TestParsedResumeSchema:
    def test_defaults_are_empty_collections(self):
        r = ParsedResumeSchema()
        assert r.skills == []
        assert r.experience == []
        assert r.education == []
        assert r.name is None
        assert r.email is None

    def test_full_resume_parses_correctly(self):
        r = ParsedResumeSchema(
            name="Jane Doe",
            email="jane@example.com",
            phone="+1-800-555-0101",
            skills=["Python", "SQL"],
            experience=[{"title": "SWE", "company": "ACME"}],
            education=[{"degree": "B.Tech"}],
        )
        assert r.name == "Jane Doe"
        assert "Python" in r.skills
        assert len(r.experience) == 1


# ── ParsedJDSchema ────────────────────────────────────────────────────────────


class TestParsedJDSchema:
    def test_defaults_are_empty_collections(self):
        jd = ParsedJDSchema()
        assert jd.required_skills == []
        assert jd.preferred_skills == []
        assert jd.responsibilities == []

    def test_full_jd_parses_correctly(self):
        jd = ParsedJDSchema(
            job_title="Data Engineer",
            company="TechCorp",
            required_skills=["Spark", "Python"],
            responsibilities=["Build pipelines"],
        )
        assert jd.job_title == "Data Engineer"
        assert "Spark" in jd.required_skills


# ── ScoringBreakdown ──────────────────────────────────────────────────────────


class TestScoringBreakdown:
    def test_valid_scores_pass(self):
        sb = ScoringBreakdown(
            hard_skill_score=80.0,
            semantic_score=70.0,
            education_score=100.0,
            total_ats_score=79.0,
        )
        assert sb.total_ats_score == 79.0

    def test_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            ScoringBreakdown(
                hard_skill_score=101.0,
                semantic_score=50.0,
                education_score=50.0,
                total_ats_score=75.0,
            )

    def test_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            ScoringBreakdown(
                hard_skill_score=-1.0,
                semantic_score=50.0,
                education_score=50.0,
                total_ats_score=25.0,
            )

    def test_boundary_values_pass(self):
        sb = ScoringBreakdown(
            hard_skill_score=0.0,
            semantic_score=100.0,
            education_score=0.0,
            total_ats_score=40.0,
        )
        assert sb.hard_skill_score == 0.0
        assert sb.semantic_score == 100.0

    def test_matched_and_missing_skills_default_empty(self):
        sb = ScoringBreakdown(
            hard_skill_score=50.0,
            semantic_score=50.0,
            education_score=50.0,
            total_ats_score=50.0,
        )
        assert sb.matched_skills == []
        assert sb.missing_skills == []


# ── GapAnalysis ───────────────────────────────────────────────────────────────


class TestGapAnalysis:
    def test_all_fields_default_to_empty_list(self):
        ga = GapAnalysis()
        assert ga.missing_skills == []
        assert ga.matched_skills == []
        assert ga.weakly_matched_responsibilities == []
        assert ga.strong_matched_responsibilities == []

    def test_populated_gap_analysis(self):
        ga = GapAnalysis(
            missing_skills=["Docker", "Kubernetes"],
            matched_skills=["Python"],
            weakly_matched_responsibilities=["Communicate results"],
            strong_matched_responsibilities=["Build ML models"],
        )
        assert len(ga.missing_skills) == 2
        assert len(ga.matched_skills) == 1


# ── AnalysisJobCreateResponse ─────────────────────────────────────────────────


class TestAnalysisJobCreateResponse:
    def test_valid_response(self):
        job_id = uuid.uuid4()
        resp = AnalysisJobCreateResponse(job_id=job_id, status="PENDING")
        assert resp.job_id == job_id
        assert resp.message == "Analysis job queued successfully."

    def test_custom_message(self):
        resp = AnalysisJobCreateResponse(
            job_id=uuid.uuid4(),
            status="PENDING",
            message="Custom message here.",
        )
        assert resp.message == "Custom message here."
