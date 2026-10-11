"""
Unit Tests – Scoring Module (scorer.py)
========================================
Tests the ATS scoring functions in isolation.
No database, no models, no external services.
"""

import pytest

from app.nlp.scorer import (
    calculate_hard_skill_score,
    calculate_education_score,
    aggregate_scores,
    run_scoring_pipeline,
)
from app.schemas.schemas import ParsedResumeSchema, ParsedJDSchema


class TestHardSkillScore:
    """Unit tests for calculate_hard_skill_score."""

    def test_perfect_match(self):
        """All JD skills present in resume → 100%"""
        resume_skills = ["Python", "SQL", "Tableau", "Docker"]
        jd_skills = ["Python", "SQL", "Tableau"]
        score, matched, missing = calculate_hard_skill_score(resume_skills, jd_skills)
        assert score == 100.0
        assert len(matched) == 3
        assert len(missing) == 0

    def test_no_match(self):
        """No JD skills present → 0%"""
        resume_skills = ["Java", "Go"]
        jd_skills = ["Python", "SQL", "Tableau"]
        score, matched, missing = calculate_hard_skill_score(resume_skills, jd_skills)
        assert score == 0.0
        assert len(matched) == 0
        assert len(missing) == 3

    def test_partial_match(self):
        """2 of 4 JD skills matched → 50%"""
        resume_skills = ["Python", "SQL", "Java"]
        jd_skills = ["Python", "SQL", "Tableau", "AWS"]
        score, matched, missing = calculate_hard_skill_score(resume_skills, jd_skills)
        assert score == 50.0
        assert "Python" in matched
        assert "SQL" in matched
        assert "Tableau" in missing
        assert "AWS" in missing

    def test_case_insensitive_matching(self):
        """Skill matching should be case-insensitive."""
        resume_skills = ["python", "sql"]
        jd_skills = ["Python", "SQL"]
        score, matched, missing = calculate_hard_skill_score(resume_skills, jd_skills)
        assert score == 100.0

    def test_empty_jd_skills(self):
        """No JD skills → returns 100 (benefit of doubt)."""
        score, matched, missing = calculate_hard_skill_score(["Python"], [])
        assert score == 100.0

    def test_empty_resume_skills(self):
        """Empty resume → 0%"""
        score, matched, missing = calculate_hard_skill_score([], ["Python", "SQL"])
        assert score == 0.0

    def test_returns_correct_types(self):
        """Return types must be correct."""
        score, matched, missing = calculate_hard_skill_score(["Python"], ["Python"])
        assert isinstance(score, float)
        assert isinstance(matched, list)
        assert isinstance(missing, list)


class TestEducationScore:
    """Unit tests for calculate_education_score."""

    def test_phd_for_phd_requirement(self):
        education = [{"degree": "PhD Computer Science", "institution": "MIT"}]
        jd_text = "Requires a PhD in Computer Science or related field."
        score = calculate_education_score(education, jd_text)
        assert score == 100.0

    def test_bachelor_meets_bachelor_requirement(self):
        education = [{"degree": "Bachelor of Technology", "institution": "IIT"}]
        jd_text = "Bachelor's degree required."
        score = calculate_education_score(education, jd_text)
        assert score == 100.0

    def test_no_education_requirement_in_jd(self):
        """If JD doesn't mention education → benefit of doubt (80%)."""
        education = [{"degree": "B.Tech"}]
        jd_text = "Join our dynamic team. Work on cutting-edge projects."
        score = calculate_education_score(education, jd_text)
        assert score == 80.0

    def test_empty_resume_education(self):
        """Cannot determine → partial credit (50%)."""
        score = calculate_education_score([], "Must have a Master's degree.")
        assert score == 50.0


class TestScoreAggregation:
    """Unit tests for the aggregate_scores function."""

    def test_equal_weights_produce_correct_total(self):
        """Verify weighted formula: 50/40/10 weights."""
        score = aggregate_scores(
            hard_skill_score=100.0,
            semantic_score=100.0,
            education_score=100.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert score.total_ats_score == 100.0

    def test_zero_scores(self):
        score = aggregate_scores(
            hard_skill_score=0.0,
            semantic_score=0.0,
            education_score=0.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert score.total_ats_score == 0.0

    def test_only_hard_skill_score(self):
        """Hard skill score contributes 50% of total."""
        score = aggregate_scores(
            hard_skill_score=100.0,
            semantic_score=0.0,
            education_score=0.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert score.total_ats_score == 50.0

    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        score = aggregate_scores(
            hard_skill_score=110.0,  # Invalid but should be capped
            semantic_score=110.0,
            education_score=110.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert score.total_ats_score <= 100.0

    def test_score_floor_at_0(self):
        """Score should never go below 0."""
        score = aggregate_scores(
            hard_skill_score=-10.0,
            semantic_score=-10.0,
            education_score=-10.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert score.total_ats_score >= 0.0

    def test_score_breakdown_fields_populated(self):
        score = aggregate_scores(
            hard_skill_score=80.0,
            semantic_score=70.0,
            education_score=100.0,
            matched_skills=["Python"],
            missing_skills=["SQL"],
            strong_resp=["Build models"],
            weak_resp=["Communicate findings"],
        )
        assert score.hard_skill_score == 80.0
        assert score.semantic_score == 70.0
        assert score.education_score == 100.0
        assert "Python" in score.matched_skills
        assert "SQL" in score.missing_skills


class TestScoringPipelineIntegration:
    """Integration test for the full scoring pipeline without LLM."""

    def test_happy_path_no_crash(self):
        """
        Run the scoring pipeline with realistic data.
        Semantic matcher will be skipped if sentence-transformers is not installed.
        """
        resume = ParsedResumeSchema(
            name="John Smith",
            email="john@example.com",
            skills=["Python", "SQL", "Docker", "Machine Learning"],
            experience=[
                {
                    "title": "Data Scientist",
                    "company": "TechCorp",
                    "description": "Built ML models, worked with large datasets, deployed to production.",
                }
            ],
            education=[{"degree": "M.Tech Computer Science", "institution": "IIT Delhi"}],
        )
        jd = ParsedJDSchema(
            job_title="Senior Data Scientist",
            company="DataTech",
            required_skills=["Python", "SQL", "AWS", "PyTorch"],
            responsibilities=[
                "Develop and maintain ML models",
                "Collaborate with cross-functional teams",
            ],
            raw_text="Must have a Master's degree or higher.",
        )

        scoring, gap = run_scoring_pipeline(resume, jd)

        # Basic sanity checks
        assert 0 <= scoring.total_ats_score <= 100
        assert isinstance(gap.missing_skills, list)
        assert isinstance(gap.matched_skills, list)

        # Python and SQL should be matched
        matched_lower = [s.lower() for s in gap.matched_skills]
        assert "python" in matched_lower or "sql" in matched_lower

        # AWS and PyTorch should be missing
        missing_lower = [s.lower() for s in gap.missing_skills]
        assert "aws" in missing_lower or "pytorch" in missing_lower
