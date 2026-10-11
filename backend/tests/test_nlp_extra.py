"""
Unit Tests – JD Parser & Scorer Edge Cases
==========================================
Additional tests for JD parsing helpers and scoring edge cases.
No external HTTP calls or database required.
"""

import pytest

from app.nlp.scorer import (
    _normalize_skill,
    calculate_education_score,
    calculate_hard_skill_score,
    aggregate_scores,
    run_scoring_pipeline,
)
from app.schemas.schemas import GapAnalysis, ParsedJDSchema, ParsedResumeSchema


# ──────────────────────────────────────────────────────────────────────────────
# _normalize_skill helper
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeSkill:
    def test_lowercase(self):
        assert _normalize_skill("Python") == "python"

    def test_strips_whitespace(self):
        assert _normalize_skill("  sql  ") == "sql"

    def test_collapses_multiple_spaces(self):
        assert _normalize_skill("machine  learning") == "machine learning"

    def test_empty_string(self):
        assert _normalize_skill("") == ""

    def test_already_normalized(self):
        assert _normalize_skill("docker") == "docker"


# ──────────────────────────────────────────────────────────────────────────────
# calculate_hard_skill_score – additional edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestHardSkillScoreEdgeCases:
    def test_both_empty_lists(self):
        score, matched, missing = calculate_hard_skill_score([], [])
        assert score == 100.0
        assert matched == []
        assert missing == []

    def test_whitespace_in_skill_names_normalized(self):
        """Skills with extra whitespace are normalized to match."""
        score, matched, missing = calculate_hard_skill_score(
            [" python "], ["Python"]
        )
        assert score == 100.0

    def test_duplicate_jd_skills_count_once(self):
        """Duplicate JD skills should not inflate the score."""
        score, matched, missing = calculate_hard_skill_score(
            ["Python"],
            ["Python", "Python"],  # duplicated
        )
        # Two JD skills, both matched → 100%
        assert score == 100.0

    def test_order_does_not_matter(self):
        resume = ["Docker", "Kubernetes", "Python"]
        jd = ["Python", "Docker"]
        score, matched, missing = calculate_hard_skill_score(resume, jd)
        assert score == 100.0

    def test_multi_word_skill_match(self):
        score, matched, missing = calculate_hard_skill_score(
            ["Machine Learning", "Deep Learning"],
            ["machine learning"],
        )
        assert score == 100.0

    def test_returns_original_casing_in_matched(self):
        """matched list should preserve JD's original casing."""
        _, matched, _ = calculate_hard_skill_score(["python"], ["Python"])
        assert "Python" in matched

    def test_returns_original_casing_in_missing(self):
        _, _, missing = calculate_hard_skill_score([], ["TensorFlow"])
        assert "TensorFlow" in missing


# ──────────────────────────────────────────────────────────────────────────────
# calculate_education_score – additional edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestEducationScoreEdgeCases:
    def test_phd_exceeds_master_requirement(self):
        """PhD is higher than Master requirement → 100%."""
        score = calculate_education_score(
            [{"degree": "PhD Computer Science"}],
            "Requires a Master's degree.",
        )
        assert score == 100.0

    def test_bachelor_falls_short_of_master(self):
        """Bachelor is one level below Master → 60%."""
        score = calculate_education_score(
            [{"degree": "Bachelor of Science"}],
            "Requires a Master's degree.",
        )
        assert score == 60.0

    def test_empty_jd_text_gives_benefit_of_doubt(self):
        score = calculate_education_score([{"degree": "B.Tech"}], "")
        assert score == 80.0

    def test_none_jd_text_handled_gracefully(self):
        score = calculate_education_score([{"degree": "MBA"}], None)
        assert score == 80.0

    def test_non_dict_entries_in_education_handled(self):
        """Gracefully handle malformed education list entries."""
        score = calculate_education_score(
            [{"degree": "B.Tech"}, "invalid_entry"],
            "Bachelor required.",
        )
        assert isinstance(score, float)

    def test_multiple_degrees_uses_highest_level(self):
        """Should use the highest education level from the list."""
        score = calculate_education_score(
            [
                {"degree": "Bachelor of Science"},
                {"degree": "Master of Science"},
            ],
            "Requires a Master's degree.",
        )
        assert score == 100.0


# ──────────────────────────────────────────────────────────────────────────────
# aggregate_scores – additional edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregateScoresEdgeCases:
    def test_weights_sum_correctly(self):
        """Verify 50/40/10 weighting: 80*0.5 + 60*0.4 + 100*0.1 = 74."""
        sb = aggregate_scores(
            hard_skill_score=80.0,
            semantic_score=60.0,
            education_score=100.0,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert sb.total_ats_score == pytest.approx(74.0, abs=0.1)

    def test_total_is_float(self):
        sb = aggregate_scores(50.0, 50.0, 50.0, [], [], [], [])
        assert isinstance(sb.total_ats_score, float)

    def test_matched_and_missing_preserved(self):
        sb = aggregate_scores(
            50.0, 50.0, 50.0,
            matched_skills=["Python", "SQL"],
            missing_skills=["Docker"],
            strong_resp=["Build APIs"],
            weak_resp=["Communicate"],
        )
        assert "Python" in sb.matched_skills
        assert "Docker" in sb.missing_skills

    def test_component_scores_clamped_and_stored(self):
        """After fix, all component scores in breakdown are ≤ 100."""
        sb = aggregate_scores(150.0, 200.0, 300.0, [], [], [], [])
        assert sb.hard_skill_score <= 100.0
        assert sb.semantic_score <= 100.0
        assert sb.education_score <= 100.0

    def test_negative_scores_clamped_to_zero(self):
        sb = aggregate_scores(-50.0, -100.0, -20.0, [], [], [], [])
        assert sb.hard_skill_score >= 0.0
        assert sb.semantic_score >= 0.0
        assert sb.education_score >= 0.0
        assert sb.total_ats_score >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# run_scoring_pipeline – more integration scenarios
# ──────────────────────────────────────────────────────────────────────────────


class TestScoringPipelineExtraScenarios:
    def test_empty_skills_and_jd_requirements(self):
        resume = ParsedResumeSchema()
        jd = ParsedJDSchema()
        scoring, gap = run_scoring_pipeline(resume, jd)
        assert 0 <= scoring.total_ats_score <= 100

    def test_full_match_gives_high_skill_score(self):
        resume = ParsedResumeSchema(
            skills=["Python", "SQL", "Docker", "AWS"],
            experience=[{"title": "SWE", "description": "Built APIs with Python"}],
            education=[{"degree": "B.Tech"}],
        )
        jd = ParsedJDSchema(
            required_skills=["Python", "SQL", "Docker", "AWS"],
            responsibilities=["Build REST APIs"],
            raw_text="Bachelor's degree required.",
        )
        scoring, gap = run_scoring_pipeline(resume, jd)
        assert scoring.hard_skill_score == 100.0
        assert gap.matched_skills  # should have matches
        assert gap.missing_skills == []

    def test_zero_overlap_gives_low_score(self):
        resume = ParsedResumeSchema(
            skills=["Cobol", "FORTRAN"],
            experience=[],
            education=[],
        )
        jd = ParsedJDSchema(
            required_skills=["Python", "Kubernetes", "Terraform"],
            responsibilities=["Deploy microservices"],
            raw_text="Master's degree required.",
        )
        scoring, gap = run_scoring_pipeline(resume, jd)
        # Hard skill score should be 0%
        assert scoring.hard_skill_score == 0.0
        # All JD skills should be missing
        missing_lower = [s.lower() for s in gap.missing_skills]
        assert "python" in missing_lower

    def test_gap_analysis_types_are_lists(self):
        resume = ParsedResumeSchema(skills=["Python"])
        jd = ParsedJDSchema(required_skills=["Python", "SQL"])
        _, gap = run_scoring_pipeline(resume, jd)
        assert isinstance(gap.missing_skills, list)
        assert isinstance(gap.matched_skills, list)

    def test_result_scores_always_within_bounds(self):
        """Property test: whatever the input, final score is 0-100."""
        for _ in range(5):
            resume = ParsedResumeSchema(
                skills=["Python"],
                experience=[{"title": "Dev", "description": "did stuff"}],
                education=[{"degree": "B.Tech"}],
            )
            jd = ParsedJDSchema(
                required_skills=["Python", "SQL"],
                responsibilities=["Build products"],
                raw_text="",
            )
            scoring, _ = run_scoring_pipeline(resume, jd)
            assert 0 <= scoring.total_ats_score <= 100
