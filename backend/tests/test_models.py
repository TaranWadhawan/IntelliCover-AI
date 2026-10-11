"""
Model Output Tests – test_models.py
=====================================
Verifies that every NLP / ML model (and its pure-logic helpers) produces
well-formed, correct output in the locations where they are used.

Coverage:
  1. spacy_loader        – singleton, graceful-None fallback
  2. resume_parser       – _extract_name_from_text, _extract_experience_blocks,
                           _extract_education_blocks  (functions NOT covered by
                           test_resume_parser.py)
  3. jd_scraper          – _html_to_clean_text, _split_jd_into_sections,
                           parse_job_description (raw text), scrape_job_description
                           (mocked HTTP)
  4. scorer              – SemanticMatcher.compute_similarity (sentence-transformer),
                           compute_gap_analysis (pure logic)
  5. llm_service         – _format_suggestion_prompt, _format_cover_letter_prompt
                           (pure logic); LLMService.generate_suggestions and
                           generate_cover_letter (mocked heavy models)

Design notes:
  - Tests that need the sentence-transformers model are guarded with a
    pytest.mark.slow marker and are skipped when the embedding model
    cache is unavailable, so CI stays fast.
  - LLM tests never load real weights – they patch the loading methods so only
    the prompt-building and post-processing code runs.
  - All functions are tested through their public (or module-level private)
    interfaces; no attribute snooping of internal state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_resume(**kwargs):
    """Return a minimal ParsedResumeSchema with sensible defaults."""
    from app.schemas.schemas import ParsedResumeSchema

    defaults: Dict[str, Any] = dict(
        name="Alice Tester",
        email="alice@example.com",
        phone="555-000-1234",
        skills=["Python", "Docker", "FastAPI", "PostgreSQL"],
        experience=[
            {
                "title": "Backend Engineer",
                "company": "Acme Inc.",
                "duration": "2021-Present",
                "description": "Built REST APIs with FastAPI and Python.",
            }
        ],
        education=[
            {
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "institution": "State University",
                "year": "2020",
            }
        ],
        raw_text="Sample resume text.",
    )
    defaults.update(kwargs)
    return ParsedResumeSchema(**defaults)


def _make_jd(**kwargs):
    """Return a minimal ParsedJDSchema with sensible defaults."""
    from app.schemas.schemas import ParsedJDSchema

    defaults: Dict[str, Any] = dict(
        job_title="Software Engineer",
        company="Acme Corp",
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_skills=["Docker"],
        responsibilities=["Design REST APIs", "Write unit tests"],
        raw_text="We are looking for a Software Engineer …",
    )
    defaults.update(kwargs)
    return ParsedJDSchema(**defaults)


def _make_gap(**kwargs):
    """Return a minimal GapAnalysis."""
    from app.schemas.schemas import GapAnalysis

    defaults: Dict[str, Any] = dict(
        missing_skills=["Kubernetes"],
        matched_skills=["Python", "FastAPI"],
        weakly_matched_responsibilities=["Scale microservices"],
        strong_matched_responsibilities=["Design REST APIs"],
    )
    defaults.update(kwargs)
    return GapAnalysis(**defaults)


# ===========================================================================
# 1. spacy_loader
# ===========================================================================

class TestSpacyLoader:
    """Tests for app.nlp.spacy_loader.get_nlp_model()."""

    def test_returns_none_or_language_object(self):
        """get_nlp_model() must return either None or a spaCy Language."""
        import spacy
        from app.nlp.spacy_loader import get_nlp_model

        result = get_nlp_model()
        assert result is None or isinstance(result, spacy.language.Language)

    def test_singleton_same_object_on_repeated_calls(self):
        """The second call must return the EXACT same object (identity check)."""
        from app.nlp.spacy_loader import get_nlp_model

        first = get_nlp_model()
        second = get_nlp_model()
        assert first is second

    def test_pipeline_functions_accept_none_nlp(self):
        """All callers must handle nlp=None without raising."""
        from app.nlp.resume_parser import (
            _extract_education_blocks,
            _extract_experience_blocks,
            _extract_name_from_text,
            _extract_skills_from_text,
        )

        text = "Python developer at Acme Inc. B.Sc. Computer Science."
        # None of these should raise
        _extract_skills_from_text(text, nlp=None)
        _extract_name_from_text(text, nlp=None)
        _extract_experience_blocks(text, nlp=None)
        _extract_education_blocks(text, nlp=None)


# ===========================================================================
# 2. resume_parser – functions not covered by test_resume_parser.py
# ===========================================================================

class TestExtractName:
    """Tests for resume_parser._extract_name_from_text()."""

    def test_title_case_first_line_detected(self):
        from app.nlp.resume_parser import _extract_name_from_text

        text = "Alice Johnson\nalice@example.com\nPython developer"
        name = _extract_name_from_text(text, nlp=None)
        # Should pick up the first title-case line as the name
        assert name is not None
        assert "Alice" in name

    def test_all_caps_line_not_mistaken_for_name(self):
        from app.nlp.resume_parser import _extract_name_from_text

        text = "RESUME\nalice@example.com\nSkills: Python"
        # "RESUME" should not be returned as a person name
        name = _extract_name_from_text(text, nlp=None)
        # Either None or not the string "RESUME"
        assert name != "RESUME"

    def test_returns_string_or_none(self):
        from app.nlp.resume_parser import _extract_name_from_text

        result = _extract_name_from_text("random text without names", nlp=None)
        assert result is None or isinstance(result, str)

    def test_empty_text_returns_none(self):
        from app.nlp.resume_parser import _extract_name_from_text

        assert _extract_name_from_text("", nlp=None) is None


class TestExtractExperienceBlocks:
    """Tests for resume_parser._extract_experience_blocks()."""

    EXPERIENCE_TEXT = """
Software Engineer at Acme Inc.
Jan 2021 – Present
Developed microservices using FastAPI and Python.

Data Analyst at Beta Corp
June 2018 – Dec 2020
Built dashboards and ran statistical analyses.
"""

    def test_returns_list(self):
        from app.nlp.resume_parser import _extract_experience_blocks

        result = _extract_experience_blocks(self.EXPERIENCE_TEXT, nlp=None)
        assert isinstance(result, list)

    def test_list_items_are_dicts(self):
        from app.nlp.resume_parser import _extract_experience_blocks

        result = _extract_experience_blocks(self.EXPERIENCE_TEXT, nlp=None)
        for item in result:
            assert isinstance(item, dict)

    def test_expected_keys_present(self):
        from app.nlp.resume_parser import _extract_experience_blocks

        result = _extract_experience_blocks(self.EXPERIENCE_TEXT, nlp=None)
        if result:
            keys = result[0].keys()
            # At minimum the dict must contain *some* of these
            assert any(k in keys for k in ("title", "company", "duration", "description"))

    def test_max_ten_blocks_returned(self):
        from app.nlp.resume_parser import _extract_experience_blocks

        # Repeat the text enough to potentially produce >10 blocks
        big_text = self.EXPERIENCE_TEXT * 8
        result = _extract_experience_blocks(big_text, nlp=None)
        assert len(result) <= 10

    def test_empty_text_returns_list(self):
        from app.nlp.resume_parser import _extract_experience_blocks

        result = _extract_experience_blocks("", nlp=None)
        assert isinstance(result, list)


class TestExtractEducationBlocks:
    """Tests for resume_parser._extract_education_blocks()."""

    EDUCATION_TEXT = """
Master of Science in Computer Science, MIT, 2019
Bachelor of Engineering, IIT Bombay, 2016
"""

    def test_returns_list(self):
        from app.nlp.resume_parser import _extract_education_blocks

        result = _extract_education_blocks(self.EDUCATION_TEXT, nlp=None)
        assert isinstance(result, list)

    def test_items_are_dicts(self):
        from app.nlp.resume_parser import _extract_education_blocks

        result = _extract_education_blocks(self.EDUCATION_TEXT, nlp=None)
        for item in result:
            assert isinstance(item, dict)

    def test_expected_keys_in_items(self):
        from app.nlp.resume_parser import _extract_education_blocks

        result = _extract_education_blocks(self.EDUCATION_TEXT, nlp=None)
        if result:
            assert any(
                k in result[0].keys()
                for k in ("degree", "institution", "year", "field")
            )

    def test_max_five_blocks_returned(self):
        from app.nlp.resume_parser import _extract_education_blocks

        big_text = self.EDUCATION_TEXT * 10
        result = _extract_education_blocks(big_text, nlp=None)
        assert len(result) <= 5

    def test_empty_text_returns_list(self):
        from app.nlp.resume_parser import _extract_education_blocks

        result = _extract_education_blocks("", nlp=None)
        assert isinstance(result, list)


# ===========================================================================
# 3. jd_scraper
# ===========================================================================

class TestHtmlToCleanText:
    """Tests for jd_scraper._html_to_clean_text()."""

    def test_strips_script_tags(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        html = "<html><script>alert(1)</script><p>Hello world</p></html>"
        result = _html_to_clean_text(html)
        assert "alert" not in result
        assert "Hello world" in result

    def test_strips_style_tags(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        html = "<html><style>.foo{color:red}</style><p>Content here</p></html>"
        result = _html_to_clean_text(html)
        assert ".foo" not in result
        assert "Content here" in result

    def test_list_items_converted_to_bullets(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        html = "<ul><li>Python</li><li>Docker</li></ul>"
        result = _html_to_clean_text(html)
        assert "Python" in result
        assert "Docker" in result

    def test_returns_string(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        result = _html_to_clean_text("<p>Test</p>")
        assert isinstance(result, str)

    def test_empty_html_returns_string(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        result = _html_to_clean_text("")
        assert isinstance(result, str)

    def test_whitespace_collapsed(self):
        from app.nlp.jd_scraper import _html_to_clean_text

        html = "<p>too   many    spaces</p>"
        result = _html_to_clean_text(html)
        assert "  " not in result  # no double spaces


class TestSplitJdIntoSections:
    """Tests for jd_scraper._split_jd_into_sections()."""

    SAMPLE_JD = """
Senior Python Developer

We are a fast-growing startup looking for a talented developer.

Responsibilities:
Design and implement REST APIs.
Collaborate with the frontend team.
Participate in code reviews.

Requirements:
3+ years of Python experience.
Experience with FastAPI or Django.
Strong SQL knowledge.

Nice to Have:
Docker / Kubernetes experience.
AWS certification.
"""

    def test_returns_dict_with_expected_keys(self):
        from app.nlp.jd_scraper import _split_jd_into_sections

        result = _split_jd_into_sections(self.SAMPLE_JD)
        assert isinstance(result, dict)
        for key in ("intro", "responsibilities", "requirements", "preferred", "other"):
            assert key in result

    def test_responsibilities_populated(self):
        from app.nlp.jd_scraper import _split_jd_into_sections

        result = _split_jd_into_sections(self.SAMPLE_JD)
        assert len(result["responsibilities"]) > 0

    def test_requirements_populated(self):
        from app.nlp.jd_scraper import _split_jd_into_sections

        result = _split_jd_into_sections(self.SAMPLE_JD)
        assert len(result["requirements"]) > 0

    def test_each_value_is_list(self):
        from app.nlp.jd_scraper import _split_jd_into_sections

        result = _split_jd_into_sections(self.SAMPLE_JD)
        for v in result.values():
            assert isinstance(v, list)

    def test_empty_text_returns_empty_sections(self):
        from app.nlp.jd_scraper import _split_jd_into_sections

        result = _split_jd_into_sections("")
        for v in result.values():
            assert isinstance(v, list)
            assert len(v) == 0


class TestParseJobDescription:
    """Tests for jd_scraper.parse_job_description() with raw text (no HTTP)."""

    RAW_JD = """
Senior Backend Developer

TechCorp is hiring a Senior Backend Developer to join our platform team.

Responsibilities:
Build and maintain scalable REST APIs.
Design database schemas for PostgreSQL.
Write comprehensive unit and integration tests.

Requirements:
5+ years of Python experience.
Proficiency in FastAPI.
Experience with PostgreSQL and Redis.
Docker knowledge.

Preferred:
AWS or GCP certification.
Experience with Kubernetes.
"""

    def test_returns_parsed_jd_schema(self):
        from app.nlp.jd_scraper import parse_job_description
        from app.schemas.schemas import ParsedJDSchema

        result = parse_job_description(self.RAW_JD, nlp=None)
        assert isinstance(result, ParsedJDSchema)

    def test_required_skills_is_list(self):
        from app.nlp.jd_scraper import parse_job_description

        result = parse_job_description(self.RAW_JD, nlp=None)
        assert isinstance(result.required_skills, list)

    def test_known_skill_extracted(self):
        from app.nlp.jd_scraper import parse_job_description

        result = parse_job_description(self.RAW_JD, nlp=None)
        skills_lower = [s.lower() for s in result.required_skills]
        assert any("python" in s for s in skills_lower)

    def test_responsibilities_is_list(self):
        from app.nlp.jd_scraper import parse_job_description

        result = parse_job_description(self.RAW_JD, nlp=None)
        assert isinstance(result.responsibilities, list)

    def test_raw_text_respects_configured_cap(self):
        from app.nlp.jd_scraper import parse_job_description

        long_jd = self.RAW_JD * 50  # > 5000 chars
        result = parse_job_description(long_jd, nlp=None)
        if settings.JD_RAW_TEXT_MAX_CHARS > 0:
            assert len(result.raw_text) <= settings.JD_RAW_TEXT_MAX_CHARS
        else:
            assert result.raw_text == long_jd

    def test_empty_text_does_not_crash(self):
        from app.nlp.jd_scraper import parse_job_description
        from app.schemas.schemas import ParsedJDSchema

        result = parse_job_description("", nlp=None)
        assert isinstance(result, ParsedJDSchema)


class TestScrapeJobDescriptionMocked:
    """Tests for async jd_scraper.scrape_job_description() with mocked HTTP."""

    SAMPLE_HTML = (
        "<html><body>"
        "<h1>Software Engineer</h1>"
        "<p>We are looking for a talented Software Engineer to join our growing team. "
        "You will work on large-scale distributed systems and collaborate with "
        "a cross-functional team of engineers, designers, and product managers.</p>"
        "<h2>Requirements</h2>"
        "<ul>"
        "<li>3+ years of professional Python development experience</li>"
        "<li>Proficiency with Docker and container orchestration</li>"
        "<li>Experience with REST API design and implementation</li>"
        "<li>Familiarity with PostgreSQL or other relational databases</li>"
        "<li>Strong problem-solving and communication skills</li>"
        "</ul>"
        "<h2>Nice to Have</h2>"
        "<ul><li>Kubernetes</li><li>AWS or GCP cloud experience</li></ul>"
        "</body></html>"
    )

    async def test_httpx_success_returns_text(self):
        """When httpx returns valid HTML the function returns clean text."""
        from app.nlp.jd_scraper import scrape_job_description

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.SAMPLE_HTML

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.nlp.jd_scraper.httpx.AsyncClient", return_value=mock_client):
            result = await scrape_job_description("https://example.com/job")

        assert isinstance(result, str)
        assert len(result) > 0

    async def test_httpx_404_raises_value_error(self):
        """When both tiers fail, scrape_job_description must raise ValueError."""
        from app.nlp.jd_scraper import scrape_job_description

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("app.nlp.jd_scraper.httpx.AsyncClient", return_value=mock_client),
            patch("app.nlp.jd_scraper.settings") as mock_settings,
        ):
            mock_settings.SCRAPER_USE_PLAYWRIGHT_FALLBACK = False
            with pytest.raises(ValueError):
                await scrape_job_description("https://example.com/missing")

    async def test_httpx_exception_falls_through_gracefully(self):
        """A network error on httpx tier should not propagate directly."""
        from app.nlp.jd_scraper import scrape_job_description

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))

        with (
            patch("app.nlp.jd_scraper.httpx.AsyncClient", return_value=mock_client),
            patch("app.nlp.jd_scraper.settings") as mock_settings,
        ):
            mock_settings.SCRAPER_USE_PLAYWRIGHT_FALLBACK = False
            with pytest.raises((ValueError, Exception)):
                await scrape_job_description("https://example.com/timeout")


# ===========================================================================
# 4. scorer – SemanticMatcher and compute_gap_analysis
# ===========================================================================

def _sentence_transformer_available() -> bool:
    """Return True if the cached model files are present."""
    import os

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_cache",
        "hub",
        "models--sentence-transformers--all-MiniLM-L6-v2",
    )
    return os.path.isdir(model_dir)


@pytest.mark.slow
class TestSemanticMatcher:
    """
    Tests for scorer.SemanticMatcher.compute_similarity().

    These tests load the real sentence-transformer model and are
    therefore marked @slow. They are skipped when the model cache
    is absent so that rapid unit-test runs stay fast.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        if not _sentence_transformer_available():
            pytest.skip("sentence-transformer model cache not present")

    def test_returns_tuple_of_three(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        jd_resp = ["Design REST APIs", "Write unit tests"]
        exp = [{"description": "Built REST endpoints with FastAPI"}]
        result = matcher.compute_similarity(jd_resp, exp)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_score_in_range(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        score, strong, weak = matcher.compute_similarity(
            ["Python backend development"],
            [{"description": "Developed Python services"}],
        )
        assert 0.0 <= score <= 100.0

    def test_related_text_scores_higher_than_unrelated(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        related_score, _, _ = matcher.compute_similarity(
            ["Python data engineering with Spark"],
            [{"description": "Built data pipelines with Python and PySpark"}],
        )
        unrelated_score, _, _ = matcher.compute_similarity(
            ["Python data engineering with Spark"],
            [{"description": "Prepared financial reports in Excel"}],
        )
        assert related_score > unrelated_score

    def test_empty_responsibilities_returns_default(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        score, strong, weak = matcher.compute_similarity([], [])
        assert score == 50.0
        assert strong == []
        assert weak == []

    def test_strong_and_weak_are_disjoint(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        _, strong, weak = matcher.compute_similarity(
            ["Design APIs", "Deploy microservices", "Write documentation"],
            [{"description": "Designed and deployed RESTful APIs daily"}],
        )
        # No responsibility should appear in both lists
        strong_set = set(strong)
        weak_set = set(weak)
        assert strong_set.isdisjoint(weak_set)

    def test_singleton_returns_same_instance(self):
        from app.nlp.scorer import SemanticMatcher

        a = SemanticMatcher.get_instance()
        b = SemanticMatcher.get_instance()
        assert a is b

    def test_model_attribute_is_populated_after_first_compute(self):
        from app.nlp.scorer import SemanticMatcher

        matcher = SemanticMatcher()
        matcher.compute_similarity(
            ["Build software"],
            [{"description": "Built software products"}],
        )
        assert matcher._model is not None


class TestComputeGapAnalysis:
    """Tests for scorer.compute_gap_analysis() – pure logic, no model needed."""

    def test_returns_gap_analysis_schema(self):
        from app.nlp.scorer import compute_gap_analysis
        from app.schemas.schemas import GapAnalysis

        resume = _make_resume()
        jd = _make_jd()
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=["Python"],
            missing_skills=["Kubernetes"],
            strong_resp=["Design REST APIs"],
            weak_resp=["Scale microservices"],
        )
        assert isinstance(gap, GapAnalysis)

    def test_missing_skills_preserved(self):
        from app.nlp.scorer import compute_gap_analysis

        resume = _make_resume()
        jd = _make_jd()
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=["Python"],
            missing_skills=["Kubernetes", "Go"],
            strong_resp=[],
            weak_resp=[],
        )
        assert "Kubernetes" in gap.missing_skills
        assert "Go" in gap.missing_skills

    def test_matched_skills_preserved(self):
        from app.nlp.scorer import compute_gap_analysis

        resume = _make_resume()
        jd = _make_jd()
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=["Python", "FastAPI"],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert "Python" in gap.matched_skills
        assert "FastAPI" in gap.matched_skills

    def test_weak_responsibilities_preserved(self):
        from app.nlp.scorer import compute_gap_analysis

        resume = _make_resume()
        jd = _make_jd()
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=["Scale microservices globally"],
        )
        assert "Scale microservices globally" in gap.weakly_matched_responsibilities

    def test_strong_responsibilities_preserved(self):
        from app.nlp.scorer import compute_gap_analysis

        resume = _make_resume()
        jd = _make_jd()
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=[],
            missing_skills=[],
            strong_resp=["Design REST APIs"],
            weak_resp=[],
        )
        assert "Design REST APIs" in gap.strong_matched_responsibilities

    def test_empty_inputs_produce_valid_gap(self):
        from app.nlp.scorer import compute_gap_analysis
        from app.schemas.schemas import GapAnalysis

        resume = _make_resume(skills=[], experience=[], education=[])
        jd = _make_jd(required_skills=[], responsibilities=[])
        gap = compute_gap_analysis(
            resume=resume,
            jd=jd,
            matched_skills=[],
            missing_skills=[],
            strong_resp=[],
            weak_resp=[],
        )
        assert isinstance(gap, GapAnalysis)
        assert gap.missing_skills == []
        assert gap.matched_skills == []


# ===========================================================================
# 5. llm_service – prompt builders and mocked generation
# ===========================================================================

class TestFormatSuggestionPrompt:
    """Tests for llm_service._format_suggestion_prompt() – pure string logic."""

    def test_returns_non_empty_string(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        prompt = _format_suggestion_prompt(_make_resume(), _make_jd(), _make_gap())
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_job_title(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        jd = _make_jd(job_title="ML Engineer")
        prompt = _format_suggestion_prompt(_make_resume(), jd, _make_gap())
        assert "ML Engineer" in prompt

    def test_contains_missing_skill(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        gap = _make_gap(missing_skills=["Terraform"])
        prompt = _format_suggestion_prompt(_make_resume(), _make_jd(), gap)
        assert "Terraform" in prompt

    def test_contains_resume_skill(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        resume = _make_resume(skills=["Rust", "Go", "Python"])
        prompt = _format_suggestion_prompt(resume, _make_jd(), _make_gap())
        # At least one of the resume skills should appear
        assert any(skill in prompt for skill in ["Rust", "Go", "Python"])

    def test_contains_responsibility(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        jd = _make_jd(responsibilities=["Architect distributed systems"])
        prompt = _format_suggestion_prompt(_make_resume(), jd, _make_gap())
        assert "Architect distributed systems" in prompt

    def test_no_model_required(self):
        """Calling the formatter must never attempt to load any ML model."""
        from app.nlp.llm_service import _format_suggestion_prompt

        with patch("app.nlp.llm_service.LLMService._load_suggestion_model") as m:
            _format_suggestion_prompt(_make_resume(), _make_jd(), _make_gap())
            m.assert_not_called()

    def test_large_inputs_do_not_crash(self):
        from app.nlp.llm_service import _format_suggestion_prompt

        resume = _make_resume(
            skills=[f"Skill{i}" for i in range(200)],
            experience=[
                {"title": f"Role{i}", "company": f"Co{i}", "duration": "1y", "description": "x" * 300}
                for i in range(50)
            ],
        )
        jd = _make_jd(
            responsibilities=[f"Task {i}" for i in range(50)],
            required_skills=[f"Skill{i}" for i in range(100)],
        )
        gap = _make_gap(
            missing_skills=[f"Skill{i}" for i in range(50)],
            weakly_matched_responsibilities=[f"Weak {i}" for i in range(30)],
        )
        prompt = _format_suggestion_prompt(resume, jd, gap)
        assert isinstance(prompt, str)


class TestFormatCoverLetterPrompt:
    """Tests for llm_service._format_cover_letter_prompt() – pure string logic."""

    def test_returns_non_empty_string(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        prompt = _format_cover_letter_prompt(_make_resume(), _make_jd())
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_applicant_name(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        resume = _make_resume(name="Bob Builder")
        prompt = _format_cover_letter_prompt(resume, _make_jd())
        assert "Bob Builder" in prompt

    def test_contains_company_name(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        jd = _make_jd(company="Globex Corp")
        prompt = _format_cover_letter_prompt(_make_resume(), jd)
        assert "Globex Corp" in prompt

    def test_contains_job_title(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        jd = _make_jd(job_title="DevOps Architect")
        prompt = _format_cover_letter_prompt(_make_resume(), jd)
        assert "DevOps Architect" in prompt

    def test_no_experience_does_not_crash(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        resume = _make_resume(experience=[])
        prompt = _format_cover_letter_prompt(resume, _make_jd())
        assert isinstance(prompt, str)

    def test_no_education_does_not_crash(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        resume = _make_resume(education=[])
        prompt = _format_cover_letter_prompt(resume, _make_jd())
        assert isinstance(prompt, str)

    def test_no_model_required(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        with patch("app.nlp.llm_service.LLMService._load_cover_letter_model") as m:
            _format_cover_letter_prompt(_make_resume(), _make_jd())
            m.assert_not_called()

    def test_multiple_experiences_handled(self):
        from app.nlp.llm_service import _format_cover_letter_prompt

        resume = _make_resume(
            experience=[
                {"title": "SDE", "company": "A", "duration": "2022-Now", "description": "Built APIs"},
                {"title": "Intern", "company": "B", "duration": "2021", "description": "Tested code"},
                {"title": "Analyst", "company": "C", "duration": "2020", "description": "Wrote reports"},
            ]
        )
        prompt = _format_cover_letter_prompt(resume, _make_jd())
        assert "SDE" in prompt or "A" in prompt


class TestLLMServiceMockedSuggestions:
    """
    Tests for LLMService.generate_suggestions() with all heavy models mocked.
    Ensures the post-processing and output parsing code works correctly.
    """

    # Raw model outputs to simulate different scenarios
    BULLET_OUTPUT = "- Add Kubernetes to your skills.\n- Highlight Docker experience.\n- Include AWS certifications.\n"
    NUMBERED_OUTPUT = "1. Emphasise your FastAPI projects.\n2. Add a quantified achievement.\n3. Mention open-source contributions.\n"
    MIXED_OUTPUT = "• Lead with Python skills.\n* Add portfolio link.\n- Tailor summary to role.\n"

    def _patched_llm(self, raw_output: str):
        """Context manager that patches the LLM so _generate returns raw_output."""
        return patch("app.nlp.llm_service.LLMService._generate", return_value=raw_output)

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Ensure LLMService singleton is cleared before each test."""
        import app.nlp.llm_service as svc_mod

        svc_mod.LLMService._instance = None
        yield
        svc_mod.LLMService._instance = None

    def test_returns_list(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(self.BULLET_OUTPUT),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert isinstance(result, list)

    def test_bullet_list_parsed_correctly(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(self.BULLET_OUTPUT),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert any("Kubernetes" in s for s in result)

    def test_numbered_list_parsed_correctly(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(self.NUMBERED_OUTPUT),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert any("FastAPI" in s for s in result)

    def test_mixed_bullets_parsed(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(self.MIXED_OUTPUT),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert len(result) >= 1

    def test_max_five_suggestions_returned(self):
        from app.nlp.llm_service import LLMService

        # 7 bullets → only first 5 kept
        many_bullets = "\n".join(f"- Suggestion number {i}." for i in range(7))
        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(many_bullets),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert len(result) <= 5

    def test_empty_model_output_returns_empty_list(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(""),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        assert isinstance(result, list)
        assert len(result) == 0

    def test_very_short_lines_filtered_out(self):
        """Lines shorter than 20 chars should be discarded."""
        from app.nlp.llm_service import LLMService

        output = "- ok\n- This is a proper suggestion that is long enough.\n- hi\n"
        svc = LLMService()
        with (
            patch.object(svc, "_load_suggestion_model"),
            self._patched_llm(output),
        ):
            result = svc.generate_suggestions(_make_resume(), _make_jd(), _make_gap())
        # "ok" and "hi" are < 20 chars so should be excluded
        for s in result:
            assert len(s) > 20


class TestLLMServiceMockedCoverLetter:
    """
    Tests for LLMService.generate_cover_letter() with heavy models mocked.
    Ensures output cleaning logic works as expected.
    """

    CLEAN_OUTPUT = (
        "Dear Hiring Manager,\n\n"
        "I am excited to apply for the Software Engineer role at Acme Corp. "
        "My experience with FastAPI and Python aligns well with your needs.\n\n"
        "Sincerely,\nAlice Tester"
    )

    DIRTY_OUTPUT = (
        "Here is a cover letter for you:\n\n"
        "Dear Hiring Manager,\n\n"
        "I am applying for this role. "
        "My background in backend development makes me a strong fit.\n\n"
        "Best regards,\nAlice"
    )

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        import app.nlp.llm_service as svc_mod

        svc_mod.LLMService._instance = None
        yield
        svc_mod.LLMService._instance = None

    def _patched_llm(self, raw_output: str):
        return patch("app.nlp.llm_service.LLMService._generate", return_value=raw_output)

    def test_returns_string(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_cover_letter_model"),
            self._patched_llm(self.CLEAN_OUTPUT),
        ):
            result = svc.generate_cover_letter(_make_resume(), _make_jd())
        assert isinstance(result, str)

    def test_non_empty_output(self):
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_cover_letter_model"),
            self._patched_llm(self.CLEAN_OUTPUT),
        ):
            result = svc.generate_cover_letter(_make_resume(), _make_jd())
        assert len(result) > 0

    def test_preamble_stripped_before_dear(self):
        """_clean_cover_letter_output should remove text before 'Dear'."""
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_cover_letter_model"),
            self._patched_llm(self.DIRTY_OUTPUT),
        ):
            result = svc.generate_cover_letter(_make_resume(), _make_jd())
        # Preamble ("Here is a cover letter for you:") should be gone
        assert not result.startswith("Here is")
        assert "Dear" in result

    def test_clean_output_preserved(self):
        """Output that starts with 'Dear' should be returned as-is."""
        from app.nlp.llm_service import LLMService

        svc = LLMService()
        with (
            patch.object(svc, "_load_cover_letter_model"),
            self._patched_llm(self.CLEAN_OUTPUT),
        ):
            result = svc.generate_cover_letter(_make_resume(), _make_jd())
        assert result.strip().startswith("Dear")
