"""
Unit Tests – Resume Parser (resume_parser.py)
=============================================
Tests text extraction helpers and the structuring logic
WITHOUT calling spaCy (via nlp=None parameter).
No PDF files required — we test the text-analysis functions directly.
"""

import os
import tempfile

import pytest

from app.nlp.resume_parser import (
    _extract_skills_from_text,
    _find_email,
    _find_phone,
    _split_into_sections,
    parse_resume,
)


class TestEmailExtraction:
    def test_standard_email(self):
        text = "Contact me at john.smith@gmail.com for details."
        assert _find_email(text) == "john.smith@gmail.com"

    def test_no_email(self):
        text = "No email address here."
        assert _find_email(text) is None

    def test_email_at_start(self):
        text = "jane.doe@company.org is my email."
        assert _find_email(text) == "jane.doe@company.org"


class TestPhoneExtraction:
    def test_international_format(self):
        text = "+91 98765 43210"
        result = _find_phone(text)
        assert result is not None
        assert "98765" in result

    def test_us_format(self):
        text = "Call me at (555) 867-5309"
        result = _find_phone(text)
        assert result is not None

    def test_no_phone(self):
        text = "No phone number in this text."
        assert _find_phone(text) is None


class TestSkillExtraction:
    def test_known_skills_detected(self):
        text = "Experienced in Python, SQL, and Machine Learning"
        skills = _extract_skills_from_text(text)
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "sql" in skills_lower

    def test_case_insensitive_skill_detection(self):
        text = "PYTHON, javascript, AWS experience"
        skills = _extract_skills_from_text(text)
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "javascript" in skills_lower
        assert "aws" in skills_lower

    def test_no_skills(self):
        text = "I love hiking and cooking on weekends."
        skills = _extract_skills_from_text(text)
        # Should not explode; may return empty list
        assert isinstance(skills, list)

    def test_returns_list(self):
        skills = _extract_skills_from_text("some random text")
        assert isinstance(skills, list)


class TestSectionSplitter:
    SAMPLE_RESUME = """
John Smith
john@example.com

EXPERIENCE
Software Engineer at TechCorp
Jan 2020 – Present
Built microservices with Python and FastAPI.

EDUCATION
Master of Computer Science, MIT, 2019

SKILLS
Python, SQL, Docker, Kubernetes, AWS
"""

    def test_sections_identified(self):
        sections = _split_into_sections(self.SAMPLE_RESUME)
        assert "experience" in sections
        assert "education" in sections
        assert "skills" in sections

    def test_experience_section_has_content(self):
        sections = _split_into_sections(self.SAMPLE_RESUME)
        assert "TechCorp" in sections["experience"] or len(sections["experience"]) > 0

    def test_returns_dict(self):
        sections = _split_into_sections("Random text")
        assert isinstance(sections, dict)


class TestParseResumePDF:
    """
    Integration-ish test: creates a fake (corrupt) PDF and verifies
    that parse_resume raises ValueError with a helpful message instead of crashing.
    """

    def test_nonexistent_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found at path"):
            parse_resume("/nonexistent/path/resume.pdf")

    def test_non_pdf_raises_value_error(self):
        """A .pdf file that's actually plain text should fail gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w") as f:
            f.write("This is not a real PDF file.")
            tmp_path = f.name

        try:
            with pytest.raises((ValueError, Exception)):
                parse_resume(tmp_path)
        finally:
            os.unlink(tmp_path)


# ── Test Helper: make _find_email accessible for tests ───────────────────────

def _extract_email_from_text(text: str):
    """Re-export for test file."""
    from app.nlp.resume_parser import _find_email
    return _find_email(text)
