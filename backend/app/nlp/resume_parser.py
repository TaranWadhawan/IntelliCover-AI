"""
Stage 1 – Resume Parsing Module
================================
Extracts raw text from PDF files and structures it using NER.

Pipeline:
  PDF file → PyMuPDF (raw text) → spaCy NER → structured JSON
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.schemas import ParsedResumeSchema

logger = get_logger(__name__)

# ── Common skill keywords (curated seed list; supplements NER) ────────────────
COMMON_SKILLS_SEED = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "kotlin", "swift", "r", "matlab", "julia", "php", "ruby", "sql",
    # Frameworks / Libraries
    "react", "vue", "angular", "next.js", "fastapi", "django", "flask",
    "spring", "express", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "spark", "kafka",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "github actions", "jenkins",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "snowflake",
    "bigquery",
    # Data / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "data engineering", "mlops", "llm", "transformers",
    "bert", "gpt", "langchain",
    # Soft / Methodology
    "agile", "scrum", "kanban", "leadership", "communication",
}


# ── PDF Text Extraction ───────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.

    Handles:
      - Multi-page PDFs (text merged with newlines).
      - Corrupted / password-protected PDFs (raises ValueError).

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        The concatenated text content of all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF is encrypted or yields no extractable text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume PDF not found at path: {pdf_path}")

    logger.info("extracting_pdf_text", path=str(path))

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    if doc.is_encrypted:
        raise ValueError(
            f"PDF '{pdf_path}' is password-protected and cannot be parsed."
        )

    pages_text: List[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()

    if not pages_text:
        raise ValueError(
            f"PDF '{pdf_path}' yielded no extractable text. "
            "It may be an image-only scan."
        )

    raw_text = "\n".join(pages_text)
    logger.info(
        "pdf_text_extracted",
        path=str(path),
        num_pages=len(pages_text),
        char_count=len(raw_text),
    )
    return raw_text


# ── Regex helpers ─────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

_SECTION_HEADERS = {
    "experience": [
        "experience", "work experience", "employment", "employment history",
        "professional experience", "work history", "career history",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "degrees",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tools", "expertise",
    ],
}


def _find_email(text: str) -> Optional[str]:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def _find_phone(text: str) -> Optional[str]:
    match = _PHONE_RE.search(text)
    return match.group(0).strip() if match else None


def _split_into_sections(text: str) -> Dict[str, str]:
    """
    Heuristically split resume text into named sections.
    Returns a dict like: {"experience": "...", "education": "...", "skills": "..."}
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {"header": [], "skills": [], "experience": [], "education": [], "other": []}
    current_section = "header"

    for line in lines:
        stripped = line.strip().lower()
        matched_section = None
        for section_name, keywords in _SECTION_HEADERS.items():
            if any(stripped == kw or stripped.startswith(kw) for kw in keywords):
                matched_section = section_name
                break

        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


def _extract_skills_from_text(text: str, nlp=None) -> List[str]:
    """
    Extract skills using two complementary approaches:
    1. Seed-list keyword matching (fast, high-precision for known skills)
    2. spaCy NER for additional entities (if nlp model provided)
    """
    found_skills: set = set()
    text_lower = text.lower()

    # Approach 1: Seed list matching
    for skill in COMMON_SKILLS_SEED:
        # Word-boundary matching to avoid "spark" matching "sparkle"
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill.title() if len(skill) > 3 else skill.upper())

    # Approach 2: spaCy NER (if model loaded)
    if nlp:
        doc = nlp(text[:10000])  # limit to 10k chars for performance
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "LANGUAGE", "SKILL"):
                candidate = ent.text.strip()
                if 2 < len(candidate) < 50:
                    found_skills.add(candidate)

    return sorted(found_skills)


def _extract_name_from_text(text: str, nlp=None) -> Optional[str]:
    """
    Try to extract full name from the top of the resume.
    Strategy: first non-empty, non-email, non-URL line that looks like a name.
    """
    # Try spaCy PERSON entity first
    if nlp:
        snippet = text[:500]
        doc = nlp(snippet)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                return ent.text.strip()

    # Fallback: first non-junk, title-case line at top
    for line in text.split("\n")[:10]:
        line = line.strip()
        if (
            line
            and not _EMAIL_RE.search(line)
            and not _URL_RE.search(line)
            and not _PHONE_RE.search(line)
            and len(line.split()) in range(2, 5)
            and line[0].isupper()
        ):
            return line

    return None


def _extract_experience_blocks(experience_text: str, nlp=None) -> List[Dict[str, Any]]:
    """
    Parse the experience section into a list of job entries.
    Each entry: { "title": ..., "company": ..., "duration": ..., "description": ... }
    """
    if not experience_text.strip():
        return []

    entries = []
    # Split on double newlines as entry separators
    blocks = [b.strip() for b in re.split(r"\n{2,}", experience_text) if b.strip()]

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        entry: Dict[str, Any] = {
            "title": None,
            "company": None,
            "duration": None,
            "description": block,
        }

        # Try NER to find ORG and JOB_TITLE
        if nlp and block:
            doc = nlp(block[:1000])
            for ent in doc.ents:
                if ent.label_ == "ORG" and entry["company"] is None:
                    entry["company"] = ent.text.strip()
                if ent.label_ in ("PERSON", "WORK_OF_ART") and entry["title"] is None:
                    pass  # Skip person entities for title

        # Fallback: assume first line is title or company
        if lines:
            entry["title"] = lines[0]
        if len(lines) > 1:
            entry["company"] = lines[1]

        # Look for date patterns
        date_pattern = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|"
            r"february|march|april|june|july|august|september|october|"
            r"november|december)[\w\s,–-]*\d{4}",
            block.lower(),
        )
        if date_pattern:
            entry["duration"] = date_pattern.group(0).strip()

        entries.append(entry)

    return entries[:10]  # Limit to 10 entries max


def _extract_education_blocks(education_text: str, nlp=None) -> List[Dict[str, Any]]:
    """Parse education section into structured entries."""
    if not education_text.strip():
        return []

    entries = []
    blocks = [b.strip() for b in re.split(r"\n{2,}", education_text) if b.strip()]

    degree_keywords = ["bachelor", "master", "phd", "b.e.", "b.tech", "m.tech",
                       "mba", "bsc", "msc", "b.sc", "m.sc", "b.s.", "m.s."]

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        entry: Dict[str, Any] = {
            "degree": None,
            "institution": None,
            "year": None,
            "field": None,
        }

        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in degree_keywords) and not entry["degree"]:
                entry["degree"] = line
            elif nlp:
                doc = nlp(line)
                for ent in doc.ents:
                    if ent.label_ == "ORG" and not entry["institution"]:
                        entry["institution"] = ent.text.strip()
                    if ent.label_ == "DATE" and not entry["year"]:
                        entry["year"] = ent.text.strip()

        if not entry["degree"] and lines:
            entry["degree"] = lines[0]

        entries.append(entry)

    return entries[:5]


# ── Main Public Function ──────────────────────────────────────────────────────

def parse_resume(pdf_path: str, nlp=None) -> ParsedResumeSchema:
    """
    Full resume parsing pipeline.

    Args:
        pdf_path: Path to resume PDF.
        nlp: Optional pre-loaded spaCy model for NER. If None, only
             regex/seed-based extraction is used.

    Returns:
        ParsedResumeSchema with all extracted structured data.
    """
    logger.info("resume_parsing_started", path=pdf_path)

    raw_text = extract_text_from_pdf(pdf_path)
    sections = _split_into_sections(raw_text)

    name = _extract_name_from_text(raw_text, nlp)
    email = _find_email(raw_text)
    phone = _find_phone(raw_text)

    skills_text = sections.get("skills", "") + "\n" + sections.get("other", "")
    skills = _extract_skills_from_text(skills_text or raw_text, nlp)

    experience = _extract_experience_blocks(sections.get("experience", ""), nlp)
    education = _extract_education_blocks(sections.get("education", ""), nlp)

    max_raw_chars = settings.RESUME_RAW_TEXT_MAX_CHARS
    raw_text_for_storage = (
        raw_text if max_raw_chars <= 0 else raw_text[:max_raw_chars]
    )

    result = ParsedResumeSchema(
        name=name,
        email=email,
        phone=phone,
        skills=skills,
        experience=experience,
        education=education,
        raw_text=raw_text_for_storage,
    )

    logger.info(
        "resume_parsing_complete",
        name=name,
        num_skills=len(skills),
        num_experience=len(experience),
        num_education=len(education),
    )
    return result
