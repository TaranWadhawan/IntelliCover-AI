"""
Stage 1 – Job Description Scraper & Parser
==========================================
Scrapes a job description URL and extracts structured data.

Two-tier scraping strategy:
  Tier 1: httpx + BeautifulSoup (fast, handles static sites)
  Tier 2: Playwright headless browser (falls back for JS-heavy sites)

After scraping, uses regex + NER to produce a ParsedJDSchema.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.schemas import ParsedJDSchema

logger = get_logger(__name__)

# Job-board-friendly headers to reduce bot-detection blocks
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# HTML tags whose content we want to discard completely
_TAGS_TO_REMOVE = {
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "svg", "img", "figure", "picture",
}

# Regex patterns for JD section detection
_JD_SECTION_PATTERNS = {
    "responsibilities": [
        r"responsibilities", r"what you.ll do", r"your role",
        r"key duties", r"duties", r"you will",
    ],
    "requirements": [
        r"requirements", r"qualifications", r"what we.re looking for",
        r"must have", r"must-have", r"required skills", r"you have",
        r"minimum qualifications",
    ],
    "preferred": [
        r"preferred", r"nice to have", r"bonus", r"plus", r"advantageous",
    ],
}

# Known skill terms (shared with resume_parser, duplicated here for standalone usage)
_SKILL_SEED = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "kotlin", "swift", "r", "sql", "postgresql", "mysql", "mongodb",
    "react", "vue", "angular", "next.js", "fastapi", "django", "flask",
    "spring", "express", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "numpy", "spark", "kafka", "aws", "gcp", "azure", "docker", "kubernetes",
    "terraform", "ansible", "machine learning", "deep learning", "nlp",
    "computer vision", "data science", "mlops", "llm", "bert", "agile", "scrum",
    "redis", "elasticsearch", "snowflake", "bigquery", "git", "linux",
}


# ── HTML Cleaning ─────────────────────────────────────────────────────────────

def _html_to_clean_text(html: str) -> str:
    """
    Convert raw HTML to clean plain text, stripping noise tags.
    Preserves list structure by converting <li> to "• item" lines.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.find_all(_TAGS_TO_REMOVE):
        tag.decompose()

    # Convert list items to text with bullet prefix
    for li in soup.find_all("li"):
        li.insert(0, "• ")
        li.append("\n")

    # Preserve heading structure
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.insert(0, "\n\n### ")
        tag.append("\n")

    text = soup.get_text(separator="\n")

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── Tier 1: httpx + BeautifulSoup ────────────────────────────────────────────

async def _scrape_with_httpx(url: str) -> Optional[str]:
    """
    Attempt a simple HTTP GET request and return clean text.
    Returns None if the request fails or yields < 200 chars of content.
    """
    try:
        async with httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=settings.SCRAPER_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        clean_text = _html_to_clean_text(response.text)

        if len(clean_text) < 200:
            logger.warning(
                "httpx_scrape_insufficient_content",
                url=url,
                char_count=len(clean_text),
            )
            return None

        logger.info("httpx_scrape_success", url=url, char_count=len(clean_text))
        return clean_text

    except httpx.HTTPStatusError as exc:
        logger.warning("httpx_scrape_http_error", url=url, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("httpx_scrape_request_error", url=url, error=str(exc))
        return None


# ── Tier 2: Playwright (Headless Browser) ────────────────────────────────────

async def _scrape_with_playwright(url: str) -> Optional[str]:
    """
    Full browser rendering using Playwright.
    Falls back to this when httpx fails (JS-heavy sites, SPAs, etc.).
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=_DEFAULT_HEADERS["User-Agent"],
                locale="en-US",
            )
            page = await context.new_page()

            # Block images/fonts for speed
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf}",
                lambda route: route.abort(),
            )

            await page.goto(url, timeout=settings.SCRAPER_TIMEOUT_SECONDS * 1000)
            # Wait for main content to stabilize
            await page.wait_for_load_state("networkidle", timeout=15000)

            html = await page.content()
            await browser.close()

        clean_text = _html_to_clean_text(html)
        if len(clean_text) < 200:
            logger.warning("playwright_scrape_insufficient_content", url=url)
            return None

        logger.info("playwright_scrape_success", url=url, char_count=len(clean_text))
        return clean_text

    except ImportError:
        logger.error(
            "playwright_not_installed",
            message="Install with: `pip install playwright && playwright install chromium`",
        )
        return None
    except Exception as exc:
        logger.error("playwright_scrape_error", url=url, error=str(exc))
        return None


# ── Main Scraper Entrypoint ───────────────────────────────────────────────────

async def scrape_job_description(url: str) -> str:
    """
    Orchestrates the two-tier scraping strategy.

    Strategy:
      1. Try httpx (fast). If it returns sufficient content → use it.
      2. If httpx fails and Playwright is enabled → use Playwright.
      3. If both fail → raise a descriptive exception.

    Args:
        url: Publicly accessible URL of the job description page.

    Returns:
        Cleaned plain text of the job description.

    Raises:
        ValueError: If neither scraping method yields usable content.
    """
    if url.startswith("text:"):
        logger.info("jd_scraping_bypassed_using_raw_text")
        return url[5:].strip()

    logger.info("jd_scraping_started", url=url)

    text = await _scrape_with_httpx(url)

    if text is None and settings.SCRAPER_USE_PLAYWRIGHT_FALLBACK:
        logger.info("escalating_to_playwright", url=url)
        text = await _scrape_with_playwright(url)

    if not text:
        raise ValueError(
            f"We couldn't access the job description at '{url}'. "
            "This sometimes happens with sites that require a login or use "
            "heavy JavaScript. Please try pasting the job description text directly."
        )

    return text


# ── JD Structuring ────────────────────────────────────────────────────────────

def _split_jd_into_sections(text: str) -> Dict[str, List[str]]:
    """
    Split the JD text into logical sections using header detection.
    Returns dict with lists of bullet-point strings per section.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {
        "intro": [],
        "responsibilities": [],
        "requirements": [],
        "preferred": [],
        "other": [],
    }
    current_section = "intro"

    for line in lines:
        stripped = line.strip().lower().rstrip(":")
        matched = False
        for section_name, patterns in _JD_SECTION_PATTERNS.items():
            if any(re.search(p, stripped) for p in patterns):
                current_section = section_name
                matched = True
                break

        if not matched and line.strip():
            # Skip heading markers we inserted
            clean_line = line.strip().lstrip("•").lstrip("###").strip()
            if clean_line:
                sections[current_section].append(clean_line)

    return sections


def _extract_skills_from_jd(text: str, nlp=None) -> List[str]:
    """Extract required and preferred skills from JD text."""
    found: set = set()
    text_lower = text.lower()

    for skill in _SKILL_SEED:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found.add(skill.title() if len(skill) > 3 else skill.upper())

    if nlp:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            if ent.label_ in ("PRODUCT", "ORG", "LANGUAGE"):
                candidate = ent.text.strip()
                if 2 < len(candidate) < 50:
                    found.add(candidate)

    return sorted(found)


def _extract_company_name(text: str, nlp=None) -> Optional[str]:
    """Try to find the hiring company name."""
    if nlp:
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ == "ORG":
                candidate = ent.text.strip()
                if 1 < len(candidate) < 100:
                    return candidate

    # Regex fallback: "At {Company}" or "{Company} is hiring"
    patterns = [
        r"at ([A-Z][A-Za-z\s&,\.]+?)(,| we| is| are| has)",
        r"([A-Z][A-Za-z\s&,\.]+?) is (hiring|looking|seeking)",
        r"company:\s*(.+)",
        r"employer:\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000])
        if match:
            return match.group(1).strip()

    return None


def _extract_job_title(text: str, nlp=None) -> Optional[str]:
    """Extract the job title from the beginning of the JD."""
    # Common patterns: "Job Title:", "Role:", "Position:"
    title_patterns = [
        r"(?:job title|position|role|title):\s*(.+)",
        r"^### (.+)",  # Our heading markup
        r"we are looking for (?:a |an )?(.+?) to join",
        r"hiring (?:a |an )?(.+?) for",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, text[:1000], re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip().rstrip(".")
            if 3 < len(candidate) < 100:
                return candidate

    # Fallback: first line that looks like a title
    for line in text.split("\n")[:5]:
        line = line.strip().lstrip("###").strip()
        if line and 3 < len(line) < 100 and line[0].isupper():
            return line

    return None


def parse_job_description(raw_text: str, nlp=None) -> ParsedJDSchema:
    """
    Structure raw JD text into a ParsedJDSchema.

    Args:
        raw_text: Clean plain text of the job description.
        nlp: Optional pre-loaded spaCy model.

    Returns:
        ParsedJDSchema with job_title, company, skills, responsibilities, etc.
    """
    logger.info("jd_parsing_started", text_length=len(raw_text))

    sections = _split_jd_into_sections(raw_text)

    job_title = _extract_job_title(raw_text, nlp)
    company = _extract_company_name(raw_text, nlp)

    # Combine requirements + preferred for skill extraction
    requirements_text = "\n".join(sections["requirements"] + sections["preferred"])
    all_required_skills = _extract_skills_from_jd(requirements_text or raw_text, nlp)

    # Preferred skills: only from the "preferred" section
    preferred_skills = _extract_skills_from_jd(
        "\n".join(sections["preferred"]), nlp
    ) if sections["preferred"] else []

    # Required = all - preferred
    preferred_set = set(preferred_skills)
    required_skills = [s for s in all_required_skills if s not in preferred_set]

    responsibilities = [
        r for r in sections["responsibilities"] if len(r) > 15
    ][:15]  # cap at 15

    max_raw_chars = settings.JD_RAW_TEXT_MAX_CHARS
    raw_text_for_storage = (
        raw_text if max_raw_chars <= 0 else raw_text[:max_raw_chars]
    )

    result = ParsedJDSchema(
        job_title=job_title,
        company=company,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        raw_text=raw_text_for_storage,
    )

    logger.info(
        "jd_parsing_complete",
        job_title=job_title,
        company=company,
        num_required_skills=len(required_skills),
        num_responsibilities=len(responsibilities),
    )
    return result
