"""
Stage 2 – Analysis & Scoring Module
=====================================
Implements the weighted ATS scoring system:

  Component                Weight
  ─────────────────────────────────
  Hard Skill Match          50%
  Semantic Experience Match 40%
  Education Match           10%
  ─────────────────────────────────
  Total ATS Score          100%

Semantic matching uses sentence-transformers (all-MiniLM-L6-v2)
with cosine similarity between experience descriptions and JD responsibilities.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.schemas import GapAnalysis, ParsedJDSchema, ParsedResumeSchema, ScoringBreakdown

logger = get_logger(__name__)

# Education level ranking (higher = more advanced)
_EDUCATION_LEVELS = {
    "phd": 5, "doctorate": 5, "ph.d": 5,
    "master": 4, "msc": 4, "m.sc": 4, "mba": 4, "m.tech": 4, "m.s.": 4, "m.e.": 4,
    "bachelor": 3, "bsc": 3, "b.sc": 3, "b.tech": 3, "b.e.": 3, "b.s.": 3, "undergraduate": 3,
    "associate": 2, "diploma": 2,
    "high school": 1, "secondary": 1,
}


def _normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison (lowercase, strip, collapse spaces)."""
    return re.sub(r"\s+", " ", skill.lower().strip())


# ── Component 1: Hard Skill Matching (50%) ────────────────────────────────────

def calculate_hard_skill_score(
    resume_skills: List[str],
    jd_required_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Calculate overlap between resume skills and JD required skills.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_required_skills: Skills extracted from the JD.

    Returns:
        Tuple of (score_0_to_100, matched_skills, missing_skills)
    """
    if not jd_required_skills:
        logger.warning("hard_skill_score_no_jd_skills")
        return 100.0, [], []

    resume_normalized = {_normalize_skill(s) for s in resume_skills}
    jd_normalized = [_normalize_skill(s) for s in jd_required_skills]

    matched = [
        s for s in jd_required_skills
        if _normalize_skill(s) in resume_normalized
    ]
    missing = [
        s for s in jd_required_skills
        if _normalize_skill(s) not in resume_normalized
    ]

    score = (len(matched) / len(jd_normalized)) * 100.0

    logger.info(
        "hard_skill_score",
        score=round(score, 2),
        matched=len(matched),
        missing=len(missing),
        total_jd_skills=len(jd_normalized),
    )
    return round(score, 2), matched, missing


# ── Component 2: Semantic Experience Matching (40%) ──────────────────────────

class SemanticMatcher:
    """
    Lazy-loaded semantic matcher using sentence-transformers.
    
    The model is loaded ONCE (singleton pattern) and reused across tasks.
    Loading is skipped if no responsibilities or experience are provided.
    """

    _instance: Optional["SemanticMatcher"] = None
    _model = None

    @classmethod
    def get_instance(cls) -> "SemanticMatcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Load the sentence transformer model (deferred until first use)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(
                    "loading_sentence_transformer",
                    model=settings.SENTENCE_TRANSFORMER_MODEL,
                )
                self._model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
                logger.info("sentence_transformer_loaded")
            except ImportError:
                logger.error(
                    "sentence_transformers_not_installed",
                    message="Run: pip install sentence-transformers",
                )
                raise
        return self._model

    def compute_similarity(
        self,
        jd_responsibilities: List[str],
        resume_experience_descriptions: List[str],
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute average best-match cosine similarity.

        For each JD responsibility, find the highest cosine similarity
        with any of the resume experience descriptions. The final score is
        the mean of these best-match scores.

        Returns:
            Tuple of (score_0_to_100, strong_matches, weak_matches)
        """
        if not jd_responsibilities or not resume_experience_descriptions:
            logger.warning("semantic_matcher_empty_inputs")
            return 50.0, [], []

        import numpy as np

        model = self._load_model()

        logger.info(
            "semantic_matching_started",
            num_responsibilities=len(jd_responsibilities),
            num_experiences=len(resume_experience_descriptions),
        )

        # Encode all texts in batch (much faster than one-by-one)
        jd_embeddings = model.encode(
            jd_responsibilities, convert_to_numpy=True, show_progress_bar=False
        )
        resume_embeddings = model.encode(
            resume_experience_descriptions,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Cosine similarity matrix: shape (num_jd, num_resume)
        jd_norm = jd_embeddings / (
            np.linalg.norm(jd_embeddings, axis=1, keepdims=True) + 1e-9
        )
        resume_norm = resume_embeddings / (
            np.linalg.norm(resume_embeddings, axis=1, keepdims=True) + 1e-9
        )
        similarity_matrix = np.dot(jd_norm, resume_norm.T)

        # Best match scores for each JD responsibility
        best_scores = similarity_matrix.max(axis=1)  # shape: (num_jd,)

        STRONG_THRESHOLD = 0.65
        strong_matches = [
            jd_responsibilities[i]
            for i, sc in enumerate(best_scores)
            if sc >= STRONG_THRESHOLD
        ]
        weak_matches = [
            jd_responsibilities[i]
            for i, sc in enumerate(best_scores)
            if sc < STRONG_THRESHOLD
        ]

        avg_score = float(best_scores.mean()) * 100.0

        logger.info(
            "semantic_matching_complete",
            avg_score=round(avg_score, 2),
            strong_matches=len(strong_matches),
            weak_matches=len(weak_matches),
        )
        return round(avg_score, 2), strong_matches, weak_matches


def calculate_semantic_score(
    jd_responsibilities: List[str],
    resume_experience: List[Dict],
) -> Tuple[float, List[str], List[str]]:
    """
    Public interface to semantic matching.

    Extracts description strings from the resume experience dicts,
    then delegates to SemanticMatcher.

    Returns:
        Tuple of (score_0_to_100, strong_matches_list, weak_matches_list)
    """
    # Flatten experience dict → list of description strings
    descriptions = []
    for exp in resume_experience:
        if isinstance(exp, dict):
            desc = exp.get("description", "")
            title = exp.get("title", "")
            company = exp.get("company", "")
            descriptions.append(f"{title} at {company}: {desc}".strip())

    if not descriptions:
        logger.warning("semantic_score_no_experience_descriptions")
        return 0.0, [], jd_responsibilities

    matcher = SemanticMatcher.get_instance()
    return matcher.compute_similarity(jd_responsibilities, descriptions)


# ── Component 3: Education Matching (10%) ─────────────────────────────────────

def calculate_education_score(
    resume_education: List[Dict],
    jd_raw_text: str,
) -> float:
    """
    Check if the candidate's education level meets typical JD requirements.

    Heuristic:
    - If JD mentions "PhD" and candidate has PhD → 100
    - If JD mentions "Master" and candidate has Master or higher → 100
    - If JD mentions "Bachelor" and candidate has Bachelor or higher → 100
    - If no education requirement detected in JD → default to 80 (benefit of doubt)

    Returns:
        Score from 0 to 100.
    """
    if not resume_education:
        return 50.0  # Can't determine — partial credit

    jd_lower = jd_raw_text.lower() if jd_raw_text else ""

    # Determine what the JD requires
    jd_required_level = 0
    for keyword, level in _EDUCATION_LEVELS.items():
        if keyword in jd_lower:
            jd_required_level = max(jd_required_level, level)

    if jd_required_level == 0:
        # No clear requirement found → give benefit of doubt
        return 80.0

    # Determine candidate's highest education level
    candidate_level = 0
    for edu in resume_education:
        if isinstance(edu, dict):
            degree_text = (str(edu.get("degree", "")) + " " + str(edu.get("field", ""))).lower()
            for keyword, level in _EDUCATION_LEVELS.items():
                if keyword in degree_text:
                    candidate_level = max(candidate_level, level)

    if candidate_level >= jd_required_level:
        score = 100.0
    elif candidate_level == jd_required_level - 1:
        score = 60.0  # One level below → partial credit
    else:
        score = 20.0  # Significantly below

    logger.info(
        "education_score",
        score=score,
        jd_required_level=jd_required_level,
        candidate_level=candidate_level,
    )
    return score


# ── Gap Analysis ──────────────────────────────────────────────────────────────

def compute_gap_analysis(
    resume: ParsedResumeSchema,
    jd: ParsedJDSchema,
    matched_skills: List[str],
    missing_skills: List[str],
    strong_resp: List[str],
    weak_resp: List[str],
) -> GapAnalysis:
    """Build a structured gap analysis object."""
    return GapAnalysis(
        missing_skills=missing_skills,
        matched_skills=matched_skills,
        weakly_matched_responsibilities=weak_resp,
        strong_matched_responsibilities=strong_resp,
    )


# ── Score Aggregator ──────────────────────────────────────────────────────────

WEIGHTS = {
    "hard_skill": 0.50,
    "semantic": 0.40,
    "education": 0.10,
}


def aggregate_scores(
    hard_skill_score: float,
    semantic_score: float,
    education_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    strong_resp: List[str],
    weak_resp: List[str],
) -> ScoringBreakdown:
    """
    Apply weighted formula to produce the final ATS score.

    Formula:
      total = (hard_skill * 0.50) + (semantic * 0.40) + (education * 0.10)

    Returns a ScoringBreakdown Pydantic model with all components.
    """
    # Clamp each component to [0, 100] before aggregating
    hard_skill_score = round(min(max(hard_skill_score, 0.0), 100.0), 2)
    semantic_score = round(min(max(semantic_score, 0.0), 100.0), 2)
    education_score = round(min(max(education_score, 0.0), 100.0), 2)

    total = (
        hard_skill_score * WEIGHTS["hard_skill"]
        + semantic_score * WEIGHTS["semantic"]
        + education_score * WEIGHTS["education"]
    )
    total = round(min(max(total, 0), 100), 2)

    logger.info(
        "ats_score_aggregated",
        total=total,
        hard_skill=hard_skill_score,
        semantic=semantic_score,
        education=education_score,
    )

    return ScoringBreakdown(
        hard_skill_score=hard_skill_score,
        semantic_score=semantic_score,
        education_score=education_score,
        total_ats_score=total,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )


# ── Top-Level Pipeline Function ───────────────────────────────────────────────

def run_scoring_pipeline(
    resume: ParsedResumeSchema,
    jd: ParsedJDSchema,
) -> Tuple[ScoringBreakdown, GapAnalysis]:
    """
    Execute the full scoring pipeline for a resume + JD pair.

    Args:
        resume: Structured resume data.
        jd: Structured job description data.

    Returns:
        Tuple of (ScoringBreakdown, GapAnalysis)
    """
    logger.info("scoring_pipeline_started")

    # Component 1: Hard skill match
    hard_skill_score, matched_skills, missing_skills = calculate_hard_skill_score(
        resume.skills, jd.required_skills
    )

    # Component 2: Semantic experience match
    sem_score, strong_resp, weak_resp = calculate_semantic_score(
        jd.responsibilities, resume.experience
    )

    # Component 3: Education match
    edu_score = calculate_education_score(
        resume.education, jd.raw_text or ""
    )

    # Aggregate
    scoring = aggregate_scores(
        hard_skill_score, sem_score, edu_score,
        matched_skills, missing_skills, strong_resp, weak_resp,
    )

    gap = compute_gap_analysis(
        resume, jd, matched_skills, missing_skills, strong_resp, weak_resp
    )

    logger.info("scoring_pipeline_complete", ats_score=scoring.total_ats_score)
    return scoring, gap
