"""
spaCy Model Loader
==================
Centralized loader for spaCy NLP models.
Ensures the model is loaded ONCE (singleton) and reused across NLP pipeline calls.
"""

from typing import Optional

import spacy
from spacy.language import Language

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_nlp_model: Optional[Language] = None


def get_nlp_model() -> Optional[Language]:
    """
    Lazy-load and cache the spaCy language model.

    Returns the model if available. Returns None if the model
    is not installed (allows pipeline to run in degraded mode,
    using only regex-based extraction).

    To install: python -m spacy download en_core_web_lg
    """
    global _nlp_model

    if _nlp_model is not None:
        return _nlp_model

    try:
        logger.info("loading_spacy_model", model=settings.SPACY_MODEL)
        _nlp_model = spacy.load(settings.SPACY_MODEL)
        logger.info("spacy_model_loaded", model=settings.SPACY_MODEL)
        return _nlp_model
    except OSError:
        logger.warning(
            "spacy_model_not_found",
            model=settings.SPACY_MODEL,
            message=(
                f"spaCy model '{settings.SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {settings.SPACY_MODEL}. "
                "Falling back to regex-only NLP."
            ),
        )
        return None
