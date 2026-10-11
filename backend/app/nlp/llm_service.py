"""
Stage 3 & 4 – Generative AI Module
=====================================
Provides a unified LLM service that calls the Hugging Face Inference API
(router.huggingface.co – OpenAI-compatible Chat Completions endpoint)
instead of loading models locally.

Key design decisions:
  - NO local model weights are downloaded. All inference is done via HTTP.
  - Uses the new router.huggingface.co/v1/chat/completions endpoint.
  - Request format is OpenAI-compatible (messages array).
  - The same HUGGINGFACE_TOKEN from .env is reused for authentication.
  - Retries with exponential backoff handle transient 503s.
  - Separate model IDs for suggestions (Mistral) and cover letters (Llama).
"""

import re
import time
from typing import List, Optional

import httpx

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.schemas import GapAnalysis, ParsedJDSchema, ParsedResumeSchema

logger = get_logger(__name__)

# ── HuggingFace Inference API base ───────────────────────────────────────────

# HuggingFace migrated from api-inference.huggingface.co → router.huggingface.co
# The new endpoint is OpenAI-compatible (Chat Completions API).
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TEXT_GEN_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"

# Model IDs used for each task (can be overridden via .env)
# Mistral is freely accessible without gating.
# Llama-2 requires accepting the licence on huggingface.co once.
_SUGGESTION_MODEL_ID = settings.SUGGESTION_LLM_MODEL        # mistralai/Mistral-7B-Instruct-v0.2
_COVER_LETTER_MODEL_ID = settings.COVER_LETTER_BASE_MODEL   # meta-llama/Llama-2-7b-chat-hf


# ── Prompt Templates ─────────────────────────────────────────────────────────

SUGGESTION_PROMPT_TEMPLATE = """\
### INSTRUCTION
You are an expert AI career strategist. Your goal is to provide concise, actionable feedback to help a user align their resume with a job description. Analyze the provided context and generate 3-5 specific suggestions formatted as a bulleted list. Address the user directly and maintain an encouraging tone.

### JOB DESCRIPTION CONTEXT
---
Job Title: {job_title}
Required Skills: {required_skills}
Key Responsibilities:
{responsibilities}

Full Job Description Text:
{full_jd_text}
---

### RESUME CONTEXT
---
Skills Listed by Applicant: {resume_skills}
Summary of Applicant's Experience:
{experience_summary}

Full Parsed Resume Payload (JSON):
{full_resume_json}
---

### GAP ANALYSIS SUMMARY
---
- Critical Skills Mentioned in Job but Missing from Resume: {missing_skills}
- Job Responsibilities with Low Semantic Match to Resume Experience: {weak_responsibilities}
---

### YOUR SUGGESTIONS
-"""

COVER_LETTER_PROMPT_TEMPLATE = """\
### SYSTEM INSTRUCTION
You are a professional cover-letter writer.
Return ONLY the final cover letter text.
Do NOT include notes, explanations, tips, placeholders, or postscript guidance.
Do NOT include lines such as "Feel free to adjust..." or any similar advisory text.
Write in a continuous letter flow using coherent paragraphs.
Do NOT use section headings such as "Professional Background", "Conclusion", or markdown headings like "###".
Do NOT output markdown syntax such as **bold**, *italic*, # headings, or bullet lists.

### Job Description
Job Title: {job_title}
Company: {company}
Preferred Qualifications: {required_skills}
Responsibilities:
{responsibilities}

Full Job Description Text:
{full_jd_text}

### Applicant Resume
Name: {name}
Email: {email}
Phone: {phone}
Current Experience: {current_experience}
Past Experience: {past_experiences}
Skills: {skills}
Qualifications: {education}
Detailed Experience:
{detailed_experience}
Detailed Education:
{detailed_education}

Full Parsed Resume Payload (JSON):
{full_resume_json}

### Cover Letter
Write a professional, personalized cover letter with complete applicant details from the parsed resume context.
Output format rules:
1) Start directly with a salutation line (e.g., Dear Hiring Manager,).
2) Include 3-5 body paragraphs tailored to the role.
3) End with a professional closing and applicant name.
4) Output ONLY the letter text and nothing else."""


def _format_suggestion_prompt(
    resume: ParsedResumeSchema,
    jd: ParsedJDSchema,
    gap: GapAnalysis,
    jd_full_text: Optional[str] = None,
) -> str:
    """Build the full suggestion prompt string from structured data."""
    responsibilities_text = "\n".join(
        f"- {r}" for r in jd.responsibilities
    ) or "Not specified"

    full_jd_text = (jd_full_text or jd.raw_text or "").strip() or "Not available"
    max_jd_chars = settings.LLM_JD_CONTEXT_MAX_CHARS
    if max_jd_chars > 0:
        full_jd_text = full_jd_text[:max_jd_chars]

    experience_summary = "\n".join(
        f"- {exp.get('title', 'Role')} at {exp.get('company', 'Company')}: "
        f"{str(exp.get('description', ''))[:200]}"
        for exp in resume.experience[:5]
    ) or "No experience listed"

    full_resume_json = resume.model_dump_json(indent=2)
    max_resume_chars = settings.LLM_RESUME_CONTEXT_MAX_CHARS
    if max_resume_chars > 0:
        full_resume_json = full_resume_json[:max_resume_chars]

    return SUGGESTION_PROMPT_TEMPLATE.format(
        job_title=jd.job_title or "Not specified",
        required_skills=", ".join(jd.required_skills) or "Not specified",
        responsibilities=responsibilities_text,
        full_jd_text=full_jd_text,
        resume_skills=", ".join(resume.skills[:25]) or "None listed",
        experience_summary=experience_summary,
        full_resume_json=full_resume_json,
        missing_skills=", ".join(gap.missing_skills[:10]) or "None identified",
        weak_responsibilities="\n".join(
            f"- {r}" for r in gap.weakly_matched_responsibilities[:5]
        ) or "None",
    )


def _format_cover_letter_prompt(
    resume: ParsedResumeSchema,
    jd: ParsedJDSchema,
    jd_full_text: Optional[str] = None,
) -> str:
    """Build the cover letter prompt."""
    current_exp = ""
    past_exps = ""
    if resume.experience:
        first = resume.experience[0]
        current_exp = (
            f"{first.get('title', 'Professional')} at "
            f"{first.get('company', 'a company')} "
            f"({first.get('duration', 'recent')})"
        )
        if len(resume.experience) > 1:
            past_exps = "; ".join(
                f"{e.get('title', 'Role')} at {e.get('company', 'Company')}"
                for e in resume.experience[1:4]
            )

    education_summary = "; ".join(
        f"{e.get('degree', 'Degree')} from {e.get('institution', 'University')}"
        for e in resume.education[:2]
    ) or "Not specified"

    detailed_experience = "\n".join(
        "- "
        f"{e.get('title', 'Role')} at {e.get('company', 'Company')} "
        f"({e.get('duration', 'N/A')}): {str(e.get('description', ''))[:350]}"
        for e in resume.experience
    ) or "- Not specified"

    detailed_education = "\n".join(
        "- "
        f"{e.get('degree', 'Degree')} in {e.get('field', 'N/A')} from "
        f"{e.get('institution', 'University')} ({e.get('year', 'N/A')})"
        for e in resume.education
    ) or "- Not specified"

    responsibilities_text = "\n".join(
        f"- {r}" for r in jd.responsibilities
    ) or "Not specified"

    full_jd_text = (jd_full_text or jd.raw_text or "").strip() or "Not available"
    max_jd_chars = settings.LLM_JD_CONTEXT_MAX_CHARS
    if max_jd_chars > 0:
        full_jd_text = full_jd_text[:max_jd_chars]

    full_resume_json = resume.model_dump_json(indent=2)
    max_resume_chars = settings.LLM_RESUME_CONTEXT_MAX_CHARS
    if max_resume_chars > 0:
        full_resume_json = full_resume_json[:max_resume_chars]

    return COVER_LETTER_PROMPT_TEMPLATE.format(
        job_title=jd.job_title or "the position",
        company=jd.company or "your company",
        required_skills=", ".join(jd.required_skills + jd.preferred_skills),
        responsibilities=responsibilities_text,
        full_jd_text=full_jd_text,
        name=resume.name or "Applicant",
        email=resume.email or "Not provided",
        phone=resume.phone or "Not provided",
        current_experience=current_exp or "Professional with relevant experience",
        past_experiences=past_exps or "Various relevant positions",
        skills=", ".join(resume.skills[:20]) or "Diverse professional skills",
        education=education_summary,
        detailed_experience=detailed_experience,
        detailed_education=detailed_education,
        full_resume_json=full_resume_json,
    )


# ── HuggingFace Inference API Client ─────────────────────────────────────────

class HFInferenceClient:
    """
    Thin wrapper around the HuggingFace router (router.huggingface.co).

    Uses the OpenAI-compatible Chat Completions API:
      POST https://router.huggingface.co/v1/chat/completions

    Handles:
      - Authentication via Bearer token
      - Retries with exponential backoff for 503 (model warming up)
      - Configurable timeout
    """

    MAX_RETRIES = 5
    INITIAL_BACKOFF_SECONDS = 10

    def __init__(self, token: str):
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _generate_via_text_endpoint(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        timeout: int,
    ) -> str:
        """Fallback for non-chat models using HF Inference text generation API."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,
            },
        }

        url = HF_TEXT_GEN_URL_TEMPLATE.format(model_id=model_id)
        response = httpx.post(
            url,
            headers=self._headers,
            json=payload,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HF text generation API returned {response.status_code}: "
                f"{response.text[:400]}"
            )

        data = response.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text", "")
            return str(text).strip()
        if isinstance(data, dict):
            text = data.get("generated_text", "")
            return str(text).strip()

        raise RuntimeError(f"Unexpected HF text API response shape: {data}")

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 120,
    ) -> str:
        """
        Call the HF Router Chat Completions endpoint and return the reply text.

        The prompt is sent as a single user message (OpenAI format).
        Retries automatically on 503 (model still warming up on HF infra).
        Raises RuntimeError on non-retryable errors.
        """
        # OpenAI-compatible payload
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    "hf_api_request",
                    model=model_id,
                    attempt=attempt,
                    prompt_length=len(prompt),
                )
                response = httpx.post(
                    HF_ROUTER_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    # OpenAI-compatible response shape:
                    # {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
                    choices = data.get("choices", [])
                    if choices and isinstance(choices, list):
                        message = choices[0].get("message", {})
                        return str(message.get("content", "")).strip()
                    raise RuntimeError(f"Unexpected HF router response shape: {data}")

                elif response.status_code in (503, 429):
                    # 503 = model warming up, 429 = rate limited – both are retryable
                    wait_seconds = self.INITIAL_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "hf_api_retrying",
                        model=model_id,
                        status=response.status_code,
                        attempt=attempt,
                        retry_in_seconds=wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    last_error = f"{response.status_code} on attempt {attempt}"

                elif response.status_code == 401:
                    raise RuntimeError(
                        "HuggingFace token is invalid or missing. "
                        "Set HUGGINGFACE_TOKEN in your .env file. "
                        "Get one at https://huggingface.co/settings/tokens"
                    )

                elif response.status_code == 403:
                    raise RuntimeError(
                        f"Access denied for model '{model_id}'. "
                        "If it's a gated model (e.g. Llama-2), visit the model page "
                        "on huggingface.co and accept the licence terms."
                    )

                else:
                    response_text = response.text[:400]
                    if (
                        response.status_code == 400
                        and "not a chat model" in response_text.lower()
                    ):
                        logger.warning(
                            "hf_router_non_chat_model_fallback",
                            model=model_id,
                            endpoint="api-inference-text-generation",
                        )
                        return self._generate_via_text_endpoint(
                            model_id=model_id,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            timeout=timeout,
                        )

                    raise RuntimeError(
                        f"HF API returned {response.status_code}: {response_text}"
                    )

            except httpx.TimeoutException:
                wait_seconds = self.INITIAL_BACKOFF_SECONDS * attempt
                logger.warning(
                    "hf_api_timeout",
                    model=model_id,
                    attempt=attempt,
                    retry_in_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)
                last_error = f"Timeout on attempt {attempt}"

        raise RuntimeError(
            f"HF Router API failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# ── LLMService Singleton ──────────────────────────────────────────────────────

class LLMService:
    """
    Singleton wrapper for all LLM inference via the HF Inference API.

    No local model loading — everything is done over HTTPS using
    the HUGGINGFACE_TOKEN from your .env.
    """

    _instance: Optional["LLMService"] = None
    _client: Optional[HFInferenceClient] = None

    @classmethod
    def get_instance(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_client(self) -> HFInferenceClient:
        if self._client is None:
            token = settings.HUGGINGFACE_TOKEN
            if not token:
                raise RuntimeError(
                    "HUGGINGFACE_TOKEN is not set. "
                    "Add it to your .env file. "
                    "Get a free token at https://huggingface.co/settings/tokens"
                )
            self._client = HFInferenceClient(token=token)
        client = self._client
        if client is None:
            raise RuntimeError("HFInferenceClient failed to initialize.")
        return client

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_suggestions(
        self,
        resume: ParsedResumeSchema,
        jd: ParsedJDSchema,
        gap: GapAnalysis,
        jd_full_text: Optional[str] = None,
    ) -> List[str]:
        """
        Generate 3-5 actionable resume improvement suggestions via HF API.

        Returns a list of suggestion strings.
        """
        logger.info("suggestion_generation_started", model=_SUGGESTION_MODEL_ID)

        prompt = _format_suggestion_prompt(resume, jd, gap, jd_full_text=jd_full_text)
        client = self._get_client()

        raw_output = client.generate(
            model_id=_SUGGESTION_MODEL_ID,
            prompt=prompt,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
        )

        suggestions = _parse_suggestions(raw_output)
        logger.info("suggestion_generation_complete", count=len(suggestions))
        return suggestions

    def generate_cover_letter(
        self,
        resume: ParsedResumeSchema,
        jd: ParsedJDSchema,
        jd_full_text: Optional[str] = None,
    ) -> str:
        """
        Generate a professional cover letter via HF API.

        Returns cleaned cover letter text.
        """
        logger.info("cover_letter_generation_started", model=_COVER_LETTER_MODEL_ID)

        prompt = _format_cover_letter_prompt(resume, jd, jd_full_text=jd_full_text)
        client = self._get_client()

        raw_output = client.generate(
            model_id=_COVER_LETTER_MODEL_ID,
            prompt=prompt,
            max_new_tokens=800,
            temperature=0.7,
            top_p=settings.LLM_TOP_P,
            timeout=180,  # cover letters are longer; give more time
        )

        cleaned = _clean_cover_letter_output(raw_output)
        logger.info("cover_letter_generation_complete", length=len(cleaned))
        return cleaned


# ── Output Cleaning ───────────────────────────────────────────────────────────

def _parse_suggestions(raw_text: str) -> List[str]:
    """
    Parse model output into a clean list of suggestion strings.
    Handles various bullet formats: -, •, *, numbered lists.
    """
    lines = raw_text.strip().split("\n")
    suggestions = []

    for line in lines:
        stripped = line.strip()
        # Strip bullet markers
        for prefix in ["-", "•", "*", "·"]:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                break
        # Strip numbered list markers like "1." or "1)"
        stripped = re.sub(r"^\d+[\.\)]\s*", "", stripped)

        if len(stripped) > 20:  # Filter out very short/empty lines
            suggestions.append(stripped)

    return suggestions[:5]  # Return max 5 suggestions


def _clean_cover_letter_output(raw_text: str) -> str:
    """
    Clean up the raw model output for cover letters.
    Strip everything before "Dear" to get the letter body.
    """
    raw_text = re.sub(
        r"(?im)^\s*(feel free to adjust[^\n]*|note:\s*[^\n]*|disclaimer:\s*[^\n]*)\s*$",
        "",
        raw_text,
    )

    # Remove markdown heading lines (e.g., ### Professional Background)
    raw_text = re.sub(r"(?im)^\s*#{1,6}\s+.*$", "", raw_text)

    # Remove explicit section-heading style lines that often break letter flow.
    raw_text = re.sub(
        r"(?im)^\s*(professional background|relevant skills and certifications|why [^\n:]+\??|conclusion)\s*:??\s*$",
        "",
        raw_text,
    )

    # Normalize markdown emphasis markers into plain text if model returns them.
    raw_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_text)
    raw_text = re.sub(r"\*(.*?)\*", r"\1", raw_text)

    dear_match = re.search(r"\bDear\b", raw_text, re.IGNORECASE)
    if dear_match:
        raw_text = raw_text[dear_match.start():]

    raw_text = raw_text.strip()
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    return raw_text
