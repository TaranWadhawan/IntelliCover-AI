# IntelliCover AI – Backend

An AI-powered career assistant backend built with FastAPI, Celery, and a multi-stage NLP pipeline.

## 🗂️ Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── auth.py          # AUTH endpoints (register, login, /me)
│   │       ├── jobs.py          # JOBS endpoints (analyze, status, cover-letter)
│   │       └── router.py        # Aggregates all v1 routers
│   ├── core/
│   │   ├── config.py            # All settings (environment variables via pydantic-settings)
│   │   ├── database.py          # Async SQLAlchemy engine + session factory
│   │   ├── logging_config.py    # Structured JSON logging (structlog)
│   │   └── security.py          # JWT + bcrypt password hashing
│   ├── models/
│   │   └── db_models.py         # SQLAlchemy ORM: User, AnalysisJob, CoverLetter
│   ├── nlp/                     # ← THE CORE NLP PIPELINE
│   │   ├── resume_parser.py     # Stage 1: PDF extraction + NER structuring
│   │   ├── jd_scraper.py        # Stage 1: Web scraping (httpx → Playwright fallback)
│   │   ├── scorer.py            # Stage 2: ATS scoring (hard skill + semantic + education)
│   │   ├── llm_service.py       # Stage 3&4: LLM singleton (Mistral + Llama-2 LoRA)
│   │   └── spacy_loader.py      # spaCy model singleton loader
│   ├── schemas/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── auth_dependency.py   # FastAPI JWT auth dependency
│   │   └── storage_service.py   # File storage (local or S3)
│   ├── worker/
│   │   ├── celery_app.py        # Celery application instance + config
│   │   └── tasks.py             # Celery tasks: analysis pipeline + cover letter
│   └── main.py                  # FastAPI app entrypoint (lifespan, CORS, routes)
├── alembic/                     # Database migrations
├── tests/
│   ├── test_scorer.py           # Unit tests: scoring functions (20+ cases)
│   └── test_resume_parser.py    # Unit tests: parser helpers
├── .env.example                 # Environment variables template
├── requirements.txt             # All dependencies
├── pytest.ini                   # Test configuration
├── ruff.toml                    # Linting / formatting
├── Dockerfile                   # API service image
├── Dockerfile.worker            # Celery worker image (with ML libs)
└── docker-compose.yml           # Full local development stack
```

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### 1. Set up environment

```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values (DB URL, Redis URL, HF token, etc.)
```

For higher quality suggestions and cover letters, keep full JD context enabled:

```bash
# 0 means no truncation (recommended for best AI grounding)
JD_RAW_TEXT_MAX_CHARS=0
LLM_JD_CONTEXT_MAX_CHARS=0
RESUME_RAW_TEXT_MAX_CHARS=0
LLM_RESUME_CONTEXT_MAX_CHARS=0
```

### 3. Set up the database

```bash
# Option A: Auto-create tables (dev only, set DEBUG=true in .env)
# Tables are created on first startup automatically.

# Option B: Alembic migrations (recommended for prod)
alembic upgrade head
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_lg
```

### 5. Install Playwright browser (for JS-heavy job sites)

```bash
playwright install chromium
```

### 6. Start the services

**Terminal 1 – FastAPI API server:**

```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 – Celery worker (NLP pipeline):**

```bash
celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
```

API docs available at: http://localhost:8000/docs

---

## 🐳 Docker Compose (Recommended)

```bash
# Start everything (DB + Redis + API + Worker + Flower)
docker-compose up

# Flower monitoring UI
open http://localhost:5555

# Stop and clean up
docker-compose down
```

---

## 🧠 NLP Pipeline

The NLP pipeline runs inside the Celery worker:

```
PDF Resume  ──→  [resume_parser.py]  ──→  ParsedResumeSchema (JSON)
                                                    │
Job URL  ──→  [jd_scraper.py]  ──→  ParsedJDSchema ┘
                                                    │
                                              [scorer.py]
                                                    │
                                    ┌───────────────┼──────────────────┐
                                    │               │                  │
                            Hard Skill Match  Semantic Match   Education Match
                               (50% weight)    (40% weight)     (10% weight)
                                    │               │                  │
                                    └───────────────┴──────────────────┘
                                                    │
                                            ATS Score (0-100)
                                            + Gap Analysis
                                                    │
                                         [llm_service.py]
                                                    │
                                      Mistral-7B → Suggestions
                                      Llama-2+LoRA → Cover Letter
```

### Scoring Weights

| Component        | Weight | Method                                                         |
| ---------------- | ------ | -------------------------------------------------------------- |
| Hard Skill Match | 50%    | Set intersection between skills lists                          |
| Semantic Match   | 40%    | Cosine similarity via `sentence-transformers/all-MiniLM-L6-v2` |
| Education Match  | 10%    | Degree level comparison and heuristics                         |

---

## 🔌 API Endpoints

| Method | Endpoint                                      | Auth | Description                          |
| ------ | --------------------------------------------- | ---- | ------------------------------------ |
| POST   | `/api/v1/auth/register`                       | No   | Create a new user account            |
| POST   | `/api/v1/auth/login`                          | No   | Get a JWT access token               |
| GET    | `/api/v1/auth/me`                             | Yes  | Get current user profile             |
| POST   | `/api/v1/jobs/analyze`                        | Yes  | Submit resume + URL, starts pipeline |
| GET    | `/api/v1/jobs/{job_id}/status`                | Yes  | Poll for analysis results            |
| POST   | `/api/v1/jobs/{job_id}/generate-cover-letter` | Yes  | Start cover letter generation        |
| GET    | `/api/v1/jobs/{job_id}/cover-letter/status`   | Yes  | Poll for cover letter                |
| GET    | `/health`                                     | No   | Health check                         |

---

## 🧪 Running Tests

```bash
cd backend
pytest                        # Run all tests
pytest tests/test_scorer.py  # Run only scoring tests
pytest -m unit                # Run only unit tests (fast)
pytest --cov=app              # With coverage report
```

---

## 📝 Logging

Structured JSON logs are written to:

- `logs/app.log` – All log levels (rotated at 10MB, 5 backups)
- `logs/errors.log` – ERROR level only

Each log entry includes `user_id`, `job_id`, `stage`, and timing information for easy querying in Datadog/CloudWatch.

---

## 🔒 Security Notes

- Passwords are hashed with **bcrypt** via `passlib`
- JWTs signed with **HS256** (configurable secret key)
- File uploads validated for MIME type (PDF only) and size (10MB max)
- S3 uploads use **server-side AES256 encryption**
- CORS configured to only allow the frontend origin
- Non-root Docker user for container security
