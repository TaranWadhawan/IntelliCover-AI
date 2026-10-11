"""
Integration Tests – Auth & Jobs API Endpoints
==============================================
Tests the full HTTP request/response cycle using FastAPI's async test client
and a live PostgreSQL database (must be running via docker-compose).

All tests share a single asyncio event loop (asyncio_default_test_loop_scope =
session in pytest.ini).  Tables are created once per session and each test
starts with all rows deleted, so tests remain fully isolated without needing
SAVEPOINT / nested transactions (which are unreliable with asyncpg).

Run with:
    pytest tests/test_api.py -v --tb=short --no-cov
"""

import io
import uuid
from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.database import Base
from app.main import app  # importing app also registers all ORM models

# ──────────────────────────────────────────────────────────────────────────────
# Test database setup
# ──────────────────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = settings.DATABASE_URL

# Module-level engine is fine here because pytest.ini sets
# asyncio_default_fixture_loop_scope = session, so this engine will always be
# used within the single session-wide event loop.
_test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    """Create all tables once at the start of the test session."""
    from app.models import db_models  # noqa: F401 – ensures models are on Base
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _test_engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_tables(create_tables):
    """Delete all rows before each test to guarantee clean state."""
    async with _test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())


@pytest_asyncio.fixture()
async def client(clean_tables) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client that goes through the real FastAPI app with
    its real database dependency (no mocking).  Each test starts with
    empty tables (ensured by the clean_tables fixture above).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_REGISTER_URL = "/api/v1/auth/register"
_LOGIN_URL = "/api/v1/auth/login"
_ME_URL = "/api/v1/auth/me"
_ANALYZE_URL = "/api/v1/jobs/analyze"


def _unique_email() -> str:
    return f"user_{uuid.uuid4().hex[:8]}@test.com"


async def _register_and_login(client: AsyncClient) -> tuple[str, dict]:
    """Register a new user and return (token, user_data)."""
    email = _unique_email()
    password = "TestPassword123!"
    reg = await client.post(
        _REGISTER_URL,
        json={"email": email, "password": password, "full_name": "Test User"},
    )
    assert reg.status_code == 201, reg.text
    login = await client.post(_LOGIN_URL, json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return token, reg.json()


# ──────────────────────────────────────────────────────────────────────────────
# Auth: POST /register
# ──────────────────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_successful_registration_returns_201(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL,
            json={"email": _unique_email(), "password": "StrongPass1!"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "id" in body
        assert "password" not in body  # password never exposed

    async def test_returned_user_has_correct_email(self, client: AsyncClient):
        email = _unique_email()
        resp = await client.post(
            _REGISTER_URL, json={"email": email, "password": "StrongPass1!"}
        )
        assert resp.json()["email"] == email

    async def test_is_active_true_by_default(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL,
            json={"email": _unique_email(), "password": "StrongPass1!"},
        )
        assert resp.json()["is_active"] is True

    async def test_duplicate_email_returns_409(self, client: AsyncClient):
        email = _unique_email()
        payload = {"email": email, "password": "StrongPass1!"}
        await client.post(_REGISTER_URL, json=payload)
        resp2 = await client.post(_REGISTER_URL, json=payload)
        assert resp2.status_code == 409

    async def test_invalid_email_returns_422(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL,
            json={"email": "not-an-email", "password": "StrongPass1!"},
        )
        assert resp.status_code == 422

    async def test_short_password_returns_422(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL,
            json={"email": _unique_email(), "password": "short"},
        )
        assert resp.status_code == 422

    async def test_missing_email_field_returns_422(self, client: AsyncClient):
        resp = await client.post(_REGISTER_URL, json={"password": "strongpass"})
        assert resp.status_code == 422

    async def test_missing_password_field_returns_422(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL, json={"email": _unique_email()}
        )
        assert resp.status_code == 422

    async def test_full_name_stored_correctly(self, client: AsyncClient):
        resp = await client.post(
            _REGISTER_URL,
            json={
                "email": _unique_email(),
                "password": "StrongPass1!",
                "full_name": "Alice Wonderland",
            },
        )
        assert resp.json()["full_name"] == "Alice Wonderland"


# ──────────────────────────────────────────────────────────────────────────────
# Auth: POST /login
# ──────────────────────────────────────────────────────────────────────────────


class TestLogin:
    async def test_valid_login_returns_token(self, client: AsyncClient):
        email = _unique_email()
        password = "StrongPass1!"
        await client.post(_REGISTER_URL, json={"email": email, "password": password})
        resp = await client.post(_LOGIN_URL, json={"email": email, "password": password})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    async def test_wrong_password_returns_401(self, client: AsyncClient):
        email = _unique_email()
        await client.post(_REGISTER_URL, json={"email": email, "password": "CorrectPass1!"})
        resp = await client.post(
            _LOGIN_URL, json={"email": email, "password": "WrongPassword!"}
        )
        assert resp.status_code == 401

    async def test_nonexistent_user_returns_401(self, client: AsyncClient):
        resp = await client.post(
            _LOGIN_URL,
            json={"email": "ghost@nowhere.com", "password": "anypass123"},
        )
        assert resp.status_code == 401

    async def test_login_returns_expires_in_field(self, client: AsyncClient):
        email = _unique_email()
        password = "TestPass123!"
        await client.post(_REGISTER_URL, json={"email": email, "password": password})
        resp = await client.post(_LOGIN_URL, json={"email": email, "password": password})
        assert "expires_in" in resp.json()
        assert resp.json()["expires_in"] > 0

    async def test_invalid_email_format_returns_422(self, client: AsyncClient):
        resp = await client.post(
            _LOGIN_URL, json={"email": "bad-email", "password": "anypass123"}
        )
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Auth: GET /me
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMe:
    async def test_returns_current_user_profile(self, client: AsyncClient):
        email = _unique_email()
        password = "StrongPass1!"
        reg_resp = await client.post(
            _REGISTER_URL,
            json={"email": email, "password": password, "full_name": "Bob Smith"},
        )
        token, _ = await _register_and_login(client)
        # Use the newly registered user's token via re-login
        login = await client.post(_LOGIN_URL, json={"email": email, "password": password})
        token = login.json()["access_token"]

        me = await client.get(_ME_URL, headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == email

    async def test_no_token_returns_403(self, client: AsyncClient):
        resp = await client.get(_ME_URL)
        # HTTPBearer returns 403 when no Authorization header present
        assert resp.status_code in (401, 403)

    async def test_invalid_token_returns_401(self, client: AsyncClient):
        resp = await client.get(
            _ME_URL, headers={"Authorization": "Bearer invalidtoken"}
        )
        assert resp.status_code == 401

    async def test_malformed_auth_header_returns_error(self, client: AsyncClient):
        resp = await client.get(_ME_URL, headers={"Authorization": "NotBearer token"})
        assert resp.status_code in (401, 403, 422)


# ──────────────────────────────────────────────────────────────────────────────
# Jobs: POST /analyze  (file upload validation only – no Celery)
# ──────────────────────────────────────────────────────────────────────────────


class TestAnalyzeEndpoint:
    async def test_non_pdf_file_returns_422(self, client: AsyncClient):
        token, _ = await _register_and_login(client)
        fake_file = io.BytesIO(b"This is plain text, not a PDF")
        resp = await client.post(
            _ANALYZE_URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"url": "https://example.com/job"},
            files={"resume": ("resume.txt", fake_file, "text/plain")},
        )
        assert resp.status_code == 422

    async def test_unauthenticated_request_returns_403(self, client: AsyncClient):
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = await client.post(
            _ANALYZE_URL,
            data={"url": "https://example.com/job"},
            files={"resume": ("resume.pdf", fake_pdf, "application/pdf")},
        )
        assert resp.status_code in (401, 403)

    async def test_missing_url_field_returns_422(self, client: AsyncClient):
        token, _ = await _register_and_login(client)
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake")
        resp = await client.post(
            _ANALYZE_URL,
            headers={"Authorization": f"Bearer {token}"},
            files={"resume": ("resume.pdf", fake_pdf, "application/pdf")},
        )
        assert resp.status_code == 422

    async def test_missing_resume_field_returns_422(self, client: AsyncClient):
        token, _ = await _register_and_login(client)
        resp = await client.post(
            _ANALYZE_URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"url": "https://example.com/job"},
        )
        assert resp.status_code == 422
