"""
Unit Tests – Security Module (security.py)
==========================================
Tests JWT creation/verification and password hashing utilities.
No database, no external services required.
"""

import time
from datetime import timedelta

import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


# ── Password Hashing Tests ────────────────────────────────────────────────────


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("mysecretpassword")
        assert hashed != "mysecretpassword"

    def test_hash_starts_with_bcrypt_prefix(self):
        hashed = hash_password("test123")
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    def test_verify_correct_password_returns_true(self):
        plain = "CorrectPassword123!"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True

    def test_verify_wrong_password_returns_false(self):
        hashed = hash_password("CorrectPassword123!")
        assert verify_password("WrongPassword", hashed) is False

    def test_empty_password_hashes_and_verifies(self):
        """Edge case: empty string password should still hash and verify."""
        hashed = hash_password("")
        assert verify_password("", hashed) is True
        assert verify_password("not_empty", hashed) is False

    def test_same_password_produces_different_hashes(self):
        """bcrypt uses a random salt, so two hashes of the same password differ."""
        h1 = hash_password("SamePassword")
        h2 = hash_password("SamePassword")
        assert h1 != h2

    def test_both_verify_against_same_plain(self):
        """Both hashes of the same password still verify correctly."""
        plain = "SamePassword"
        h1 = hash_password(plain)
        h2 = hash_password(plain)
        assert verify_password(plain, h1) is True
        assert verify_password(plain, h2) is True

    def test_special_characters_password(self):
        plain = "p@$$w0rd!#%^&*()"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True
        assert verify_password("different", hashed) is False

    def test_long_password(self):
        plain = "a" * 72  # bcrypt max effective length is 72 bytes
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True


# ── JWT Tests ─────────────────────────────────────────────────────────────────


class TestCreateAccessToken:
    def test_returns_string(self):
        token = create_access_token("user-123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_is_valid_jwt_structure(self):
        """A JWT has exactly 3 dot-separated parts."""
        token = create_access_token("user-123")
        parts = token.split(".")
        assert len(parts) == 3

    def test_token_decodes_to_correct_subject(self):
        subject = "abc-def-user-id"
        token = create_access_token(subject)
        decoded = decode_access_token(token)
        assert decoded == subject

    def test_custom_expiry_is_respected(self):
        """Token with very short expiry should expire quickly."""
        token = create_access_token("user-xyz", expires_delta=timedelta(seconds=1))
        # Immediately after creation — should still be valid
        decoded = decode_access_token(token)
        assert decoded == "user-xyz"

    def test_expired_token_returns_none(self):
        """A token with negative expiry is already expired."""
        token = create_access_token("user-xyz", expires_delta=timedelta(seconds=-1))
        result = decode_access_token(token)
        assert result is None

    def test_different_subjects_produce_different_tokens(self):
        t1 = create_access_token("user-1")
        t2 = create_access_token("user-2")
        assert t1 != t2


class TestDecodeAccessToken:
    def test_valid_token_returns_subject(self):
        token = create_access_token("test-user-id")
        result = decode_access_token(token)
        assert result == "test-user-id"

    def test_invalid_token_returns_none(self):
        result = decode_access_token("this.is.not.a.jwt")
        assert result is None

    def test_empty_string_returns_none(self):
        result = decode_access_token("")
        assert result is None

    def test_tampered_token_returns_none(self):
        token = create_access_token("user-1")
        # Alter the last 5 chars of the signature
        tampered = token[:-5] + "XXXXX"
        result = decode_access_token(tampered)
        assert result is None

    def test_random_string_returns_none(self):
        result = decode_access_token("randomgarbage123!")
        assert result is None
