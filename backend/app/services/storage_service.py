"""
Storage Service
===============
Abstracts file storage operations.

Two modes (controlled by settings.USE_LOCAL_STORAGE):
  - Local: files stored in settings.LOCAL_UPLOAD_DIR directory.
  - S3: files stored in AWS S3 bucket (production mode).

The public interface (save_file, get_local_file_path) is the same
regardless of mode, so the callers never need to think about the backend.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Local Storage ─────────────────────────────────────────────────────────────

def _ensure_upload_dir() -> Path:
    """Create the local upload directory if it doesn't exist."""
    upload_dir = Path(settings.LOCAL_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _save_local(file: UploadFile, storage_key: str) -> str:
    """Save an uploaded file to the local filesystem."""
    upload_dir = _ensure_upload_dir()
    file_path = upload_dir / storage_key

    # The storage key contains nested subdirs (resumes/{user_id}/filename).
    # Ensure the full directory tree exists before writing.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("file_saved_local", key=storage_key, size_bytes=len(content))
    return storage_key


def get_local_file_path(storage_key: str) -> str:
    """Resolve a storage key to an absolute local file path."""
    path = Path(settings.LOCAL_UPLOAD_DIR) / storage_key
    if not path.exists():
        raise FileNotFoundError(
            f"File with storage key '{storage_key}' not found at '{path}'."
        )
    return str(path.resolve())


# ── S3 Storage ────────────────────────────────────────────────────────────────

def _save_s3(file: UploadFile, storage_key: str) -> str:
    """Upload a file to AWS S3."""
    import boto3

    s3_client = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    content = file.file.read()
    s3_client.put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=storage_key,
        Body=content,
        ContentType="application/pdf",
        ServerSideEncryption="AES256",  # encrypt at rest
    )

    logger.info(
        "file_saved_s3",
        bucket=settings.S3_BUCKET_NAME,
        key=storage_key,
        size_bytes=len(content),
    )
    return storage_key


def _download_from_s3(storage_key: str, local_path: str) -> str:
    """Download a file from S3 to a local temp path for processing."""
    import boto3

    s3_client = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    s3_client.download_file(settings.S3_BUCKET_NAME, storage_key, local_path)
    logger.info("file_downloaded_from_s3", key=storage_key, local_path=local_path)
    return local_path


# ── Public Interface ──────────────────────────────────────────────────────────

def generate_storage_key(user_id: str, original_filename: str) -> str:
    """
    Generate a unique, predictable storage key for a file.
    Format: resumes/{user_id}/{uuid4}_{original_filename}
    """
    safe_filename = Path(original_filename).name  # strip directory traversal
    unique_id = uuid.uuid4().hex[:8]
    return f"resumes/{user_id}/{unique_id}_{safe_filename}"


async def save_uploaded_file(file: UploadFile, storage_key: str) -> str:
    """
    Save an uploaded file using the configured storage backend.

    Args:
        file: FastAPI UploadFile object.
        storage_key: The key under which to store the file.

    Returns:
        The storage key (for storing in the database).
    """
    if settings.USE_LOCAL_STORAGE:
        return _save_local(file, storage_key)
    else:
        return _save_s3(file, storage_key)


def delete_file(storage_key: str) -> None:
    """Delete a file from storage (used in account deletion flow)."""
    if settings.USE_LOCAL_STORAGE:
        path = Path(settings.LOCAL_UPLOAD_DIR) / storage_key
        if path.exists():
            path.unlink()
            logger.info("file_deleted_local", key=storage_key)
    else:
        import boto3
        s3 = boto3.client("s3", region_name=settings.AWS_REGION)
        s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=storage_key)
        logger.info("file_deleted_s3", key=storage_key)
