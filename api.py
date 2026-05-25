"""Entry point for uvicorn: ``uvicorn api:app``."""

from backend.app import app

__all__ = ["app"]
