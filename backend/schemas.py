"""Request/response models for the API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.config import DEFAULT_MODEL, DEFAULT_TOP_K, MODEL_CONFIG


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default=DEFAULT_MODEL)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)
    is_aligned: bool = Field(default=False)
    language: str | None = None
    document_type: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in MODEL_CONFIG:
            raise ValueError(f"model must be one of: {', '.join(MODEL_CONFIG)}")
        return v


class CompareQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model_a: str = Field(default="BGE-M3")
    model_b: str = Field(default="mSBERT")
    is_aligned_a: bool = Field(default=False)
    is_aligned_b: bool = Field(default=False)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("model_a", "model_b")
    @classmethod
    def validate_models(cls, v: str) -> str:
        if v not in MODEL_CONFIG:
            raise ValueError(f"model must be one of: {', '.join(MODEL_CONFIG)}")
        return v


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    query: str
    model: str
    count: int
    detected_language: str


class CompareResponse(BaseModel):
    query: str
    model_a: str
    model_b: str
    is_aligned_a: bool
    is_aligned_b: bool
    detected_language: str
    results_a: list[dict[str, Any]]
    results_b: list[dict[str, Any]]
