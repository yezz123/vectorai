"""Pydantic models for the Vector Database REST API."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# Enums
class IndexType(str, Enum):
    LINEAR = "linear"
    KDTREE = "kdtree"
    LSH = "lsh"


class DemoStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Base Models
class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., description="The text content of the chunk")
    embedding: list[float] = Field(..., description="Vector embedding of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        """Convert float timestamps to datetime objects."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v)
        return v


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Name of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunks: list[Chunk] = Field(default_factory=list, description="List of chunks in the document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        """Convert float timestamps to datetime objects."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v)
        return v


class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Name of the library")
    description: str = Field(..., description="Description of the library")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Library metadata")
    documents: list[Document] = Field(default_factory=list, description="List of documents in the library")
    index_type: IndexType | None = Field(None, description="Type of index built for the library")
    index_built_at: datetime | None = Field(None, description="When the index was last built")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("created_at", "updated_at", "index_built_at", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        """Convert float timestamps to datetime objects."""
        if v is not None and isinstance(v, int | float):
            return datetime.fromtimestamp(v)
        return v


# Request Models
class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=1000)
    metadata: dict[str, Any] | None = Field(default_factory=dict)


class UpdateLibraryRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, min_length=1, max_length=1000)
    metadata: dict[str, Any] | None = None


class CreateDocumentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    metadata: dict[str, Any] | None = Field(default_factory=dict)


class UpdateDocumentRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    metadata: dict[str, Any] | None = None


class CreateChunkRequest(BaseModel):
    text: str = Field(..., min_length=1)
    embedding: list[float] = Field(..., description="Vector embedding")
    metadata: dict[str, Any] | None = Field(default_factory=dict)


class UpdateChunkRequest(BaseModel):
    text: str | None = Field(None, min_length=1)
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class SearchQuery(BaseModel):
    query_embedding: list[float] = Field(..., description="Query vector embedding")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    filters: dict[str, Any] | None = Field(default_factory=dict, description="Metadata filters")


class BuildIndexRequest(BaseModel):
    index_type: IndexType = Field(..., description="Type of index to build")
    num_hashes: int | None = Field(None, ge=1, le=1000, description="Number of hash functions for LSH")
    num_buckets: int | None = Field(None, ge=1, le=10000, description="Number of hash buckets for LSH")


# Demo Models
class DemoRequest(BaseModel):
    library_name: str = Field(default="Technical Documentation Demo", description="Name for the demo library")
    library_description: str = Field(
        default="A collection of technical articles for demonstration", description="Description for the demo library"
    )
    use_cohere: bool = Field(default=True, description="Whether to use Cohere API for embeddings")
    cohere_api_key: str | None = Field(None, description="Cohere API key (optional if set in environment)")


class DemoResponse(BaseModel):
    demo_id: str = Field(..., description="Unique identifier for the demo")
    status: DemoStatus = Field(..., description="Current status of the demo")
    message: str = Field(..., description="Status message")
    library_id: str | None = Field(None, description="ID of the created library")
    estimated_duration: str | None = Field(None, description="Estimated time to completion")
    started_at: datetime | None = Field(None, description="When the demo started")
    completed_at: datetime | None = Field(None, description="When the demo completed")
    progress: dict[str, Any] | None = Field(None, description="Current progress information")
    api_docs_url: str | None = Field(None, description="URL to API documentation")
    health_check_url: str | None = Field(None, description="URL to health check endpoint")


class DemoProgress(BaseModel):
    step: str = Field(..., description="Current step being executed")
    completed_steps: list[str] = Field(default_factory=list, description="Steps that have been completed")
    total_steps: int = Field(..., description="Total number of steps")
    current_document: str | None = Field(None, description="Current document being processed")
    current_index: str | None = Field(None, description="Current index being built")
    current_search: str | None = Field(None, description="Current search being performed")


# Response Models
class SearchResult(BaseModel):
    chunks: list[Chunk] = Field(..., description="List of similar chunks")
    scores: list[float] = Field(..., description="Similarity scores for each chunk")
    total_found: int = Field(..., description="Total number of chunks found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    index_type: str = Field(..., description="Type of index used for search")


class LibraryStats(BaseModel):
    id: str = Field(..., description="Library ID")
    name: str = Field(..., description="Library name")
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    index_info: dict[str, Any] | None = Field(None, description="Information about the built index")
    created_at: datetime = Field(..., description="When the library was created")
    last_updated: datetime = Field(..., description="When the library was last updated")


class SystemStats(BaseModel):
    total_libraries: int = Field(..., description="Total number of libraries")
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    active_indexes: int = Field(..., description="Number of active indexes")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    last_backup: datetime | None = Field(None, description="Last backup timestamp")


# Error Models
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = Field(None, description="Request identifier for tracking")


class ValidationError(BaseModel):
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Value that failed validation")


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationError] = Field(..., description="List of validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = Field(None, description="Request identifier for tracking")
