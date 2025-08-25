"""Dependency injection system for FastAPI.

Provides services and repositories to API endpoints.
"""

import os

from app.core.config import get_settings
from app.core.database import ThreadSafeDatabase
from app.core.logging_config import setup_default_logging  # Ensure logging is configured
from app.repositories.base import ChunkRepository, DocumentRepository, LibraryRepository
from app.services.demo_service import DemoService
from app.services.document_service import DocumentService
from app.services.library_service import LibraryService
from app.services.search_service import SearchService

# Ensure logging is configured
setup_default_logging()

# Global database instance
_database = None


def get_database() -> ThreadSafeDatabase:
    """Get the global database instance."""
    global _database
    if _database is None:
        settings = get_settings()
        # Enable persistence by default
        persistence_path = settings.persistence_path
        os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
        _database = ThreadSafeDatabase(persistence_path=persistence_path)
    return _database


def get_library_repository() -> LibraryRepository:
    """Get library repository instance."""
    database = get_database()
    return LibraryRepository(database)


def get_document_repository() -> DocumentRepository:
    """Get document repository instance."""
    database = get_database()
    return DocumentRepository(database)


def get_chunk_repository() -> ChunkRepository:
    """Get chunk repository instance."""
    database = get_database()
    return ChunkRepository(database)


def get_library_service() -> LibraryService:
    """Get library service instance."""
    library_repo = get_library_repository()
    return LibraryService(library_repo)


def get_document_service() -> DocumentService:
    """Get document service instance."""
    document_repo = get_document_repository()
    library_repo = get_library_repository()
    return DocumentService(document_repo, library_repo)


def get_search_service() -> SearchService:
    """Get search service instance."""
    chunk_repo = get_chunk_repository()
    library_repo = get_library_repository()
    return SearchService(chunk_repo, library_repo)


def get_demo_service() -> DemoService:
    """Get demo service instance."""
    return DemoService()
