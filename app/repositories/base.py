"""Base repository interface and implementation for data access abstraction.

Follows the Repository pattern to separate business logic from data access.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from app.models.base import Chunk, Document, Library

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository interface."""

    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    def get_by_id(self, entity_id: str) -> T | None:
        """Get entity by ID."""
        pass

    @abstractmethod
    def get_all(self) -> list[T]:
        """Get all entities."""
        pass

    @abstractmethod
    def update(self, entity_id: str, updates: dict[str, Any]) -> T | None:
        """Update an entity."""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        pass


class LibraryRepository(BaseRepository[Library]):
    """Repository for Library entities."""

    def __init__(self, database):
        self.database = database

    def create(self, library: Library) -> Library:
        """Create a new library."""
        return self.database.create_library(library)

    def get_by_id(self, library_id: str) -> Library | None:
        """Get library by ID."""
        return self.database.get_library(library_id)

    def get_all(self) -> list[Library]:
        """Get all libraries."""
        return self.database.get_all_libraries()

    def update(self, library_id: str, updates: dict[str, Any]) -> Library | None:
        """Update a library."""
        return self.database.update_library(library_id, updates)

    def delete(self, library_id: str) -> bool:
        """Delete a library."""
        return self.database.delete_library(library_id)

    def build_index(self, library_id: str, index_type: str = "linear", **kwargs) -> bool:
        """Build or rebuild index for a library."""
        return self.database.build_index(library_id, index_type, **kwargs)

    def get_index_info(self, library_id: str) -> dict[str, Any] | None:
        """Get index information for a library."""
        return self.database.get_index_info(library_id)


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document entities."""

    def __init__(self, database):
        self.database = database

    def create(self, document: Document, library_id: str) -> Document | None:
        """Create a new document in a library."""
        return self.database.create_document(library_id, document)

    def get_by_id(self, document_id: str, library_id: str) -> Document | None:
        """Get document by ID from a library."""
        return self.database.get_document(library_id, document_id)

    def get_all(self, library_id: str) -> list[Document]:
        """Get all documents from a library."""
        library = self.database.get_library(library_id)
        return library.documents if library else []

    def update(self, document_id: str, library_id: str, updates: dict[str, Any]) -> Document | None:
        """Update a document in a library."""
        return self.database.update_document(library_id, document_id, updates)

    def delete(self, document_id: str, library_id: str) -> bool:
        """Delete a document from a library."""
        return self.database.delete_document(library_id, document_id)

    def add_chunks(self, document_id: str, library_id: str, chunks: list[Chunk]) -> bool:
        """Add chunks to a document."""
        return self.database.add_chunks_to_document(library_id, document_id, chunks)


class ChunkRepository:
    """Repository for Chunk operations."""

    def __init__(self, database):
        self.database = database

    def search(self, library_id: str, query_embedding: list[float], k: int = 5) -> tuple[list[Chunk], list[float]]:
        """Search for similar chunks in a library."""
        return self.database.search(library_id, query_embedding, k)
