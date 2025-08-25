"""Library service for business logic operations.

Happens library creation, updates, deletion, and indexing operations.
"""

from typing import Any

from app.core.indexing import IndexFactory
from app.models.base import CreateLibraryRequest, Library, UpdateLibraryRequest
from app.repositories.base import LibraryRepository


class LibraryService:
    """Service for library business logic operations."""

    def __init__(self, library_repository: LibraryRepository):
        self.library_repository = library_repository

    def create_library(self, request: CreateLibraryRequest) -> Library:
        """Create a new library with validation."""
        # Validate request
        if not request.name or not request.name.strip():
            raise ValueError("Library name cannot be empty")

        # Create library entity
        library = Library(name=request.name.strip(), description=request.description, metadata=request.metadata or {})

        # Save to repository
        return self.library_repository.create(library)

    def get_library(self, library_id: str) -> Library | None:
        """Get a library by ID."""
        return self.library_repository.get_by_id(library_id)

    def get_all_libraries(self) -> list[Library]:
        """Get all libraries."""
        return self.library_repository.get_all()

    def update_library(self, library_id: str, request: UpdateLibraryRequest) -> Library | None:
        """Update a library with validation."""
        # Validate request
        updates = {}

        if request.name is not None:
            if not request.name.strip():
                raise ValueError("Library name cannot be empty")
            updates["name"] = request.name.strip()

        if request.description is not None:
            updates["description"] = request.description

        if request.metadata is not None:
            updates["metadata"] = request.metadata

        if not updates:
            raise ValueError("No valid updates provided")

        # Update in repository
        return self.library_repository.update(library_id, updates)

    def delete_library(self, library_id: str) -> bool:
        """Delete a library."""
        return self.library_repository.delete(library_id)

    def build_index(self, library_id: str, index_type: str = "linear", **kwargs) -> bool:
        """Build or rebuild an index for a library."""
        # Validate index type
        valid_types = ["linear", "kdtree", "lsh"]
        if index_type not in valid_types:
            raise ValueError(f"Invalid index type. Must be one of: {valid_types}")

        # Validate LSH parameters
        if index_type == "lsh":
            num_hashes = kwargs.get("num_hashes", 10)
            num_buckets = kwargs.get("num_buckets", 100)

            if num_hashes <= 0 or num_buckets <= 0:
                raise ValueError("LSH parameters must be positive integers")

        # Build index
        return self.library_repository.build_index(library_id, index_type, **kwargs)

    def get_index_info(self, library_id: str) -> dict[str, Any] | None:
        """Get information about a library's index."""
        return self.library_repository.get_index_info(library_id)

    def get_available_index_types(self) -> dict[str, dict[str, Any]]:
        """Get information about available index types."""
        return {
            "linear": IndexFactory.get_index_info("linear"),
            "kdtree": IndexFactory.get_index_info("kdtree"),
            "lsh": IndexFactory.get_index_info("lsh"),
        }

    def get_library_stats(self, library_id: str) -> dict[str, Any] | None:
        """Get statistics for a specific library."""
        library = self.get_library(library_id)
        if not library:
            return None

        index_info = self.get_index_info(library_id)

        return {
            "library_id": library_id,
            "name": library.name,
            "description": library.description,
            "total_documents": len(library.documents),
            "total_chunks": sum(len(doc.chunks) for doc in library.documents),
            "created_at": library.created_at,
            "updated_at": library.updated_at,
            "index_info": index_info,
            "metadata": library.metadata,
        }
