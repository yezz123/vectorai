"""Document service for business logic operations.

Handles document creation, updates, deletion, and chunk management.
"""

from typing import Any

from app.models.base import Chunk, CreateDocumentRequest, Document, UpdateDocumentRequest
from app.repositories.base import DocumentRepository, LibraryRepository


class DocumentService:
    """Service for document business logic operations."""

    def __init__(self, document_repository: DocumentRepository, library_repository: LibraryRepository):
        self.document_repository = document_repository
        self.library_repository = library_repository

    def create_document(self, library_id: str, request: CreateDocumentRequest) -> Document | None:
        """Create a new document in a library with validation."""
        # Validate library exists
        library = self.library_repository.get_by_id(library_id)
        if not library:
            raise ValueError(f"Library with ID {library_id} not found")

        # Validate request
        if not request.name or not request.name.strip():
            raise ValueError("Document name cannot be empty")

        # Check for duplicate names in the same library
        existing_docs = self.document_repository.get_all(library_id)
        for doc in existing_docs:
            if doc.name == request.name.strip():
                raise ValueError(f"Document with name '{request.name}' already exists in this library")

        # Create document entity
        document = Document(name=request.name.strip(), metadata=request.metadata or {})

        # Save to repository
        return self.document_repository.create(document, library_id)

    def get_document(self, library_id: str, document_id: str) -> Document | None:
        """Get a document by ID from a library."""
        return self.document_repository.get_by_id(document_id, library_id)

    def get_all_documents(self, library_id: str) -> list[Document]:
        """Get all documents from a library."""
        return self.document_repository.get_all(library_id)

    def update_document(self, library_id: str, document_id: str, request: UpdateDocumentRequest) -> Document | None:
        """Update a document with validation."""
        # Validate document exists
        document = self.get_document(library_id, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

        # Validate request
        updates = {}

        if request.name is not None:
            if not request.name.strip():
                raise ValueError("Document name cannot be empty")

            # Check for duplicate names (excluding current document)
            existing_docs = self.document_repository.get_all(library_id)
            for doc in existing_docs:
                if doc.id != document_id and doc.name == request.name.strip():
                    raise ValueError(f"Document with name '{request.name}' already exists in this library")

            updates["name"] = request.name.strip()

        if request.metadata is not None:
            updates["metadata"] = request.metadata

        if not updates:
            raise ValueError("No valid updates provided")

        # Update in repository
        return self.document_repository.update(document_id, library_id, updates)

    def delete_document(self, library_id: str, document_id: str) -> bool:
        """Delete a document from a library."""
        return self.document_repository.delete(document_id, library_id)

    def add_chunks_to_document(self, library_id: str, document_id: str, chunks: list[Chunk]) -> bool:
        """Add chunks to a document with validation."""
        # Validate document exists
        document = self.get_document(library_id, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found in library {library_id}")

        # Validate chunks
        if not chunks:
            raise ValueError("No chunks provided")

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not chunk.embedding:
                raise ValueError("Chunk embedding cannot be empty")

            if not isinstance(chunk.embedding, list) or not all(isinstance(x, int | float) for x in chunk.embedding):
                raise ValueError("Chunk embedding must be a list of numbers")

        # Add chunks to document
        return self.document_repository.add_chunks(document_id, library_id, chunks)

    def get_document_stats(self, library_id: str, document_id: str) -> dict[str, Any] | None:
        """Get statistics for a specific document."""
        if document := self.get_document(library_id, document_id):
            return {
                "document_id": document_id,
                "library_id": library_id,
                "name": document.name,
                "total_chunks": len(document.chunks),
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "metadata": document.metadata,
            }
        else:
            return None

    def search_documents_by_name(self, library_id: str, name_query: str) -> list[Document]:
        """Search documents by name (case-insensitive partial match)."""
        documents = self.get_all_documents(library_id)
        name_query_lower = name_query.lower()

        return [doc for doc in documents if name_query_lower in doc.name.lower()]

    def get_documents_by_metadata(self, library_id: str, metadata_filters: dict[str, Any]) -> list[Document]:
        """Get documents filtered by metadata."""
        documents = self.get_all_documents(library_id)
        filtered_documents = []

        for doc in documents:
            matches_all_filters = not any(
                key not in doc.metadata or doc.metadata[key] != value for key, value in metadata_filters.items()
            )
            if matches_all_filters:
                filtered_documents.append(doc)

        return filtered_documents
