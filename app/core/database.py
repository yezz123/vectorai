"""Thread-safe in-memory database implementation with concurrency control.

Uses locks and atomic operations to prevent data races between reads and writes.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any

from app.core.indexing import BaseIndex, IndexFactory
from app.models.base import Chunk, Document, Library

# Get logger for this module
logger = logging.getLogger(__name__)


class ThreadSafeDatabase:
    """Thread-safe in-memory database with concurrency control.

    Design Choices:
    1. **Read-Write Lock (RWMutex)**: Allows multiple concurrent reads but exclusive writes
    2. **Atomic Operations**: Critical sections are protected by locks
    3. **In-Memory Storage**: Fast access with optional disk persistence
    4. **Index Management**: Separate indexes for each library with their own locks

    Concurrency Strategy:
    - Multiple readers can access data simultaneously
    - Writers get exclusive access (blocking all other operations)
    - Each library has its own index and lock for better performance
    - Bulk operations are atomic to maintain consistency
    """

    def __init__(self, persistence_path: str | None = None):
        self._data: dict[str, Library] = {}
        self._indexes: dict[str, BaseIndex] = {}
        self._rw_lock = threading.RLock()  # Reentrant lock for read-write operations
        self._index_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._persistence_path = persistence_path

        # Load data from disk if persistence is enabled
        if self._persistence_path:
            self._load_from_disk()

    def _acquire_read_lock(self):
        """Acquire read lock - allows multiple concurrent readers."""
        self._rw_lock.acquire()

    def _release_read_lock(self):
        """Release read lock."""
        self._rw_lock.release()

    def _acquire_write_lock(self):
        """Acquire write lock - exclusive access for writers."""
        self._rw_lock.acquire()

    def _release_write_lock(self):
        """Release write lock."""
        self._rw_lock.release()

    def _acquire_index_lock(self, library_id: str):
        """Acquire lock for a specific library's index."""
        self._index_locks[library_id].acquire()

    def _release_index_lock(self, library_id: str):
        """Release lock for a specific library's index."""
        self._index_locks[library_id].release()

    # Library CRUD operations
    def create_library(self, library: Library) -> Library:
        """Create a new library atomically."""
        try:
            self._acquire_write_lock()

            if library.id in self._data:
                raise ValueError(f"Library with ID {library.id} already exists")

            # Create index for the library
            index = IndexFactory.create_index("linear")  # Default to linear search
            self._indexes[library.id] = index
            self._data[library.id] = library

            # Persist to disk if enabled
            if self._persistence_path:
                self._persist_to_disk()

            return library

        finally:
            self._release_write_lock()

    def get_library(self, library_id: str) -> Library | None:
        """Get a library by ID (read operation)."""
        try:
            self._acquire_read_lock()
            return self._data.get(library_id)
        finally:
            self._release_read_lock()

    def get_all_libraries(self) -> list[Library]:
        """Get all libraries (read operation)."""
        try:
            self._acquire_read_lock()
            return list(self._data.values())
        finally:
            self._release_read_lock()

    def update_library(self, library_id: str, updates: dict[str, Any]) -> Library | None:
        """Update a library atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return None

            library = self._data[library_id]

            # Update fields
            for field, value in updates.items():
                if hasattr(library, field) and field not in ["id", "created_at"]:
                    setattr(library, field, value)

            library.updated_at = time.time()
            self._data[library_id] = library

            # Persist to disk if enabled
            if self._persistence_path:
                self._persist_to_disk()

            return library

        finally:
            self._release_write_lock()

    def delete_library(self, library_id: str) -> bool:
        """Delete a library and its index atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return False

            # Remove library and its index
            del self._data[library_id]
            if library_id in self._indexes:
                del self._indexes[library_id]

            # Persist to disk if enabled
            if self._persistence_path:
                self._persist_to_disk()

            return True

        finally:
            self._release_write_lock()

    # Document CRUD operations
    def create_document(self, library_id: str, document: Document) -> Document | None:
        """Create a document in a library atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return None

            library = self._data[library_id]
            library.documents.append(document)
            library.updated_at = time.time()

            # Update index with new chunks
            if library_id in self._indexes:
                self._acquire_index_lock(library_id)
                try:
                    index = self._indexes[library_id]
                    index.add_chunks(document.chunks)
                    index.build()
                finally:
                    self._release_index_lock(library_id)

            # Persist to disk if enabled
            if self._persistence_path:
                self._persist_to_disk()

            return document

        finally:
            self._release_write_lock()

    def get_document(self, library_id: str, document_id: str) -> Document | None:
        """Get a document from a library (read operation)."""
        try:
            self._acquire_read_lock()

            if library_id not in self._data:
                return None

            library = self._data[library_id]
            for doc in library.documents:
                if doc.id == document_id:
                    return doc

            return None

        finally:
            self._release_read_lock()

    def update_document(self, library_id: str, document_id: str, updates: dict[str, Any]) -> Document | None:
        """Update a document atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return None

            library = self._data[library_id]
            for i, doc in enumerate(library.documents):
                if doc.id == document_id:
                    # Update fields
                    for field, value in updates.items():
                        if hasattr(doc, field) and field not in ["id", "created_at"]:
                            setattr(doc, field, value)

                    doc.updated_at = time.time()
                    library.documents[i] = doc
                    library.updated_at = time.time()

                    # Rebuild index if chunks were modified
                    if "chunks" in updates:
                        self._acquire_index_lock(library_id)
                        try:
                            index = self._indexes[library_id]
                            index.chunks = []
                            index.embeddings = []
                            for document in library.documents:
                                index.add_chunks(document.chunks)
                            index.build()
                        finally:
                            self._release_index_lock(library_id)

                    # Persist to disk if enabled
                    if self._persistence_path:
                        self._persist_to_disk()

                    return doc

            return None

        finally:
            self._release_write_lock()

    def delete_document(self, library_id: str, document_id: str) -> bool:
        """Delete a document from a library atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return False

            library = self._data[library_id]
            for i, doc in enumerate(library.documents):
                if doc.id == document_id:
                    del library.documents[i]
                    library.updated_at = time.time()

                    # Rebuild index
                    if library_id in self._indexes:
                        self._acquire_index_lock(library_id)
                        try:
                            index = self._indexes[library_id]
                            index.chunks = []
                            index.embeddings = []
                            for document in library.documents:
                                index.add_chunks(document.chunks)
                            index.build()
                        finally:
                            self._release_index_lock(library_id)

                    # Persist to disk if enabled
                    if self._persistence_path:
                        self._persist_to_disk()

                    return True

            return False

        finally:
            self._release_write_lock()

    # Chunk operations
    def add_chunks_to_document(self, library_id: str, document_id: str, chunks: list[Chunk]) -> bool:
        """Add chunks to a document atomically."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return False

            library = self._data[library_id]
            for doc in library.documents:
                if doc.id == document_id:
                    doc.chunks.extend(chunks)
                    doc.updated_at = time.time()
                    library.updated_at = time.time()

                    # Update index
                    if library_id in self._indexes:
                        self._acquire_index_lock(library_id)
                        try:
                            index = self._indexes[library_id]
                            index.add_chunks(chunks)
                            index.build()
                        finally:
                            self._release_index_lock(library_id)

                    # Persist to disk if enabled
                    if self._persistence_path:
                        self._persist_to_disk()

                    return True

            return False

        finally:
            self._release_write_lock()

    # Indexing operations
    def build_index(self, library_id: str, index_type: str = "linear", **kwargs) -> bool:
        """Build or rebuild an index for a library."""
        try:
            self._acquire_write_lock()

            if library_id not in self._data:
                return False

            library = self._data[library_id]

            # Create new index
            index = IndexFactory.create_index(index_type, **kwargs)

            # Add all chunks from all documents
            all_chunks = []
            for doc in library.documents:
                all_chunks.extend(doc.chunks)

            index.add_chunks(all_chunks)
            index.build()

            # Replace old index
            self._indexes[library_id] = index

            # Persist to disk if enabled
            if self._persistence_path:
                self._persist_to_disk()

            return True

        finally:
            self._release_write_lock()

    def search(self, library_id: str, query_embedding: list[float], k: int = 5) -> tuple[list[Chunk], list[float]]:
        """Search for similar chunks in a library."""
        try:
            self._acquire_read_lock()

            if library_id not in self._indexes:
                return [], []

            # Acquire index-specific lock for search
            self._acquire_index_lock(library_id)
            try:
                index = self._indexes[library_id]
                return index.search(query_embedding, k)
            finally:
                self._release_index_lock(library_id)

        finally:
            self._release_read_lock()

    def get_index_info(self, library_id: str) -> dict[str, Any] | None:
        """Get information about a library's index."""
        try:
            self._acquire_read_lock()

            if library_id not in self._indexes:
                return None

            index = self._indexes[library_id]
            return {
                "type": type(index).__name__,
                "is_built": index.is_built,
                "num_chunks": len(index.chunks),
                "num_embeddings": len(index.embeddings),
            }

        finally:
            self._release_read_lock()

    # Persistence methods
    def _persist_to_disk(self):
        """Persist database state to disk."""
        if not self._persistence_path:
            return

        try:
            # Convert data to serializable format
            serializable_data = {}
            for lib_id, library in self._data.items():
                serializable_data[lib_id] = library.model_dump()

            # Save to file
            with open(self._persistence_path, "w") as f:
                json.dump(serializable_data, f, indent=2, default=str)

        except Exception as e:
            # Log error but don't fail the operation
            logger.error(f"Failed to persist database: {e}")

    def _load_from_disk(self):
        """Load database state from disk."""
        if not self._persistence_path or not os.path.exists(self._persistence_path):
            return

        try:
            with open(self._persistence_path) as f:
                data = json.load(f)

            # Reconstruct objects
            for lib_id, lib_data in data.items():
                try:
                    library = Library(**lib_data)
                    self._data[lib_id] = library

                    # Create default index
                    index = IndexFactory.create_index("linear")
                    self._indexes[lib_id] = index

                    # Add chunks to index
                    for doc in library.documents:
                        index.add_chunks(doc.chunks)

                    index.build()
                except Exception as lib_error:
                    logger.error(f"Failed to load library {lib_id}: {lib_error}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load database from disk: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            self._acquire_read_lock()

            total_libraries = len(self._data)
            total_documents = sum(len(lib.documents) for lib in self._data.values())
            total_chunks = sum(len(doc.chunks) for lib in self._data.values() for doc in lib.documents)

            return {
                "total_libraries": total_libraries,
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "indexed_libraries": len(self._indexes),
                "persistence_enabled": self._persistence_path is not None,
            }

        finally:
            self._release_read_lock()
