"""Search service for vector similarity search operations.

Handles k-NN search with metadata filtering capabilities.
"""

import logging
from typing import Any

from app.models.base import Chunk, SearchQuery, SearchResult
from app.repositories.base import ChunkRepository, LibraryRepository

# Get logger for this module
logger = logging.getLogger(__name__)


class SearchService:
    """Service for vector similarity search operations."""

    def __init__(self, chunk_repository: ChunkRepository, library_repository: LibraryRepository):
        self.chunk_repository = chunk_repository
        self.library_repository = library_repository

    def search_similar_chunks(self, library_id: str, query: SearchQuery) -> SearchResult:
        """Search for similar chunks in a library using vector similarity."""
        # Validate library exists
        library = self.library_repository.get_by_id(library_id)
        if not library:
            raise ValueError(f"Library with ID {library_id} not found")

        # Validate query
        if not query.query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if not isinstance(query.query_embedding, list) or not all(
            isinstance(x, int | float) for x in query.query_embedding
        ):
            raise ValueError("Query embedding must be a list of numbers")

        if query.k <= 0 or query.k > 100:
            raise ValueError("k must be between 1 and 100")

        # Perform vector search
        import time

        start_time = time.time()
        chunks, scores = self.chunk_repository.search(library_id, query.query_embedding, query.k)
        search_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Apply metadata filters if provided
        if query.filters:
            chunks, scores = self._apply_metadata_filters(chunks, scores, query.filters)

        # Get index type from library
        library = self.library_repository.get_by_id(library_id)
        index_type = library.index_type.value if library and library.index_type else "unknown"

        return SearchResult(
            chunks=chunks, scores=scores, total_found=len(chunks), search_time_ms=search_time_ms, index_type=index_type
        )

    def _apply_metadata_filters(
        self, chunks: list[Chunk], scores: list[float], filters: dict[str, Any]
    ) -> tuple[list[Chunk], list[float]]:
        """Apply metadata filters to search results."""
        filtered_chunks = []
        filtered_scores = []

        for chunk, score in zip(chunks, scores, strict=False):
            if self._chunk_matches_filters(chunk, filters):
                filtered_chunks.append(chunk)
                filtered_scores.append(score)

        return filtered_chunks, filtered_scores

    def _chunk_matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check if a chunk matches the given metadata filters."""
        for key, value in filters.items():
            if key not in chunk.metadata:
                return False

            chunk_value = chunk.metadata[key]

            # Handle different filter types
            if isinstance(value, dict) and "operator" in value:
                # Advanced filtering with operators
                if not self._evaluate_filter_operator(chunk_value, value):
                    return False
            else:
                # Simple equality check
                if chunk_value != value:
                    return False

        return True

    def _evaluate_filter_operator(self, chunk_value: Any, filter_spec: dict[str, Any]) -> bool:
        """Evaluate filter with operators like gt, lt, contains, etc."""
        operator = filter_spec.get("operator")
        filter_value = filter_spec.get("value")

        if operator == "gt":
            return chunk_value > filter_value
        elif operator == "gte":
            return chunk_value >= filter_value
        elif operator == "lt":
            return chunk_value < filter_value
        elif operator == "lte":
            return chunk_value <= filter_value
        elif operator == "contains":
            if isinstance(chunk_value, str) and isinstance(filter_value, str):
                return filter_value.lower() in chunk_value.lower()
            return False
        elif operator == "in":
            return chunk_value in filter_value
        elif operator == "not_in":
            return chunk_value not in filter_value
        elif operator == "regex":
            import re

            try:
                return bool(re.search(filter_value, str(chunk_value)))
            except re.error:
                return False
        else:
            # Unknown operator, default to equality
            return chunk_value == filter_value

    def search_across_libraries(
        self, query: SearchQuery, library_ids: list[str] | None = None
    ) -> dict[str, SearchResult]:
        """Search for similar chunks across multiple libraries."""
        results = {}

        # If no specific libraries provided, search all
        if not library_ids:
            libraries = self.library_repository.get_all()
            library_ids = [lib.id for lib in libraries]

        for library_id in library_ids:
            try:
                result = self.search_similar_chunks(library_id, query)
                results[library_id] = result
            except Exception as e:
                # Log error but continue with other libraries
                logger.error(f"Error searching library {library_id}: {e}")
                results[library_id] = SearchResult(
                    chunks=[], scores=[], total_found=0, search_time_ms=0.0, index_type="unknown"
                )

        return results

    def get_search_suggestions(self, library_id: str, partial_query: str, limit: int = 5) -> list[str]:
        """Get search suggestions based on chunk text content."""
        # This is a simple implementation - in a real system, you might use
        # more sophisticated approaches like prefix matching or fuzzy search

        library = self.library_repository.get_by_id(library_id)
        if not library:
            return []

        suggestions = set()
        partial_lower = partial_query.lower()

        for doc in library.documents:
            for chunk in doc.chunks:
                words = chunk.text.lower().split()
                for word in words:
                    if word.startswith(partial_lower) and len(word) > len(partial_lower):
                        suggestions.add(word)
                        if len(suggestions) >= limit:
                            break
                if len(suggestions) >= limit:
                    break
            if len(suggestions) >= limit:
                break

        return list(suggestions)[:limit]

    def get_search_analytics(self, library_id: str) -> dict[str, Any]:
        """Get analytics about search performance and usage."""
        library = self.library_repository.get_by_id(library_id)
        if not library:
            return {}

        # Get index information
        index_info = self.library_repository.get_index_info(library_id)

        # Calculate chunk statistics
        total_chunks = sum(len(doc.chunks) for doc in library.documents)
        avg_chunk_length = 0
        if total_chunks > 0:
            total_text_length = sum(len(chunk.text) for doc in library.documents for chunk in doc.chunks)
            avg_chunk_length = total_text_length / total_chunks

        # Calculate embedding dimensions
        embedding_dim = 0
        if total_chunks > 0:
            for doc in library.documents:
                for chunk in doc.chunks:
                    if chunk.embedding:
                        embedding_dim = len(chunk.embedding)
                        break
                if embedding_dim > 0:
                    break

        return {
            "library_id": library_id,
            "total_documents": len(library.documents),
            "total_chunks": total_chunks,
            "average_chunk_length": round(avg_chunk_length, 2),
            "embedding_dimension": embedding_dim,
            "index_info": index_info,
            "search_capabilities": {
                "vector_search": True,
                "metadata_filtering": True,
                "cross_library_search": True,
                "search_suggestions": True,
            },
        }
