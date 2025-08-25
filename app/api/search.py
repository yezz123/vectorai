"""FastAPI endpoints for vector similarity search operations.

Implements k-NN search with metadata filtering capabilities.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_search_service
from app.models.base import SearchQuery, SearchResult
from app.services.search_service import SearchService

# Get logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/libraries/{library_id}", response_model=SearchResult)
async def search_similar_chunks(
    library_id: str, query: SearchQuery, search_service: SearchService = Depends(get_search_service)
) -> SearchResult:
    """Search for similar chunks in a specific library using vector similarity."""
    logger.info(f"Searching for similar chunks in library: {library_id}, k={query.k}")
    try:
        result = search_service.search_similar_chunks(library_id, query)
        logger.info(f"Search completed for library {library_id}: found {result.total_found} results")
        return result
    except ValueError as e:
        logger.warning(f"Invalid search query for library {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to perform search in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to perform search: {str(e)}"
        )


@router.post("/libraries", status_code=status.HTTP_200_OK)
async def search_across_libraries(
    query: SearchQuery, library_ids: list[str] = None, search_service: SearchService = Depends(get_search_service)
) -> dict[str, SearchResult]:
    """Search for similar chunks across multiple libraries."""
    logger.info(f"Searching across libraries, k={query.k}, library_ids={library_ids}")
    try:
        results = search_service.search_across_libraries(query, library_ids)
        total_results = sum(result.total_found for result in results.values())
        logger.info(
            f"Cross-library search completed: found {total_results} total results across {len(results)} libraries"
        )
        return results
    except ValueError as e:
        logger.warning(f"Invalid cross-library search query: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to perform cross-library search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform cross-library search: {str(e)}",
        )


@router.get("/libraries/{library_id}/suggestions", status_code=status.HTTP_200_OK)
async def get_search_suggestions(
    library_id: str, partial_query: str, limit: int = 5, search_service: SearchService = Depends(get_search_service)
) -> list[str]:
    """Get search suggestions based on chunk text content."""
    logger.debug(f"Getting search suggestions for library {library_id}, query: '{partial_query}', limit: {limit}")
    try:
        if not partial_query or not partial_query.strip():
            logger.warning(f"Empty partial query for library {library_id}")
            raise ValueError("Partial query cannot be empty")

        if limit <= 0 or limit > 20:
            logger.warning(f"Invalid limit for search suggestions: {limit}")
            raise ValueError("Limit must be between 1 and 20")

        suggestions = search_service.get_search_suggestions(library_id, partial_query.strip(), limit)
        logger.info(f"Generated {len(suggestions)} search suggestions for library {library_id}")
        return suggestions
    except ValueError as e:
        logger.warning(f"Invalid request for search suggestions in library {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get search suggestions for library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get search suggestions: {str(e)}"
        )


@router.get("/libraries/{library_id}/analytics", status_code=status.HTTP_200_OK)
async def get_search_analytics(
    library_id: str, search_service: SearchService = Depends(get_search_service)
) -> dict[str, Any]:
    """Get analytics about search performance and usage for a library."""
    logger.debug(f"Getting search analytics for library: {library_id}")
    try:
        analytics = search_service.get_search_analytics(library_id)
        logger.info(f"Search analytics retrieved for library: {library_id}")
        return analytics
    except Exception as e:
        logger.error(f"Failed to get search analytics for library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get search analytics: {str(e)}"
        )


@router.post("/demo", status_code=status.HTTP_200_OK)
async def demo_search(
    query: SearchQuery, search_service: SearchService = Depends(get_search_service)
) -> dict[str, Any]:
    """Demo search endpoint that searches across all libraries."""
    logger.info(f"Performing demo search across all libraries, k={query.k}")
    try:
        # Search across all libraries
        results = search_service.search_across_libraries(query)

        # Aggregate results
        total_chunks = sum(result.total_found for result in results.values())
        total_libraries = len(results)

        # Get top results across all libraries
        all_chunks = []

        for library_id, result in results.items():
            for chunk, score in zip(result.chunks, result.scores, strict=False):
                all_chunks.append({"chunk": chunk, "score": score, "library_id": library_id})

        # Sort by score and get top k
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_chunks[: query.k]

        logger.info(f"Demo search completed: found {total_chunks} total chunks across {total_libraries} libraries")
        return {
            "query": {"k": query.k, "filters": query.filters},
            "results": {
                "total_chunks_found": total_chunks,
                "total_libraries_searched": total_libraries,
                "top_results": top_results,
            },
            "search_metadata": {"search_type": "cross_library", "algorithm": "aggregated_knn"},
        }
    except ValueError as e:
        logger.warning(f"Invalid demo search query: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to perform demo search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to perform demo search: {str(e)}"
        )
