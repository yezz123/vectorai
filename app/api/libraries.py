"""FastAPI endpoints for library operations.

Implements CRUD operations for libraries with proper error handling and validation.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.dependencies import get_library_service
from app.models.base import CreateLibraryRequest, Library, UpdateLibraryRequest
from app.services.library_service import LibraryService

# Get logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("/", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(
    request: CreateLibraryRequest, library_service: LibraryService = Depends(get_library_service)
) -> Library:
    """Create a new library."""
    logger.info(f"Creating library: {request.name}")
    try:
        library = library_service.create_library(request)
        logger.info(f"Library created successfully with ID: {library.id}")
        return library
    except ValueError as e:
        logger.warning(f"Invalid request for library creation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create library: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create library: {str(e)}",
        ) from e


@router.get("/", response_model=list[Library])
async def get_all_libraries(library_service: LibraryService = Depends(get_library_service)) -> list[Library]:
    """Get all libraries."""
    logger.debug("Retrieving all libraries")
    try:
        libraries = library_service.get_all_libraries()
        logger.info(f"Retrieved {len(libraries)} libraries")
        return libraries
    except Exception as e:
        logger.error(f"Failed to retrieve libraries: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve libraries: {str(e)}",
        ) from e


@router.get("/{library_id}", response_model=Library)
async def get_library(library_id: str, library_service: LibraryService = Depends(get_library_service)) -> Library:
    """Get a library by ID."""
    logger.debug(f"Retrieving library: {library_id}")
    try:
        library = library_service.get_library(library_id)
        if not library:
            logger.warning(f"Library not found: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")
        logger.debug(f"Library retrieved successfully: {library_id}")
        return library
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve library: {str(e)}",
        ) from e


@router.put("/{library_id}", response_model=Library)
async def update_library(
    library_id: str, request: UpdateLibraryRequest, library_service: LibraryService = Depends(get_library_service)
) -> Library:
    """Update a library."""
    logger.info(f"Updating library: {library_id}")
    try:
        library = library_service.update_library(library_id, request)
        if not library:
            logger.warning(f"Library not found for update: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")
        logger.info(f"Library updated successfully: {library_id}")
        return library
    except ValueError as e:
        logger.warning(f"Invalid request for library update {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update library: {str(e)}",
        ) from e


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: str, library_service: LibraryService = Depends(get_library_service)):
    """Delete a library."""
    logger.info(f"Deleting library: {library_id}")
    try:
        success = library_service.delete_library(library_id)
        if not success:
            logger.warning(f"Library not found for deletion: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")
        logger.info(f"Library deleted successfully: {library_id}")
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete library: {str(e)}",
        ) from e


@router.post("/{library_id}/index", status_code=status.HTTP_200_OK)
async def build_index(
    library_id: str,
    index_type: str = "linear",
    num_hashes: int = 10,
    num_buckets: int = 100,
    library_service: LibraryService = Depends(get_library_service),
) -> dict[str, Any]:
    """Build or rebuild an index for a library."""
    logger.info(f"Building {index_type} index for library: {library_id}")
    try:
        # Prepare kwargs for LSH index
        kwargs = {}
        if index_type == "lsh":
            kwargs = {"num_hashes": num_hashes, "num_buckets": num_buckets}

        success = library_service.build_index(library_id, index_type, **kwargs)
        if not success:
            logger.warning(f"Library not found for index building: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")

        logger.info(f"{index_type} index built successfully for library: {library_id}")
        return {
            "message": f"Index built successfully for library {library_id}",
            "index_type": index_type,
            "library_id": library_id,
        }
    except ValueError as e:
        logger.warning(f"Invalid index parameters for library {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build {index_type} index for library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build index: {str(e)}",
        ) from e


@router.get("/{library_id}/index", status_code=status.HTTP_200_OK)
async def get_index_info(
    library_id: str, library_service: LibraryService = Depends(get_library_service)
) -> dict[str, Any]:
    """Get information about a library's index."""
    logger.debug(f"Getting index info for library: {library_id}")
    try:
        index_info = library_service.get_index_info(library_id)
        if not index_info:
            logger.warning(f"Library not found or has no index: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found or has no index"
            )
        logger.debug(f"Index info retrieved for library: {library_id}")
        return index_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index info for library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index info: {str(e)}",
        ) from e


@router.get("/{library_id}/stats", status_code=status.HTTP_200_OK)
async def get_library_stats(
    library_id: str, library_service: LibraryService = Depends(get_library_service)
) -> dict[str, Any]:
    """Get statistics for a specific library."""
    logger.debug(f"Getting stats for library: {library_id}")
    try:
        stats = library_service.get_library_stats(library_id)
        if not stats:
            logger.warning(f"Library not found for stats: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")
        logger.debug(f"Stats retrieved for library: {library_id}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get library stats: {str(e)}",
        ) from e


@router.get("/index-types/info", status_code=status.HTTP_200_OK)
async def get_index_types_info(
    library_service: LibraryService = Depends(get_library_service),
) -> dict[str, dict[str, Any]]:
    """Get information about available index types."""
    logger.debug("Getting available index types info")
    try:
        info = library_service.get_available_index_types()
        logger.debug("Index types info retrieved successfully")
        return info
    except Exception as e:
        logger.error(f"Failed to get index types info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index types info: {str(e)}",
        ) from e
