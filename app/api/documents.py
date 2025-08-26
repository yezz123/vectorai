"""FastAPI endpoints for document operations.

Implements CRUD operations for documents within libraries.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_document_service
from app.models.base import Chunk, CreateDocumentRequest, Document, UpdateDocumentRequest
from app.services.document_service import DocumentService

# Get logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])


@router.post("/", response_model=Document, status_code=status.HTTP_201_CREATED)
async def create_document(
    library_id: str, request: CreateDocumentRequest, document_service: DocumentService = Depends(get_document_service)
) -> Document:
    """Create a new document in a library."""
    logger.info(f"Creating document '{request.name}' in library: {library_id}")
    try:
        document = document_service.create_document(library_id, request)
        if not document:
            logger.warning(f"Library not found for document creation: {library_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library with ID {library_id} not found")
        logger.info(f"Document created successfully with ID: {document.id}")
        return document
    except ValueError as e:
        logger.warning(f"Invalid request for document creation in library {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create document in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}",
        ) from e


@router.get("/", response_model=list[Document])
async def get_all_documents(
    library_id: str, document_service: DocumentService = Depends(get_document_service)
) -> list[Document]:
    """Get all documents from a library."""
    logger.debug(f"Retrieving all documents from library: {library_id}")
    try:
        documents = document_service.get_all_documents(library_id)
        logger.info(f"Retrieved {len(documents)} documents from library: {library_id}")
        return documents
    except Exception as e:
        logger.error(f"Failed to retrieve documents from library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}",
        ) from e


@router.get("/{document_id}", response_model=Document)
async def get_document(
    library_id: str, document_id: str, document_service: DocumentService = Depends(get_document_service)
) -> Document:
    """Get a document by ID from a library."""
    logger.debug(f"Retrieving document {document_id} from library: {library_id}")
    try:
        document = document_service.get_document(library_id, document_id)
        if not document:
            logger.warning(f"Document not found: {document_id} in library: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in library {library_id}",
            )
        logger.debug(f"Document retrieved successfully: {document_id}")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve document {document_id} from library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}",
        ) from e


@router.put("/{document_id}", response_model=Document)
async def update_document(
    library_id: str,
    document_id: str,
    request: UpdateDocumentRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> Document:
    """Update a document in a library."""
    logger.info(f"Updating document {document_id} in library: {library_id}")
    try:
        document = document_service.update_document(library_id, document_id, request)
        if not document:
            logger.warning(f"Document not found for update: {document_id} in library: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in library {library_id}",
            )
        logger.info(f"Document updated successfully: {document_id}")
        return document
    except ValueError as e:
        logger.warning(f"Invalid request for document update {document_id} in library {library_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {document_id} in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}",
        ) from e


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    library_id: str, document_id: str, document_service: DocumentService = Depends(get_document_service)
):
    """Delete a document from a library."""
    logger.info(f"Deleting document {document_id} from library: {library_id}")
    try:
        success = document_service.delete_document(library_id, document_id)
        if not success:
            logger.warning(f"Document not found for deletion: {document_id} in library: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in library {library_id}",
            )
        logger.info(f"Document deleted successfully: {document_id}")
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id} from library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        ) from e


@router.post("/{document_id}/chunks", status_code=status.HTTP_200_OK)
async def add_chunks_to_document(
    library_id: str,
    document_id: str,
    chunks: list[Chunk],
    document_service: DocumentService = Depends(get_document_service),
) -> dict[str, Any]:
    """Add chunks to a document."""
    logger.info(f"Adding {len(chunks)} chunks to document {document_id} in library: {library_id}")
    try:
        success = document_service.add_chunks_to_document(library_id, document_id, chunks)
        if not success:
            logger.warning(f"Document not found for chunk addition: {document_id} in library: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in library {library_id}",
            )

        logger.info(f"Successfully added {len(chunks)} chunks to document {document_id}")
        return {
            "message": f"Added {len(chunks)} chunks to document {document_id}",
            "document_id": document_id,
            "library_id": library_id,
            "chunks_added": len(chunks),
        }
    except ValueError as e:
        logger.warning(f"Invalid request for adding chunks to document {document_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add chunks to document {document_id} in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add chunks: {str(e)}",
        ) from e


@router.get("/{document_id}/stats", status_code=status.HTTP_200_OK)
async def get_document_stats(
    library_id: str, document_id: str, document_service: DocumentService = Depends(get_document_service)
) -> dict[str, Any]:
    """Get statistics for a specific document."""
    logger.debug(f"Getting stats for document {document_id} in library: {library_id}")
    try:
        stats = document_service.get_document_stats(library_id, document_id)
        if not stats:
            logger.warning(f"Document not found for stats: {document_id} in library: {library_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in library {library_id}",
            )
        logger.debug(f"Document stats retrieved successfully: {document_id}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for document {document_id} in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document stats: {str(e)}",
        ) from e


@router.get("/search/name", response_model=list[Document])
async def search_documents_by_name(
    library_id: str, name_query: str, document_service: DocumentService = Depends(get_document_service)
) -> list[Document]:
    """Search documents by name (case-insensitive partial match)."""
    logger.debug(f"Searching documents by name '{name_query}' in library: {library_id}")
    try:
        documents = document_service.search_documents_by_name(library_id, name_query)
        logger.info(f"Found {len(documents)} documents matching '{name_query}' in library: {library_id}")
        return documents
    except Exception as e:
        logger.error(f"Failed to search documents by name in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search documents: {str(e)}",
        ) from e


@router.post("/search/metadata", response_model=list[Document])
async def search_documents_by_metadata(
    library_id: str, metadata_filters: dict[str, Any], document_service: DocumentService = Depends(get_document_service)
) -> list[Document]:
    """Search documents by metadata filters."""
    logger.debug(f"Searching documents by metadata filters in library: {library_id}")
    try:
        documents = document_service.get_documents_by_metadata(library_id, metadata_filters)
        logger.info(f"Found {len(documents)} documents matching metadata filters in library: {library_id}")
        return documents
    except Exception as e:
        logger.error(f"Failed to search documents by metadata in library {library_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search documents by metadata: {str(e)}",
        ) from e
