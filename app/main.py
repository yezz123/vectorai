"""Main FastAPI application for the Vector Database REST API."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import demo, documents, libraries, search
from app.core.config import get_settings
from app.core.dependencies import get_database
from app.core.logging_config import get_logger, log_startup_info

# Get logger for this module
logger = get_logger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    log_startup_info()
    yield
    # Shutdown (if needed in the future)


# Create FastAPI app
app = FastAPI(
    title="Vector Database REST API",
    description="A REST API for indexing and querying documents within a Vector Database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Include API routers
app.include_router(libraries.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(demo.router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Vector Database REST API",
        "version": "1.0.0",
        "description": "A REST API for indexing and querying documents within a Vector Database",
        "endpoints": {
            "libraries": "/api/v1/libraries",
            "documents": "/api/v1/libraries/{library_id}/documents",
            "search": "/api/v1/search",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "features": [
            "Create, read, update, and delete libraries",
            "Create, read, update, and delete documents within libraries",
            "Add chunks with vector embeddings to documents",
            "Build indexes using different algorithms (Linear, KD-Tree, LSH)",
            "Perform k-NN vector similarity search",
            "Metadata filtering for enhanced search results",
            "Thread-safe operations with concurrency control",
            "Optional disk persistence for data durability",
            "Cohere API integration for real embeddings",
        ],
        "configuration": {
            "host": settings.host,
            "port": settings.port,
            "default_index_type": settings.default_index_type,
            "cohere_enabled": settings.cohere_api_key is not None,
        },
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint accessed")
    try:
        # Check database connection
        database = get_database()
        stats = database.get_stats()

        return {
            "status": "healthy",
            "database": {"status": "connected", "stats": stats},
            "timestamp": datetime.now(UTC).isoformat(),
            "configuration": {
                "host": settings.host,
                "port": settings.port,
                "cohere_enabled": settings.cohere_api_key is not None,
                "persistence_enabled": True,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}") from e


@app.get("/api/v1/stats", tags=["stats"])
async def get_system_stats():
    """Get system-wide statistics."""
    logger.debug("System stats endpoint accessed")
    try:
        database = get_database()
        return database.get_stats()
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}") from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc), "type": type(exc).__name__}
    )
