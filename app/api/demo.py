"""Demo API endpoints for showcasing Vector Database functionality with Cohere embeddings."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.dependencies import get_demo_service
from app.models.base import DemoRequest, DemoResponse, DemoStatus
from app.services.demo_service import DemoService

# Get logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post("/cohere", response_model=DemoResponse)
async def run_cohere_demo(
    request: DemoRequest, background_tasks: BackgroundTasks, demo_service: DemoService = Depends(get_demo_service)
):
    """Run the Cohere demo to showcase Vector Database functionality.

    This endpoint creates a sample library with documents, generates real embeddings
    using Cohere API, builds different types of indexes, and performs vector similarity searches.
    """
    logger.info(f"Starting Cohere demo with library: {request.library_name}")
    try:
        # Start the demo in the background
        demo_id = demo_service.start_cohere_demo(
            library_name=request.library_name,
            library_description=request.library_description,
            use_cohere=request.use_cohere,
            cohere_api_key=request.cohere_api_key,
        )

        # Add background task to run the demo
        background_tasks.add_task(demo_service.run_demo_async, demo_id)
        logger.info(f"Cohere demo {demo_id} started successfully in background")

        return DemoResponse(
            demo_id=demo_id,
            status=DemoStatus.STARTED,
            message="Cohere demo started successfully",
            estimated_duration="2-3 minutes",
            api_docs_url="/docs",
            health_check_url="/health",
        )

    except Exception as e:
        logger.error(f"Failed to start Cohere demo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start Cohere demo: {str(e)}") from e


@router.get("/cohere/status/{demo_id}", response_model=DemoResponse)
async def get_demo_status(demo_id: str, demo_service: DemoService = Depends(get_demo_service)):
    """Get the current status of a running demo."""
    logger.debug(f"Getting status for demo: {demo_id}")
    try:
        return demo_service.get_demo_status(demo_id)
    except ValueError as e:
        logger.warning(f"Demo {demo_id} not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get demo status for {demo_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get demo status: {str(e)}") from e


@router.get("/cohere/list", response_model=list[DemoResponse])
async def list_demos(demo_service: DemoService = Depends(get_demo_service)):
    """List all available demos with their status."""
    logger.debug("Listing all demos")
    try:
        demos = demo_service.list_all_demos()
        logger.info(f"Retrieved {len(demos)} demos")
        return demos
    except Exception as e:
        logger.error(f"Failed to list demos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list demos: {str(e)}") from e


@router.delete("/cohere/{demo_id}")
async def delete_demo(demo_id: str, demo_service: DemoService = Depends(get_demo_service)):
    """Delete a demo and clean up associated data."""
    logger.info(f"Deleting demo: {demo_id}")
    try:
        demo_service.delete_demo(demo_id)
        logger.info(f"Demo {demo_id} deleted successfully")
        return {"message": f"Demo {demo_id} deleted successfully"}
    except ValueError as e:
        logger.warning(f"Demo {demo_id} not found for deletion: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to delete demo {demo_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete demo: {str(e)}") from e


@router.post("/cohere/quick")
async def run_quick_demo(demo_service: DemoService = Depends(get_demo_service)):
    """Run a quick demo that creates a minimal library and shows basic functionality.

    This is useful for testing the API quickly.
    """
    logger.info("Starting quick demo")
    try:
        demo_id = demo_service.start_quick_demo()

        # Run the quick demo immediately (not in background)
        result = demo_service.run_quick_demo_sync(demo_id)
        logger.info(f"Quick demo {demo_id} completed successfully")

        return {
            "demo_id": demo_id,
            "status": "completed",
            "message": "Quick demo completed successfully",
            "result": result,
        }

    except Exception as e:
        logger.error(f"Failed to run quick demo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to run quick demo: {str(e)}") from e
