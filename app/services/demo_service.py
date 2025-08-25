"""Demo service for showcasing Vector Database functionality with Cohere embeddings."""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from app.models.base import Chunk, CreateDocumentRequest, CreateLibraryRequest, DemoResponse, DemoStatus
from app.utils.embedding_utils import create_test_embeddings, get_embedding_provider

# Get logger for this module
logger = logging.getLogger(__name__)


class DemoService:
    """Service for managing and running Vector Database demos."""

    def __init__(self, persistence_path: str | None = None):
        self.demos: dict[str, dict[str, Any]] = {}
        self.demo_lock = Lock()
        self.embedding_provider = get_embedding_provider()
        self.persistence_path = persistence_path or "data/demos.json"

        # Load existing demos from disk if persistence is enabled
        if self.persistence_path:
            self._load_demos_from_disk()

        # Sample data for demos
        self.sample_documents = [
            {
                "name": "Python Programming Guide",
                "texts": [
                    "Python is a high-level programming language known for its simplicity and readability.",
                    "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                    "Python has a large standard library and extensive third-party package ecosystem.",
                    "It's widely used in web development, data science, machine learning, and automation.",
                ],
            },
            {
                "name": "Machine Learning Basics",
                "texts": [
                    "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                    "Supervised learning uses labeled training data to make predictions on new, unseen data.",
                    "Unsupervised learning finds patterns in data without predefined labels.",
                    "Deep learning uses neural networks with multiple layers to model complex patterns.",
                ],
            },
            {
                "name": "Database Design Principles",
                "texts": [
                    "Database design involves creating a logical and physical structure for storing data.",
                    "Normalization helps eliminate data redundancy and improve data integrity.",
                    "Indexing improves query performance by creating efficient data access paths.",
                    "Proper database design considers scalability, performance, and maintainability.",
                ],
            },
            {
                "name": "Vector Database Concepts",
                "texts": [
                    "Vector databases store and index high-dimensional vector embeddings for similarity search.",
                    "They use specialized indexing algorithms like KD-trees, LSH, and HNSW for efficient retrieval.",
                    "Vector databases are essential for applications involving natural language processing and recommendation systems.",
                    "They enable semantic search by finding vectors that are mathematically similar in high-dimensional space.",
                ],
            },
        ]

        self.search_queries = [
            "Python programming language features",
            "machine learning algorithms and techniques",
            "database optimization and performance",
            "neural networks and deep learning",
            "vector similarity search methods",
            "semantic understanding and embeddings",
        ]

    def _persist_demos_to_disk(self):
        """Persist demos to disk."""
        if not self.persistence_path:
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

            # Convert datetime objects to ISO format strings for JSON serialization
            serializable_demos = {}
            for demo_id, demo_data in self.demos.items():
                serializable_demo = demo_data.copy()
                if serializable_demo.get("started_at"):
                    serializable_demo["started_at"] = serializable_demo["started_at"].isoformat()
                if serializable_demo.get("completed_at"):
                    serializable_demo["completed_at"] = serializable_demo["completed_at"].isoformat()
                serializable_demos[demo_id] = serializable_demo

            # Save to file
            with open(self.persistence_path, "w") as f:
                json.dump(serializable_demos, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist demos: {e}")

    def _load_demos_from_disk(self):
        """Load demos from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            for demo_id, demo_data in data.items():
                if demo_data.get("started_at"):
                    try:
                        demo_data["started_at"] = datetime.fromisoformat(demo_data["started_at"].replace("Z", "+00:00"))
                    except ValueError:
                        demo_data["started_at"] = None
                if demo_data.get("completed_at"):
                    try:
                        demo_data["completed_at"] = datetime.fromisoformat(
                            demo_data["completed_at"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        demo_data["completed_at"] = None

                self.demos[demo_id] = demo_data

        except Exception as e:
            logger.error(f"Failed to load demos from disk: {e}")

    def start_cohere_demo(
        self, library_name: str, library_description: str, use_cohere: bool = True, cohere_api_key: str | None = None
    ) -> str:
        """Start a new Cohere demo."""
        demo_id = str(uuid.uuid4())
        logger.info(f"Starting Cohere demo: {demo_id} with library: {library_name}")

        with self.demo_lock:
            self.demos[demo_id] = {
                "id": demo_id,
                "status": DemoStatus.PENDING,
                "library_name": library_name,
                "library_description": library_description,
                "use_cohere": use_cohere,
                "cohere_api_key": cohere_api_key,
                "library_id": None,
                "started_at": None,
                "completed_at": None,
                "progress": {
                    "step": "Initializing",
                    "completed_steps": [],
                    "total_steps": 12,
                    "current_document": None,
                    "current_index": None,
                    "current_search": None,
                },
                "results": {},
                "error": None,
            }

        # Persist to disk
        self._persist_demos_to_disk()

        logger.info(f"Demo {demo_id} started successfully")
        return demo_id

    def start_quick_demo(self) -> str:
        """Start a quick demo for testing."""
        demo_id = str(uuid.uuid4())
        logger.info(f"Starting quick demo: {demo_id}")

        with self.demo_lock:
            self.demos[demo_id] = {
                "id": demo_id,
                "status": DemoStatus.PENDING,
                "library_name": "Quick Demo Library",
                "library_description": "Quick demonstration of Vector Database functionality",
                "use_cohere": False,  # Use hash-based for speed
                "cohere_api_key": None,
                "library_id": None,
                "started_at": None,
                "completed_at": None,
                "progress": {
                    "step": "Initializing",
                    "completed_steps": [],
                    "total_steps": 6,
                    "current_document": None,
                    "current_index": None,
                    "current_search": None,
                },
                "results": {},
                "error": None,
            }

        # Persist to disk
        self._persist_demos_to_disk()

        logger.info(f"Quick demo {demo_id} started successfully")
        return demo_id

    def get_demo_status(self, demo_id: str) -> DemoResponse:
        """Get the current status of a demo."""
        with self.demo_lock:
            if demo_id not in self.demos:
                raise ValueError(f"Demo {demo_id} not found")

            demo = self.demos[demo_id]

            return DemoResponse(
                demo_id=demo["id"],
                status=demo["status"],
                message=self._get_status_message(demo["status"]),
                library_id=demo["library_id"],
                estimated_duration="2-3 minutes" if demo["status"] == DemoStatus.STARTED else None,
                started_at=demo["started_at"],
                completed_at=demo["completed_at"],
                progress=demo["progress"],
                api_docs_url="/docs",
                health_check_url="/health",
            )

    def list_all_demos(self) -> list[DemoResponse]:
        """List all available demos."""
        with self.demo_lock:
            return [
                DemoResponse(
                    demo_id=demo["id"],
                    status=demo["status"],
                    message=self._get_status_message(demo["status"]),
                    library_id=demo["library_id"],
                    estimated_duration="2-3 minutes" if demo["status"] == DemoStatus.STARTED else None,
                    started_at=demo["started_at"],
                    completed_at=demo["completed_at"],
                    progress=demo["progress"],
                    api_docs_url="/docs",
                    health_check_url="/health",
                )
                for demo in self.demos.values()
            ]

    def delete_demo(self, demo_id: str) -> None:
        """Delete a demo and clean up associated data."""
        with self.demo_lock:
            if demo_id not in self.demos:
                raise ValueError(f"Demo {demo_id} not found")

            del self.demos[demo_id]
            self._persist_demos_to_disk()

    async def run_demo_async(self, demo_id: str) -> None:
        """Run the demo asynchronously."""
        try:
            self._run_demo_sync(demo_id)
        except Exception as e:
            with self.demo_lock:
                if demo_id in self.demos:
                    self.demos[demo_id]["status"] = DemoStatus.FAILED
                    self.demos[demo_id]["error"] = str(e)
                    self.demos[demo_id]["completed_at"] = datetime.now(UTC)
                    self._persist_demos_to_disk()

    def run_quick_demo_sync(self, demo_id: str) -> dict[str, Any]:
        """Run a quick demo synchronously."""
        try:
            return self._run_demo_sync(demo_id, quick=True)
        except Exception as e:
            with self.demo_lock:
                if demo_id in self.demos:
                    self.demos[demo_id]["status"] = DemoStatus.FAILED
                    self.demos[demo_id]["error"] = str(e)
                    self.demos[demo_id]["completed_at"] = datetime.now(UTC)
                    self._persist_demos_to_disk()
            raise

    def _run_demo_sync(self, demo_id: str, quick: bool = False) -> dict[str, Any]:
        """Run the demo synchronously."""
        with self.demo_lock:
            if demo_id not in self.demos:
                raise ValueError(f"Demo {demo_id} not found")

            demo = self.demos[demo_id]
            demo["status"] = DemoStatus.RUNNING
            demo["started_at"] = datetime.now(UTC)

        logger.info(f"Starting demo execution: {demo_id} (quick: {quick})")

        try:
            # Step 1: Create library
            self._update_progress(demo_id, "Creating library", 1)
            logger.info(f"Demo {demo_id}: Creating library")
            from app.core.dependencies import get_library_service

            library_service = get_library_service()
            library_data = CreateLibraryRequest(
                name=demo["library_name"],
                description=demo["library_description"],
                metadata={
                    "category": "demo",
                    "created_by": "demo_service",
                    "embedding_provider": "cohere" if demo["use_cohere"] else "hash",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            library = library_service.create_library(library_data)
            library_id = library.id
            logger.info(f"Demo {demo_id}: Library created with ID: {library_id}")

            with self.demo_lock:
                demo["library_id"] = library_id
                demo["progress"]["completed_steps"].append("Library created")

            # Step 2: Create documents and add chunks
            from app.core.dependencies import get_document_service

            document_service = get_document_service()
            documents_created = []

            # Use fewer documents for quick demo
            documents_to_process = self.sample_documents[:2] if quick else self.sample_documents
            logger.info(f"Demo {demo_id}: Processing {len(documents_to_process)} documents")

            for i, doc_data in enumerate(documents_to_process):
                self._update_progress(demo_id, f"Creating document: {doc_data['name']}", 2 + i)
                logger.info(f"Demo {demo_id}: Creating document: {doc_data['name']}")

                # Create document
                doc_request = CreateDocumentRequest(
                    name=doc_data["name"],
                    metadata={"category": "technical", "difficulty": "intermediate", "topic": doc_data["name"].lower()},
                )
                document = document_service.create_document(library_id, doc_request)
                documents_created.append(document)

                # Add chunks with embeddings
                self._update_progress(demo_id, f"Adding chunks to: {doc_data['name']}", 3 + i)
                logger.info(f"Demo {demo_id}: Adding chunks to document: {doc_data['name']}")

                try:
                    embeddings = create_test_embeddings(doc_data["texts"], use_cohere=demo["use_cohere"])
                    logger.debug(f"Demo {demo_id}: Generated {len(embeddings)} embeddings for {doc_data['name']}")
                except Exception as e:
                    logger.warning(
                        f"Demo {demo_id}: Failed to generate Cohere embeddings, falling back to hash-based: {e}"
                    )
                    # Fallback to hash-based embeddings
                    embeddings = create_test_embeddings(doc_data["texts"], use_cohere=False)

                chunks = []
                for text, embedding in zip(doc_data["texts"], embeddings, strict=False):
                    chunk = Chunk(
                        text=text,
                        embedding=embedding,
                        metadata={
                            "length": len(text),
                            "word_count": len(text.split()),
                            "created_by": "demo_service",
                            "embedding_dimension": len(embedding),
                            "embedding_provider": "cohere" if len(embedding) > 100 else "hash",
                        },
                    )
                    chunks.append(chunk)

                document_service.add_chunks_to_document(library_id, document.id, chunks)
                logger.info(f"Demo {demo_id}: Added {len(chunks)} chunks to document: {doc_data['name']}")

                with self.demo_lock:
                    demo["progress"]["completed_steps"].append(f"Document created: {doc_data['name']}")
                    demo["progress"]["completed_steps"].append(f"Chunks added: {doc_data['name']}")

            # Step 3: Build indexes
            index_types = ["linear", "kdtree"] if quick else ["linear", "kdtree", "lsh"]
            logger.info(f"Demo {demo_id}: Building {len(index_types)} indexes: {index_types}")

            for i, index_type in enumerate(index_types):
                self._update_progress(demo_id, f"Building {index_type} index", 6 + i if quick else 8 + i)
                logger.info(f"Demo {demo_id}: Building {index_type} index")

                # Build index using the library service
                library_service.build_index(library_id, index_type)
                logger.info(f"Demo {demo_id}: {index_type} index built successfully")

                with self.demo_lock:
                    demo["progress"]["completed_steps"].append(f"Index built: {index_type}")

            # Step 4: Perform searches
            from app.core.dependencies import get_search_service

            search_service = get_search_service()
            search_results = []

            # Use fewer queries for quick demo
            queries_to_process = self.search_queries[:3] if quick else self.search_queries
            logger.info(f"Demo {demo_id}: Performing {len(queries_to_process)} searches")

            for i, query in enumerate(queries_to_process):
                self._update_progress(demo_id, f"Searching for: {query}", 8 + i if quick else 11 + i)
                logger.info(f"Demo {demo_id}: Searching for: {query}")

                try:
                    query_embedding = self.embedding_provider.text_to_embedding(query, use_cohere=demo["use_cohere"])
                except Exception as e:
                    logger.warning(
                        f"Demo {demo_id}: Failed to generate query embedding with Cohere, using hash-based: {e}"
                    )
                    query_embedding = self.embedding_provider.text_to_embedding(query, use_cohere=False)

                # Perform search
                from app.models.base import SearchQuery

                search_query = SearchQuery(query_embedding=query_embedding, k=2, filters={})
                result = search_service.search_similar_chunks(library_id, search_query)
                logger.info(f"Demo {demo_id}: Search for '{query}' returned {len(result.chunks)} results")

                search_results.append(
                    {
                        "query": query,
                        "results_count": len(result.chunks),
                        "top_score": result.scores[0] if result.scores else 0,
                    }
                )

                with self.demo_lock:
                    demo["progress"]["completed_steps"].append(f"Search completed: {query}")

            # Step 5: Get library statistics
            self._update_progress(demo_id, "Getting library statistics", 11 if quick else 12)
            logger.info(f"Demo {demo_id}: Getting library statistics")
            stats = library_service.get_library_stats(library_id)

            # Mark demo as completed
            with self.demo_lock:
                demo["status"] = DemoStatus.COMPLETED
                demo["completed_at"] = datetime.now(UTC)
                demo["progress"]["step"] = "Completed"
                demo["progress"]["completed_steps"].append("Demo completed successfully")
                demo["results"] = {
                    "library_id": library_id,
                    "documents_created": len(documents_created),
                    "total_chunks": sum(len(doc.chunks) for doc in documents_created),
                    "indexes_built": len(index_types),
                    "searches_performed": len(search_results),
                    "search_results": search_results,
                    "library_stats": {
                        "name": stats["name"],
                        "total_documents": stats["total_documents"],
                        "total_chunks": stats["total_chunks"],
                    },
                }
                self._persist_demos_to_disk()

            logger.info(f"Demo {demo_id} completed successfully")
            logger.info(f"Demo {demo_id} results: {demo['results']}")
            return demo["results"]

        except Exception as e:
            logger.error(f"Demo {demo_id} failed: {str(e)}", exc_info=True)
            with self.demo_lock:
                demo["status"] = DemoStatus.FAILED
                demo["error"] = str(e)
                demo["completed_at"] = datetime.now(UTC)
                demo["progress"]["step"] = f"Failed: {str(e)}"
                self._persist_demos_to_disk()
            raise

    def _update_progress(self, demo_id: str, step: str, step_number: int):
        """Update the progress of a demo."""
        with self.demo_lock:
            if demo_id in self.demos:
                demo = self.demos[demo_id]
                demo["progress"]["step"] = step
                if step_number <= demo["progress"]["total_steps"]:
                    demo["progress"]["current_step"] = step_number

    def _get_status_message(self, status: DemoStatus) -> str:
        """Get a human-readable status message."""
        messages = {
            DemoStatus.PENDING: "Demo is pending and will start soon",
            DemoStatus.STARTED: "Demo has been started and is queued for execution",
            DemoStatus.RUNNING: "Demo is currently running",
            DemoStatus.COMPLETED: "Demo has completed successfully",
            DemoStatus.FAILED: "Demo failed with an error",
            DemoStatus.CANCELLED: "Demo was cancelled",
        }
        return messages.get(status, "Unknown status")
