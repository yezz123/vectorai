"""Tests for the Demo API endpoints."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.base import DemoStatus
from app.services.demo_service import DemoService


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_demo_service():
    """Create a mock demo service."""
    return Mock(spec=DemoService)


@pytest.fixture
def sample_demo_request():
    """Sample demo request data."""
    return {
        "library_name": "Test Demo Library",
        "library_description": "A test library for demonstration",
        "use_cohere": True,
        "cohere_api_key": "test_key_123",
    }


@pytest.fixture
def sample_demo_response():
    """Sample demo response data."""
    return {
        "demo_id": "demo-123",
        "status": DemoStatus.STARTED,
        "message": "Demo has been started and is queued for execution",
        "library_id": None,
        "estimated_duration": "2-3 minutes",
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
        "api_docs_url": "/docs",
        "health_check_url": "/health",
    }


class TestDemoAPI:
    """Test cases for Demo API endpoints."""

    def test_get_demo_info(self, client):
        """Test getting demo information."""
        response = client.get("/api/v1/demo/cohere/info")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Cohere Vector Database Demo"
        assert "features" in data
        assert "endpoints" in data
        assert "sample_queries" in data
        assert len(data["features"]) > 0
        assert len(data["endpoints"]) > 0

    def test_start_cohere_demo_success(self, client, sample_demo_request):
        """Test starting a Cohere demo successfully."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.start_cohere_demo.return_value = "demo-123"
            mock_get_service.return_value = mock_service

            response = client.post("/api/v1/demo/cohere", json=sample_demo_request)

            assert response.status_code == 200
            data = response.json()

            assert data["demo_id"] == "demo-123"
            assert data["status"] == DemoStatus.STARTED
            assert "Cohere demo started successfully" in data["message"]
            assert data["estimated_duration"] == "2-3 minutes"

            # Verify service was called correctly
            mock_service.start_cohere_demo.assert_called_once_with(
                library_name=sample_demo_request["library_name"],
                library_description=sample_demo_request["library_description"],
                use_cohere=sample_demo_request["use_cohere"],
                cohere_api_key=sample_demo_request["cohere_api_key"],
            )

    def test_start_cohere_demo_with_defaults(self, client):
        """Test starting a Cohere demo with default values."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.start_cohere_demo.return_value = "demo-456"
            mock_get_service.return_value = mock_service

            # Send minimal request
            response = client.post("/api/v1/demo/cohere", json={})

            assert response.status_code == 200
            data = response.json()

            assert data["demo_id"] == "demo-456"
            assert data["status"] == DemoStatus.STARTED

            # Verify service was called with defaults
            mock_service.start_cohere_demo.assert_called_once_with(
                library_name="Technical Documentation Demo",
                library_description="A collection of technical articles for demonstration",
                use_cohere=True,
                cohere_api_key=None,
            )

    def test_start_cohere_demo_service_error(self, client, sample_demo_request):
        """Test handling service errors when starting demo."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.start_cohere_demo.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            response = client.post("/api/v1/demo/cohere", json=sample_demo_request)

            assert response.status_code == 500
            data = response.json()
            assert "Failed to start Cohere demo" in data["detail"]

    def test_get_demo_status_success(self, client, sample_demo_response):
        """Test getting demo status successfully."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_demo_status.return_value = sample_demo_response
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/demo/cohere/status/demo-123")

            assert response.status_code == 200
            data = response.json()

            assert data["demo_id"] == "demo-123"
            assert data["status"] == DemoStatus.STARTED
            assert "message" in data
            assert "progress" in data

            # Verify service was called correctly
            mock_service.get_demo_status.assert_called_once_with("demo-123")

    def test_get_demo_status_not_found(self, client):
        """Test getting status for non-existent demo."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_demo_status.side_effect = ValueError("Demo not found")
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/demo/cohere/status/non-existent")

            assert response.status_code == 404
            data = response.json()
            assert "Demo not found" in data["detail"]

    def test_get_demo_status_service_error(self, client):
        """Test handling service errors when getting demo status."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_demo_status.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/demo/cohere/status/demo-123")

            assert response.status_code == 500
            data = response.json()
            assert "Failed to get demo status" in data["detail"]

    def test_list_demos_success(self, client, sample_demo_response):
        """Test listing all demos successfully."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.list_all_demos.return_value = [sample_demo_response]
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/demo/cohere/list")

            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["demo_id"] == "demo-123"
            assert data[0]["status"] == DemoStatus.STARTED

            # Verify service was called correctly
            mock_service.list_all_demos.assert_called_once()

    def test_list_demos_service_error(self, client):
        """Test handling service errors when listing demos."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.list_all_demos.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/demo/cohere/list")

            assert response.status_code == 500
            data = response.json()
            assert "Failed to list demos" in data["detail"]

    def test_delete_demo_success(self, client):
        """Test deleting a demo successfully."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            response = client.delete("/api/v1/demo/cohere/demo-123")

            assert response.status_code == 200
            data = response.json()
            assert "Demo demo-123 deleted successfully" in data["message"]

            # Verify service was called correctly
            mock_service.delete_demo.assert_called_once_with("demo-123")

    def test_delete_demo_not_found(self, client):
        """Test deleting a non-existent demo."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.delete_demo.side_effect = ValueError("Demo not found")
            mock_get_service.return_value = mock_service

            response = client.delete("/api/v1/demo/cohere/non-existent")

            assert response.status_code == 404
            data = response.json()
            assert "Demo not found" in data["detail"]

    def test_delete_demo_service_error(self, client):
        """Test handling service errors when deleting demo."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.delete_demo.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            response = client.delete("/api/v1/demo/cohere/demo-123")

            assert response.status_code == 500
            data = response.json()
            assert "Failed to delete demo" in data["detail"]

    def test_run_quick_demo_success(self, client):
        """Test running a quick demo successfully."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.start_quick_demo.return_value = "quick-demo-123"
            mock_service.run_quick_demo_sync.return_value = {
                "library_id": "lib-123",
                "documents_created": 2,
                "total_chunks": 8,
                "indexes_built": 2,
                "searches_performed": 3,
            }
            mock_get_service.return_value = mock_service

            response = client.post("/api/v1/demo/cohere/quick")

            assert response.status_code == 200
            data = response.json()

            assert data["demo_id"] == "quick-demo-123"
            assert data["status"] == "completed"
            assert "Quick demo completed successfully" in data["message"]
            assert "result" in data
            assert data["result"]["library_id"] == "lib-123"

            # Verify service was called correctly
            mock_service.start_quick_demo.assert_called_once()
            mock_service.run_quick_demo_sync.assert_called_once_with("quick-demo-123")

    def test_run_quick_demo_service_error(self, client):
        """Test handling service errors when running quick demo."""
        with patch("app.api.demo.get_demo_service") as mock_get_service:
            mock_service = Mock()
            mock_service.start_quick_demo.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            response = client.post("/api/v1/demo/cohere/quick")

            assert response.status_code == 500
            data = response.json()
            assert "Failed to run quick demo" in data["detail"]

    def test_demo_request_validation(self, client):
        """Test demo request validation."""
        # Test with invalid data
        invalid_request = {
            "library_name": "",  # Empty name
            "library_description": "A" * 1001,  # Too long description
            "use_cohere": "not_a_boolean",  # Invalid boolean
        }

        response = client.post("/api/v1/demo/cohere", json=invalid_request)

        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_demo_endpoints_are_registered(self, client):
        """Test that all demo endpoints are properly registered."""
        # Test all demo endpoints exist
        endpoints = [
            ("POST", "/api/v1/demo/cohere"),
            ("GET", "/api/v1/demo/cohere/status/demo-123"),
            ("GET", "/api/v1/demo/cohere/list"),
            ("DELETE", "/api/v1/demo/cohere/demo-123"),
            ("POST", "/api/v1/demo/cohere/quick"),
            ("GET", "/api/v1/demo/cohere/info"),
        ]

        for method, endpoint in endpoints:
            if method == "POST":
                response = client.post(endpoint, json={})
            elif method == "GET":
                response = client.get(endpoint)
            elif method == "DELETE":
                response = client.delete(endpoint)

            # Should not get 404 (endpoint not found)
            assert response.status_code != 404, f"Endpoint {method} {endpoint} not found"


class TestDemoServiceIntegration:
    """Integration tests for the demo service."""

    def test_demo_service_creation(self):
        """Test that demo service can be created."""
        service = DemoService()
        assert service is not None
        assert hasattr(service, "demos")
        assert hasattr(service, "sample_documents")
        assert hasattr(service, "search_queries")

    def test_demo_service_sample_data(self):
        """Test that demo service has proper sample data."""
        service = DemoService()

        assert len(service.sample_documents) == 4
        assert len(service.search_queries) == 6

        # Check document structure
        for doc in service.sample_documents:
            assert "name" in doc
            assert "texts" in doc
            assert isinstance(doc["texts"], list)
            assert len(doc["texts"]) > 0

        # Check query structure
        for query in service.search_queries:
            assert isinstance(query, str)
            assert len(query) > 0

    def test_demo_status_messages(self):
        """Test that demo status messages are properly formatted."""
        service = DemoService()

        status_messages = {
            DemoStatus.PENDING: "Demo is pending and will start soon",
            DemoStatus.STARTED: "Demo has been started and is queued for execution",
            DemoStatus.RUNNING: "Demo is currently running",
            DemoStatus.COMPLETED: "Demo has completed successfully",
            DemoStatus.FAILED: "Demo failed with an error",
            DemoStatus.CANCELLED: "Demo was cancelled",
        }

        for status, expected_message in status_messages.items():
            message = service._get_status_message(status)
            assert message == expected_message


@pytest.mark.asyncio
class TestDemoAsyncOperations:
    """Test async operations in the demo service."""

    async def test_run_demo_async_success(self):
        """Test running demo asynchronously successfully."""
        service = DemoService()
        demo_id = service.start_cohere_demo("Test", "Description")

        # Mock the sync method to avoid actual execution
        with patch.object(service, "_run_demo_sync") as mock_run:
            await service.run_demo_async(demo_id)
            mock_run.assert_called_once_with(demo_id)

    async def test_run_demo_async_error_handling(self):
        """Test error handling in async demo execution."""
        service = DemoService()
        demo_id = service.start_cohere_demo("Test", "Description")

        # Mock the sync method to raise an error
        with patch.object(service, "_run_demo_sync", side_effect=Exception("Test error")):
            await service.run_demo_async(demo_id)

            # Check that demo status was updated to failed
            status = service.get_demo_status(demo_id)
            assert status.status == DemoStatus.FAILED
            assert "Test error" in status.progress["step"]
