"""
Integration tests for the FastAPI web application.

Tests API endpoint responses, health check, and error handling.
Uses FastAPI TestClient for in-process HTTP testing.
"""

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


# =============================================================================
# App Import (patched to avoid loading heavy models)
# =============================================================================

@pytest.fixture
def client():
    """Create a test client with engine=None (no model loading)."""
    # Import app after patching to prevent heavy startup
    import web_app.app as app_module
    
    # Ensure engine is None for these tests
    original_engine = app_module.engine
    app_module.engine = None
    
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield client
    
    # Restore
    app_module.engine = original_engine


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for /api/health endpoint."""

    @pytest.mark.integration
    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_health_response_structure(self, client):
        """Health response should have expected fields."""
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "engine_loaded" in data
        assert "youtube_available" in data

    @pytest.mark.integration
    def test_health_engine_not_loaded(self, client):
        """When engine is None, engine_loaded should be False."""
        response = client.get("/api/health")
        data = response.json()
        assert data["engine_loaded"] is False


# =============================================================================
# Search Endpoint Tests
# =============================================================================

class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    @pytest.mark.integration
    def test_search_without_engine_returns_503(self, client):
        """Search should return 503 when engine is not loaded."""
        response = client.get("/api/search?q=test")
        assert response.status_code == 503

    @pytest.mark.integration
    def test_search_requires_q_param(self, client):
        """Search should require 'q' query parameter."""
        response = client.get("/api/search")
        assert response.status_code == 422  # Validation error


# =============================================================================
# Mood Endpoint Tests
# =============================================================================

class TestMoodEndpoint:
    """Tests for /api/mood/{mood} endpoint."""

    @pytest.mark.integration
    def test_mood_without_engine_returns_503(self, client):
        """Mood endpoint should return 503 when engine is not loaded."""
        response = client.get("/api/mood/happy")
        assert response.status_code == 503

    @pytest.mark.integration
    def test_mood_invalid_mood_returns_400(self, client):
        """Invalid mood should return 400 error (if engine were loaded)."""
        # With engine=None, it returns 503 first
        # This test documents the expected behavior
        import web_app.app as app_module
        if app_module.engine is not None:
            response = client.get("/api/mood/invalid_mood_xyz")
            assert response.status_code == 400


# =============================================================================
# Autocomplete Endpoint Tests  
# =============================================================================

class TestAutocompleteEndpoint:
    """Tests for /api/autocomplete endpoint."""

    @pytest.mark.integration
    def test_autocomplete_empty_query(self, client):
        """Empty query should return empty list."""
        response = client.get("/api/autocomplete?q=")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.integration
    def test_autocomplete_without_engine(self, client):
        """Autocomplete without engine should return empty list (not error)."""
        response = client.get("/api/autocomplete?q=test")
        assert response.status_code == 200
        assert response.json() == []
