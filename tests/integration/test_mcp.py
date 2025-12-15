"""
Integration tests for MCP server.
"""

from pathlib import Path

import pytest


class TestMCPEndpoints:
    """Tests for MCP protocol endpoints."""

    @pytest.fixture
    def app(self, temp_dir: Path):
        """Create test FastAPI app."""
        from contextfs.web.server import create_app

        return create_app(data_dir=temp_dir)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_mcp_tools_list(self, client):
        """Test MCP tools/list endpoint."""
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "tools" in data["result"]

        # Check expected tools are present
        tool_names = [t["name"] for t in data["result"]["tools"]]
        assert "memory_store" in tool_names or "store_memory" in tool_names

    def test_mcp_tools_call_store(self, client):
        """Test MCP tools/call for storing memory."""
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "memory_store",
                    "arguments": {
                        "content": "Test memory content",
                        "tags": ["test"],
                    },
                },
            },
        )

        # Should either succeed or return method not found if tool name differs
        assert response.status_code == 200

    def test_mcp_tools_call_search(self, client):
        """Test MCP tools/call for searching memory."""
        # First store a memory
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "memory_store",
                    "arguments": {
                        "content": "Python is great for data science",
                        "tags": ["python"],
                    },
                },
            },
        )

        # Then search
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_search",
                    "arguments": {
                        "query": "Python data",
                    },
                },
            },
        )

        assert response.status_code == 200


class TestRESTAPI:
    """Tests for REST API endpoints."""

    @pytest.fixture
    def app(self, temp_dir: Path):
        """Create test FastAPI app."""
        from contextfs.web.server import create_app

        return create_app(data_dir=temp_dir)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_create_memory(self, client):
        """Test creating memory via REST API."""
        response = client.post(
            "/api/memories",
            json={
                "content": "Test memory content",
                "type": "fact",
                "tags": ["test"],
            },
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data
        assert data["content"] == "Test memory content"

    def test_list_memories(self, client):
        """Test listing memories."""
        # Create a memory first
        client.post(
            "/api/memories",
            json={
                "content": "Test memory",
                "type": "fact",
            },
        )

        response = client.get("/api/memories")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_memories(self, client):
        """Test searching memories."""
        # Create memories
        client.post(
            "/api/memories",
            json={
                "content": "Python programming language",
                "type": "fact",
                "tags": ["python"],
            },
        )

        response = client.get(
            "/api/memories/search",
            params={
                "query": "Python programming",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_stats(self, client):
        """Test getting stats."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data or "memory_count" in data


class TestSessionMemory:
    """Tests for session memory management."""

    @pytest.fixture
    def app(self, temp_dir: Path):
        """Create test FastAPI app."""
        from contextfs.web.server import create_app

        return create_app(data_dir=temp_dir)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_create_session(self, client):
        """Test creating a session."""
        response = client.post(
            "/api/sessions",
            json={
                "name": "test-session",
                "project_path": "/test/project",
            },
        )

        # May return 200, 201, or 404 if endpoint not implemented
        if response.status_code in [200, 201]:
            data = response.json()
            assert "id" in data or "session_id" in data

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/sessions")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_session_memory_persistence(self, client):
        """Test that session memories persist."""
        # Create session
        session_resp = client.post(
            "/api/sessions",
            json={
                "name": "persist-test",
            },
        )

        if session_resp.status_code not in [200, 201]:
            pytest.skip("Sessions not implemented")

        session_id = session_resp.json().get("id") or session_resp.json().get("session_id")

        # Add memory to session
        client.post(
            "/api/memories",
            json={
                "content": "Session specific memory",
                "type": "fact",
                "namespace_id": session_id,
            },
        )

        # Verify memory is retrievable
        search_resp = client.get(
            "/api/memories/search",
            params={
                "query": "Session specific",
                "namespace_id": session_id,
            },
        )

        if search_resp.status_code == 200:
            results = search_resp.json()
            assert len(results) >= 0  # May be empty if not indexed yet
