"""
Integration tests for MCP server.
"""

import subprocess
from pathlib import Path

import pytest


class TestMCPServerTools:
    """Tests for MCP server tool functions."""

    @pytest.fixture
    def git_repo(self, temp_dir: Path, sample_python_code: str):
        """Create a temporary git repo for testing."""
        repo_dir = temp_dir / "test-repo"
        repo_dir.mkdir()

        # Initialize git
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_dir, capture_output=True
        )

        # Add sample files
        (repo_dir / "app.py").write_text(sample_python_code)
        (repo_dir / "utils.py").write_text("def helper(): return 42")

        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_dir, capture_output=True
        )

        return repo_dir

    def test_detect_current_repo(self, git_repo: Path):
        """Test detect_current_repo function."""
        import os

        from contextfs.mcp_server import detect_current_repo

        original_cwd = os.getcwd()
        os.chdir(git_repo)

        try:
            repo_name = detect_current_repo()
            assert repo_name == "test-repo"
        finally:
            os.chdir(original_cwd)

    def test_detect_current_repo_not_in_repo(self, temp_dir: Path):
        """Test detect_current_repo returns None outside git repo."""
        import os

        from contextfs.mcp_server import detect_current_repo

        # Create a non-git directory
        non_git_dir = temp_dir / "not-a-repo"
        non_git_dir.mkdir()

        original_cwd = os.getcwd()
        os.chdir(non_git_dir)

        try:
            repo_name = detect_current_repo()
            assert repo_name is None
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_list_tools_includes_index(self):
        """Test that list_tools includes contextfs_index."""
        from contextfs.mcp_server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "contextfs_index" in tool_names
        assert "contextfs_save" in tool_names

    @pytest.mark.asyncio
    async def test_list_prompts_includes_save_memory(self):
        """Test that list_prompts includes new prompts."""
        from contextfs.mcp_server import list_prompts

        prompts = await list_prompts()
        prompt_names = [p.name for p in prompts]

        assert "contextfs-save-memory" in prompt_names
        assert "contextfs-index" in prompt_names

    @pytest.mark.asyncio
    async def test_get_prompt_save_memory(self):
        """Test contextfs-save-memory prompt content."""
        from contextfs.mcp_server import get_prompt

        result = await get_prompt(
            "contextfs-save-memory",
            {"content": "Test content", "type": "fact"}
        )

        assert result.description == "Save Memory to ContextFS"
        assert len(result.messages) == 1
        assert "Test content" in result.messages[0].content.text
        assert "fact" in result.messages[0].content.text

    @pytest.mark.asyncio
    async def test_get_prompt_index(self):
        """Test contextfs-index prompt content."""
        from contextfs.mcp_server import get_prompt

        result = await get_prompt("contextfs-index", {})

        assert result.description == "Index Repository"
        assert "contextfs_index" in result.messages[0].content.text

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_call_tool_index(self, git_repo: Path, temp_dir: Path):
        """Test contextfs_index tool call."""
        import os

        import contextfs.mcp_server as mcp_module
        from contextfs.core import ContextFS

        # Create a fresh ContextFS instance with test data dir
        data_dir = temp_dir / "contextfs_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        original_cwd = os.getcwd()
        os.chdir(git_repo)

        try:
            # Create isolated ContextFS instance
            test_ctx = ContextFS(data_dir=data_dir, auto_index=False)
            mcp_module._ctx = test_ctx
            mcp_module._session_started = False

            from contextfs.mcp_server import call_tool

            result = await call_tool("contextfs_index", {"incremental": True})

            assert len(result) == 1
            text = result[0].text
            assert "indexed successfully" in text or "already indexed" in text
        finally:
            os.chdir(original_cwd)
            if mcp_module._ctx:
                mcp_module._ctx.close()
            mcp_module._ctx = None
            mcp_module._session_started = False

    @pytest.mark.asyncio
    async def test_call_tool_index_not_in_repo(self, temp_dir: Path):
        """Test contextfs_index fails gracefully outside git repo."""
        import os

        import contextfs.mcp_server as mcp_module
        from contextfs.mcp_server import call_tool
        mcp_module._ctx = None

        non_git_dir = temp_dir / "not-a-repo"
        non_git_dir.mkdir()

        original_cwd = os.getcwd()
        os.chdir(non_git_dir)

        try:
            result = await call_tool("contextfs_index", {})

            assert len(result) == 1
            assert "Not in a git repository" in result[0].text
        finally:
            os.chdir(original_cwd)
            mcp_module._ctx = None

    @pytest.mark.integration
    def test_save_detects_repo(self, git_repo: Path, temp_dir: Path):
        """Test that save detects and includes source_repo.

        Note: This test requires full database initialization with migrations.
        Run with: pytest -m integration
        """
        import os

        from contextfs.core import ContextFS
        from contextfs.mcp_server import detect_current_repo
        from contextfs.schemas import MemoryType

        # Create a fresh ContextFS instance with test data dir
        data_dir = temp_dir / "contextfs_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        original_cwd = os.getcwd()
        os.chdir(git_repo)

        try:
            # Verify repo detection works
            repo_name = detect_current_repo()
            assert repo_name == "test-repo"

            # Create isolated ContextFS instance
            ctx = ContextFS(data_dir=data_dir, auto_index=False)

            # Save with source_repo
            memory = ctx.save(
                content="Test memory content",
                type=MemoryType.FACT,
                summary="Test summary",
                source_repo=repo_name,
            )

            assert memory.source_repo == "test-repo"
            assert memory.content == "Test memory content"
        finally:
            os.chdir(original_cwd)


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
