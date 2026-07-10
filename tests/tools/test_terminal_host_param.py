"""Tests for terminal tool per-call SSH host parameter."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.terminal_tool as terminal_tool


def _minimal_terminal_config(env_type="local", cwd="/default"):
    """Create a minimal config for testing."""
    return {
        "env_type": env_type,
        "cwd": cwd,
        "timeout": 60,
        "lifetime_seconds": 3600,
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
        "ssh_persistent": False,
    }


class TestTerminalHostOmitted:
    """When host is omitted, behavior should be unchanged."""

    def test_local_execution_unchanged_when_host_omitted(self, monkeypatch):
        """Omitting host should not affect local execution."""
        calls = []

        class FakeEnv:
            env = {}

            def execute(self, command, **kwargs):
                calls.append((command, kwargs))
                return {"output": "ok", "returncode": 0}

        task_id = "test-task-1"
        monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        result = json.loads(terminal_tool.terminal_tool(command="echo hello", task_id=task_id))

        assert result["exit_code"] == 0
        assert result["output"] == "ok"
        assert "host" not in result  # No host field when omitted


class TestTerminalHostUnknown:
    """When host is set to an unknown alias, should return error."""

    def test_unknown_host_returns_error(self, monkeypatch):
        """Unknown SSH host alias should be rejected with clear error."""
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )

        result = json.loads(
            terminal_tool.terminal_tool(command="whoami", host="unknown-host")
        )

        assert result["exit_code"] == -1
        assert "error" in result
        assert "Unknown SSH host alias" in result["error"]
        assert "unknown-host" in result["error"]
        assert result["status"] == "error"
        # Should list known hosts in error message
        assert "Known aliases:" in result["error"]


class TestTerminalHostKnown:
    """When host is set to a known alias, SSH should be used."""

    def test_known_host_uses_ssh(self, monkeypatch):
        """Known SSH host should trigger SSH environment creation."""
        mock_env = MagicMock()
        mock_env.env = {}
        mock_env.execute.return_value = {"output": "user123", "returncode": 0}

        created_envs = []

        def capture_create_environment(env_type, **kwargs):
            created_envs.append({
                "env_type": env_type,
                "ssh_config": kwargs.get("ssh_config"),
            })
            return mock_env

        task_id = "test-task-ssh"
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )
        monkeypatch.setattr(
            terminal_tool, "_create_environment", capture_create_environment
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
        )

        result = json.loads(
            terminal_tool.terminal_tool(command="whoami", host="macmini", task_id=task_id)
        )

        assert result["exit_code"] == 0
        assert result["output"] == "user123"
        assert result["host"] == "macmini"

        # Verify SSH environment was created
        assert len(created_envs) == 1
        assert created_envs[0]["env_type"] == "ssh"
        ssh_config = created_envs[0]["ssh_config"]
        assert ssh_config is not None
        assert ssh_config["host"] == "192.168.1.93"

    def test_known_host_resolves_to_correct_details(self, monkeypatch):
        """Known host alias should resolve to correct hostname and user."""
        mock_env = MagicMock()
        mock_env.env = {}
        mock_env.execute.return_value = {"output": "", "returncode": 0}

        ssh_configs = []

        def capture_create_environment(env_type, **kwargs):
            if kwargs.get("ssh_config"):
                ssh_configs.append(kwargs["ssh_config"])
            return mock_env

        task_id = "test-task-2"
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("docker")
        )
        monkeypatch.setattr(
            terminal_tool, "_create_environment", capture_create_environment
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
        )

        # Test with macbook-personal
        terminal_tool.terminal_tool(
            command="ls", host="macbook-personal", task_id=task_id
        )

        assert len(ssh_configs) == 1
        assert ssh_configs[0]["host"] == "192.168.1.2"


class TestTerminalHostCacheKeying:
    """Different hosts should get distinct cached environments."""

    def test_two_calls_with_different_hosts_get_distinct_environments(self, monkeypatch):
        """Calls with different host values should create and cache distinct environments."""
        environments_created = []

        mock_env_1 = MagicMock()
        mock_env_1.env = {}
        mock_env_1.execute.return_value = {"output": "host1", "returncode": 0}

        mock_env_2 = MagicMock()
        mock_env_2.env = {}
        mock_env_2.execute.return_value = {"output": "host2", "returncode": 0}

        call_count = {"n": 0}

        def create_env_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_env_1
            else:
                return mock_env_2

        task_id = "test-task-multi-host"
        active_envs = {}

        monkeypatch.setattr(terminal_tool, "_active_environments", active_envs)
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )
        monkeypatch.setattr(
            terminal_tool, "_create_environment", create_env_side_effect
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
        )

        # First call with host="macmini"
        result1 = json.loads(
            terminal_tool.terminal_tool(
                command="echo host1", host="macmini", task_id=task_id
            )
        )

        # Second call with host="macbook-personal"
        result2 = json.loads(
            terminal_tool.terminal_tool(
                command="echo host2", host="macbook-personal", task_id=task_id
            )
        )

        # Both should succeed
        assert result1["exit_code"] == 0
        assert result2["exit_code"] == 0

        # Should have created two distinct environments
        assert len(active_envs) == 2
        # Cache keys should be distinct
        assert f"{task_id}::ssh::macmini" in active_envs
        assert f"{task_id}::ssh::macbook-personal" in active_envs

    def test_call_without_host_does_not_collide_with_call_with_host(
        self, monkeypatch
    ):
        """A call without host and a call with host should use different cache keys."""
        mock_env_local = MagicMock()
        mock_env_local.env = {}
        mock_env_local.cwd = "/default"
        mock_env_local.execute.return_value = {"output": "local", "returncode": 0}

        mock_env_ssh = MagicMock()
        mock_env_ssh.env = {}
        mock_env_ssh.execute.return_value = {"output": "remote", "returncode": 0}

        call_count = {"n": 0}

        def create_env_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_env_local
            else:
                return mock_env_ssh

        task_id = "test-task-mixed"
        active_envs = {}

        monkeypatch.setattr(terminal_tool, "_active_environments", active_envs)
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )
        monkeypatch.setattr(
            terminal_tool, "_create_environment", create_env_side_effect
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
        )

        # First call without host
        result1 = json.loads(
            terminal_tool.terminal_tool(command="echo local", task_id=task_id)
        )

        # Second call with host
        result2 = json.loads(
            terminal_tool.terminal_tool(
                command="echo remote", host="macmini", task_id=task_id
            )
        )

        # Both should succeed
        assert result1["exit_code"] == 0
        assert result2["exit_code"] == 0

        # Should have created two distinct environments
        assert len(active_envs) == 2
        # Local should use default/effective_task_id cache key
        # SSH should use host-specific key
        keys = list(active_envs.keys())
        assert len(keys) == 2
        assert any("ssh" in k and "macmini" in k for k in keys)


class TestTerminalHostEnvironmentReuse:
    """Cached environments should be reused across calls to the same host."""

    def test_same_host_reuses_cached_environment(self, monkeypatch):
        """Multiple calls to the same host should reuse the cached environment."""
        mock_env = MagicMock()
        mock_env.env = {}
        mock_env.execute.side_effect = [
            {"output": "call1", "returncode": 0},
            {"output": "call2", "returncode": 0},
        ]

        create_count = {"n": 0}

        def create_env_side_effect(**kwargs):
            create_count["n"] += 1
            return mock_env

        task_id = "test-task-reuse"
        active_envs = {}

        monkeypatch.setattr(terminal_tool, "_active_environments", active_envs)
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: _minimal_terminal_config("local")
        )
        monkeypatch.setattr(
            terminal_tool, "_create_environment", create_env_side_effect
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
        )

        # First call with host="macmini"
        result1 = json.loads(
            terminal_tool.terminal_tool(
                command="whoami", host="macmini", task_id=task_id
            )
        )

        # Second call to same host
        result2 = json.loads(
            terminal_tool.terminal_tool(
                command="whoami", host="macmini", task_id=task_id
            )
        )

        # Both should succeed with expected outputs
        assert result1["exit_code"] == 0
        assert result1["output"] == "call1"
        assert result2["exit_code"] == 0
        assert result2["output"] == "call2"

        # Environment should have been created only once
        assert create_count["n"] == 1
        # But should be cached
        assert len(active_envs) == 1
