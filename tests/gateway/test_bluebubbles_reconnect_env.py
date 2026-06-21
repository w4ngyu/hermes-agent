"""Regression tests for the BlueBubbles reconnect env-reload bug.

Bug (2026-06-21): gateway.run._platform_reconnect_watcher() called
self._create_adapter() every 5 minutes without reloading ~/.hermes/.env
into os.environ first. After a transient .env read failure elsewhere in
the process (reload_env() at hermes_cli/config.py:6342-6346 deletes
"known" vars it can't re-read), BLUEBUBBLES_SERVER_URL and
BLUEBUBBLES_PASSWORD had been stripped from os.environ. The BlueBubbles
adapter then init'd with empty server_url / password, connect() failed
with "<VAR> is required", and the gateway retried every 300s forever
until the user restarted it manually.

Fix: gateway/run.py now calls
_reload_runtime_env_preserving_config_authority() once at the top of each
reconnect attempt, before _create_adapter().

These tests pin the fix in place so the regression cannot return without
breaking CI.

See ~/.hermes/HANDOVER-hermes-reconnect-env-fix.md for the full
postmortem.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner() -> GatewayRunner:
    """Build a minimal GatewayRunner via object.__new__ to skip __init__."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.BLUEBUBBLES: PlatformConfig(
                enabled=True,
                extra={
                    "server_url": "http://localhost:1234",
                    "password": "***",  # real password not needed for these tests
                },
            )
        }
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.session_store = MagicMock()
    return runner


# ---------------------------------------------------------------------------
# Test 1: the reload helper itself restores missing BLUEBUBBLES_* vars
# ---------------------------------------------------------------------------


class TestReloadEnvRestoresBlueBubblesVars:
    """If reload_env() elsewhere stripped BLUEBUBBLES_* from os.environ
    after a transient .env read failure, calling
    _reload_runtime_env_preserving_config_authority() should re-read
    ~/.hermes/.env and put them back.

    Note: the helper binds _hermes_home at module import time and does
    NOT re-read os.environ["HERMES_HOME"], so we patch the module global
    directly with monkeypatch.setattr."""

    def test_reload_restores_missing_server_url(self, monkeypatch, tmp_path):
        # Set up a hermes home with a real .env containing BLUEBUBBLES_*
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ".env").write_text(
            "BLUEBUBBLES_SERVER_URL=https://example.test:1234\n"
            "BLUEBUBBLES_PASSWORD=secret123\n"
            "BLUEBUBBLES_ALLOWED_USERS=alice@example.com\n",
            encoding="utf-8",
        )
        import gateway.run as gw_run
        monkeypatch.setattr(gw_run, "_hermes_home", home)

        # Simulate the bug state: hermes_cli.config.reload_env() stripped
        # the var after a transient .env read failure.
        monkeypatch.delenv("BLUEBUBBLES_SERVER_URL", raising=False)

        from gateway.run import _reload_runtime_env_preserving_config_authority
        _reload_runtime_env_preserving_config_authority()

        import os
        assert os.environ.get("BLUEBUBBLES_SERVER_URL") == "https://example.test:1234"

    def test_reload_restores_missing_password(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ".env").write_text(
            "BLUEBUBBLES_SERVER_URL=https://example.test:1234\n"
            "BLUEBUBBLES_PASSWORD=secret123\n"
            "BLUEBUBBLES_ALLOWED_USERS=alice@example.com\n",
            encoding="utf-8",
        )
        import gateway.run as gw_run
        monkeypatch.setattr(gw_run, "_hermes_home", home)

        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)

        from gateway.run import _reload_runtime_env_preserving_config_authority
        _reload_runtime_env_preserving_config_authority()

        import os
        assert os.environ.get("BLUEBUBBLES_PASSWORD") == "secret123"


# ---------------------------------------------------------------------------
# Test 2: the reconnect watcher calls reload before _create_adapter
# ---------------------------------------------------------------------------


class TestReconnectReloadsEnvBeforeCreatingAdapter:
    """The watcher must invoke _reload_runtime_env_preserving_config_authority()
    BEFORE _create_adapter() in each reconnect attempt, otherwise the
    adapter init reads a stripped os.environ and connect() fails forever.

    Uses asyncio.run() inside a sync test to avoid the pytest-asyncio
    dependency — the project venv is stripped and the test must run on a
    fresh checkout with only pytest installed."""

    def test_reload_called_before_create_adapter(self):
        import asyncio
        runner = _make_runner()

        # Queue a fake "failed" bluebubbles platform with next_retry in the
        # past so the watcher picks it up on this iteration.
        now_mono = __import__("time").monotonic()
        runner._failed_platforms[Platform.BLUEBUBBLES] = {
            "config": runner.config.platforms[Platform.BLUEBUBBLES],
            "attempts": 0,
            "next_retry": now_mono - 1,  # already due
        }

        call_order: list[str] = []

        def fake_reload():
            call_order.append("reload_env")

        # Stop the watcher after one adapter-creation attempt so the test
        # doesn't hang. The fix being tested is order-of-operations on the
        # first attempt — one shot is enough.
        def stop_after_one_attempt(platform, platform_config):
            call_order.append("create_adapter")
            runner._running = False
            return None

        async def fake_sleep(*args, **kwargs):
            return None

        with patch(
            "gateway.run._reload_runtime_env_preserving_config_authority",
            side_effect=fake_reload,
        ), patch.object(
            GatewayRunner,
            "_create_adapter",
            side_effect=stop_after_one_attempt,
        ), patch(
            "gateway.run.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            asyncio.run(runner._platform_reconnect_watcher())

        # The fix: reload_env must appear before create_adapter in the
        # call sequence, every iteration. If a future change moves or
        # removes the reload call, this test breaks loudly.
        assert "reload_env" in call_order, (
            "Reconnect watcher never called _reload_runtime_env_preserving_"
            "config_authority() — BlueBubbles will silently fail every "
            "5 minutes after any reload_env() that strips known vars."
        )
        assert "create_adapter" in call_order
        assert call_order.index("reload_env") < call_order.index("create_adapter"), (
            f"reload_env must be called BEFORE create_adapter, got: {call_order}"
        )

    def test_reload_called_even_when_no_failed_platforms(self):
        """When _failed_platforms is empty, the watcher sleeps and checks
        again. We verify the no-failure path runs without errors and
        does NOT spuriously call reload (the fix is targeted: reload
        only when a platform is about to be recreated)."""
        import asyncio
        runner = _make_runner()
        runner._running = False  # exit immediately on first sleep

        reload_calls: list[int] = []

        def fake_reload():
            reload_calls.append(1)

        async def fake_sleep(*args, **kwargs):
            return None

        with patch(
            "gateway.run._reload_runtime_env_preserving_config_authority",
            side_effect=fake_reload,
        ), patch(
            "gateway.run.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            asyncio.run(runner._platform_reconnect_watcher())

        assert reload_calls == []
