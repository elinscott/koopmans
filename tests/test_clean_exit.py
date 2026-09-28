"""Closing the AiiDA engine when the interpreter exits.

The engine's connections are closed from an exit hook rather than left to
garbage collection, which reaches them after logging has been dismantled.
The test profile carries no broker, so what is checked here is that the
hook is registered once by both entry points and that it really releases
the profile's resources; the broker's exit warning is checked by hand
against a profile that has one.
"""

from __future__ import annotations

import atexit
from typing import Any

import pytest

from koopmans.aiida.setup import profile


class TestExitHook:
    """``close_engine_at_exit`` registers one hook, whoever asks."""

    def test_repeated_calls_register_one_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asking twice does not leave two hooks behind."""
        monkeypatch.setattr(profile, "_CLOSE_HOOK_REGISTERED", False)
        atexit.unregister(profile.close_engine)
        registered = atexit._ncallbacks()

        profile.close_engine_at_exit()
        profile.close_engine_at_exit()

        assert atexit._ncallbacks() == registered + 1

    def test_the_python_api_registers_the_hook(
        self, aiida_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A profile the caller loaded themselves is closed at exit too."""
        from koopmans.api import _ensure_profile

        monkeypatch.setattr(profile, "_CLOSE_HOOK_REGISTERED", False)
        atexit.unregister(profile.close_engine)
        registered = atexit._ncallbacks()

        _ensure_profile()

        assert atexit._ncallbacks() == registered + 1


class TestSetDaemonDefaults:
    """``setup_profile`` sets the runner poll interval and worker count."""

    def test_reused_profile_gets_the_defaults(
        self, aiida_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rerunning ``setup_profile`` on an existing profile still sets both options.

        ``PROFILE_NAME`` is patched to the test's own throwaway profile so the
        call takes the "already exists" branch against it, rather than the
        real ``koopmans`` profile.
        """
        from aiida.manage.configuration import get_config

        monkeypatch.setattr(profile, "PROFILE_NAME", aiida_profile.name)

        profile.setup_profile()

        config = get_config()
        poll_interval = config.get_option(  # type: ignore[no-untyped-call]
            "runner.poll.interval", scope=aiida_profile.name
        )
        default_workers = config.get_option(  # type: ignore[no-untyped-call]
            "daemon.default_workers", scope=aiida_profile.name
        )
        assert poll_interval == 5
        assert default_workers == 4


class TestCloseEngine:
    """The hook releases the loaded profile's resources."""

    def test_storage_is_released_and_reopens(self, aiida_profile: Any) -> None:
        """The storage connection is closed, and a later user gets a new one."""
        from aiida.manage import get_manager

        manager = get_manager()
        manager.get_profile_storage()
        assert manager.profile_storage_loaded

        profile.close_engine()
        assert not manager.profile_storage_loaded

        assert manager.get_profile_storage() is not None
