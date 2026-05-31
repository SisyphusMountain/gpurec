from __future__ import annotations

import gpurec._cli_commands as _commands
import gpurec._cli_helpers as _helpers
from gpurec._cli_commands import *  # noqa: F401,F403
from gpurec._cli_helpers import *  # noqa: F401,F403


_SYNC_EXCLUDED_NAMES = frozenset(
    {
        "_commands",
        "_helpers",
        "_sync_command_hooks",
        "_SYNC_EXCLUDED_NAMES",
        "_doctor_backtracking_readiness",
        "_doctor_preprocessing_readiness",
        "main",
    }
)


def _sync_command_hooks() -> None:
    for name, value in list(globals().items()):
        if name.startswith("__") or name in _SYNC_EXCLUDED_NAMES:
            continue
        if hasattr(_commands, name):
            setattr(_commands, name, value)
        if hasattr(_helpers, name):
            setattr(_helpers, name, value)


def _doctor_backtracking_readiness(
    backtrack_binary,
    package_version: str,
):
    _sync_command_hooks()
    return _helpers._doctor_backtracking_readiness(
        backtrack_binary,
        package_version=package_version,
    )


def _doctor_preprocessing_readiness(
    preprocess_native_lib,
    package_version: str,
):
    _sync_command_hooks()
    return _helpers._doctor_preprocessing_readiness(
        preprocess_native_lib,
        package_version=package_version,
    )


def main(argv: list[str] | None = None) -> None:
    _sync_command_hooks()
    _commands.main(argv)


__all__ = [name for name in globals().keys() if not name.startswith("__")]

if __name__ == "__main__":
    main()
