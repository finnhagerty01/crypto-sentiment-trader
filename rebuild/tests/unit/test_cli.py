from __future__ import annotations

import pytest

from trader.cli import COMMANDS, main


def test_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "collect-market" in capsys.readouterr().out


@pytest.mark.parametrize("command", COMMANDS)
def test_placeholder_commands_fail_clearly(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    assert (
        capsys.readouterr().err
        == f"error: command '{command}' is not implemented in Phase 03\n"
    )
