from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import PUBLICATION_SCRIPTS


def test_publication_scripts_exist() -> None:
    scripts_dir = Path(__file__).parents[1] / "scripts"
    missing = [name for name in PUBLICATION_SCRIPTS if not (scripts_dir / name).is_file()]
    assert missing == []


def test_demo_smoke_script_exists() -> None:
    script = Path(__file__).parents[1] / "scripts" / "demo_smoke.sh"
    source = script.read_text()
    assert script.is_file()
    assert source.startswith("#!/usr/bin/env bash")
    assert "--with-data-smoke" not in source
    assert "data/ required" in source


@pytest.mark.parametrize("script_name", sorted(PUBLICATION_SCRIPTS))
def test_publication_shell_scripts_are_valid_bash(script_name: str) -> None:
    if not script_name.endswith(".sh"):
        return
    script_path = Path(__file__).parents[1] / "scripts" / script_name
    source = script_path.read_text()
    assert source.startswith("#!/usr/bin/env bash")
