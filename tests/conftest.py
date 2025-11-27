from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _nova_output_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("nova_out")


@pytest.fixture
def outdir(_nova_output_root: Path, request: pytest.FixtureRequest) -> str:
    case_dir = _nova_output_root / request.node.name
    case_dir.mkdir(parents=True, exist_ok=True)
    return str(case_dir)


@pytest.fixture(scope="session")
def force_rerun() -> bool:
    return False


@pytest.fixture(scope="session")
def rank_id() -> int:
    return 0


@pytest.fixture(scope="session")
def device() -> str:
    return "auto"
