"""
Compatibility layer exposing the nova_platform bundle under the historical
``bossa_nova`` namespace required by the pytest suite.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "nova-platform"
if not _BUNDLE_DIR.is_dir():
    raise ModuleNotFoundError(
        f"nova-platform bundle is missing at {_BUNDLE_DIR}. "
        "Install the real bossa_nova package or keep nova-platform bundled with the tests."
    )

_bundle_path = str(_BUNDLE_DIR)
if _bundle_path not in sys.path:
    sys.path.insert(0, _bundle_path)


def _link_alias(alias: str, target: str) -> None:
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{alias}"] = module


MODULE_ALIASES: Dict[str, str] = {
    "base_model": "nova_platform.base_model",
    "benchmark": "nova_platform.benchmark",
    "benchmark.op_trace_obsolete": "nova_platform.benchmark.op_trace_obsolete",
    "config": "nova_platform.config",
    "cost_service": "nova_platform.cost_service",
    "cost_service.cache": "nova_platform.cost_service.cache",
    "cost_service.cache.arch": "nova_platform.cost_service.cache.arch",
    "cost_service.cache.arch.libra": "nova_platform.cost_service.cache.arch.libra",
    "cost_service.cache.arch.eltanin": "nova_platform.cost_service.cache.arch.eltanin",
    "cost_service.cache.base_cache_model": "nova_platform.cost_service.cache.base_cache_model",
    "cost_service.cache.cache_cost_service": "nova_platform.cost_service.cache.cache_cost_service",
    "data_visual": "nova_platform.data_visual",
    "data_visual.post_processor": "nova_platform.data_visual.post_processor",
    "data_visual.trace_post_processor": "nova_platform.data_visual.trace_post_processor",
    "dataflow": "nova_platform.dataflow",
    "dataflow.action": "nova_platform.dataflow.action",
    "dataflow.action.diag_action": "nova_platform.dataflow.action.diag_action",
    "executor": "nova_platform.executor",
    "executor.bossa_nova_executor": "nova_platform.executor.nova_platform_executor",
    "perfetto_protobuf": "nova_platform.perfetto_protobuf",
    "perfetto_protobuf._tgen": "nova_platform.perfetto_protobuf._tgen",
    "utils": "nova_platform.utils",
    "utils.config_utils": "nova_platform.utils.config_utils",
    "utils.cuda_utils": "nova_platform.utils.cuda_utils",
    "utils.fusion_utils": "nova_platform.utils.fusion_utils",
    "pytest_nova_lite": "nova_platform.pytest_nova_lite",
}

for alias, target in MODULE_ALIASES.items():
    _link_alias(alias, target)


__all__ = tuple(sorted({alias.split(".")[0] for alias in MODULE_ALIASES}))
