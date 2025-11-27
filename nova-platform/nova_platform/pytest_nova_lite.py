#pytest插件 通过 -p nova_platform.pytest_nova_lite添加到命令行解释之前

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pytest

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_STR = str(BUNDLE_ROOT)
if BUNDLE_STR in sys.path:
    sys.path.remove(BUNDLE_STR)
sys.path.insert(0, BUNDLE_STR)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("nova-lite")
    group.addoption(
        "--run-nova-lite",
        action="store_true",
        default=False,
        help="Run nova-lite simulation pipeline instead of collecting tests.",
    )
    group.addoption(
        "--nova-lite-config",
        action="store",
        default="config/libra_1DIE_3.2TB_24SIP_256OST.yaml",
        help="Config template path passed to SimulationPipeline.",
    )
    group.addoption(
        "--nova-lite-shape",
        action="store",
        default="1,512,256,256",
        help="Comma-separated GEMM tensor shape in B,M,K,N order.",
    )
    group.addoption(
        "--nova-lite-dtype",
        action="store",
        default="fp16",
        help="Datatype enum name (matches nova_platform.base_model.DType).",
    )
    group.addoption(
        "--nova-lite-bench-version",
        action="store",
        type=int,
        default=5,
        help="bench_gemm_op_version passed into the pipeline.",
    )
    group.addoption(
        "--nova-lite-tags",
        action="store",
        default="nova-lite-cli",
        help="Comma separated metadata tags carried with the run.",
    )
    group.addoption(
        "--nova-lite-topo",
        action="store",
        default="standalone",
        help="Topology enum name (nova_platform.config.TOPO).",
    )
    group.addoption(
        "--nova-lite-output",
        action="store",
        default=None,
        help="Optional output directory for simulation artifacts.",
    )
    group.addoption(
        "--nova-lite-force-rerun",
        action="store_true",
        default=True,
        help="Force pipeline rerun even if cached artifacts exist.",
    )

## 从命令行中捕获各种参数
def pytest_cmdline_main(config: pytest.Config) -> Optional[int]:
    if not config.getoption("--run-nova-lite"):
        return None

    from nova_lite.pipeline import SimulationPipeline
    from nova_platform.base_model import DType
    from nova_platform.config import TOPO

    reporter = config.pluginmanager.get_plugin("terminalreporter")

    def write_line(message: str) -> None:
        if reporter:
            reporter.write_line(message)
        else:
            print(message)

    try:
        shape = _parse_shape(config.getoption("--nova-lite-shape"))
    except ValueError as exc:  # pragma: no cover - surfaced to CLI
        raise pytest.UsageError(str(exc)) from exc

    dtype_name = config.getoption("--nova-lite-dtype").upper()
    try:
        dtype = DType[dtype_name]
    except KeyError as exc:  # pragma: no cover - surfaced to CLI
        valid = ", ".join(member.name.lower() for member in DType)
        raise pytest.UsageError(f"Unknown dtype '{dtype_name}'. Valid options: {valid}") from exc

    topo_name = config.getoption("--nova-lite-topo").upper()
    try:
        topo = TOPO[topo_name]
    except KeyError as exc:  # pragma: no cover - surfaced to CLI
        valid = ", ".join(member.name.lower() for member in TOPO)
        raise pytest.UsageError(f"Unknown topology '{topo_name}'. Valid options: {valid}") from exc

    config_template = Path(config.getoption("--nova-lite-config"))
    output_root = config.getoption("--nova-lite-output")
    if output_root:
        output_root = Path(output_root)
    tags = _split_tags(config.getoption("--nova-lite-tags"))
    bench_version = config.getoption("--nova-lite-bench-version")
    force_rerun = config.getoption("--nova-lite-force-rerun")

    pipeline = SimulationPipeline(config_template, output_root=output_root)
    write_line(#打印信息到终端
        "Running nova-lite pipeline with "
        f"config={config_template}, "
        f"shape={shape}, "
        f"dtype={dtype.name.lower()}, "
        f"bench_version={bench_version}, "
        f"topo={topo.name.lower()}"
    )

    try:
        result = pipeline.run_gemm(
            shape=shape,
            dtype=dtype,
            bench_version=bench_version,
            topo=topo,
            force_rerun=force_rerun,
            tags=tags,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        write_line(f"Nova-lite pipeline failed: {exc}")
        return 1

    write_line(f"Nova-lite report: {result.report_path}")
    write_line(f"Nova-lite perfetto trace: {result.trace_path}")
    write_line(f"Nova-lite output dir: {result.output_dir}")
    return 0

# 解析形状字符串
def _parse_shape(raw: str) -> List[int]:
    components = [segment.strip() for segment in raw.split(",") if segment.strip()]
    if len(components) != 4:
        raise ValueError(f"Expected 4 shape components (B,M,K,N). Received: {raw!r}")
    try:
        return [int(value) for value in components]
    except ValueError as exc:
        raise ValueError(f"Shape components must be integers. Received: {raw!r}") from exc

# 分割标签字符串
def _split_tags(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):  # pragma: no cover - defensive
        return [str(tag) for tag in raw]
    if "," in raw:
        tags = [segment.strip() for segment in raw.split(",") if segment.strip()]
    else:
        tags = [segment.strip() for segment in shlex.split(raw) if segment.strip()]
    return tags or None
