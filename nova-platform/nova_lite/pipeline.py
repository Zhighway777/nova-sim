from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from nova_platform.base_model import DType
from nova_platform.config import TOPO
from nova_platform.simulator.case import CaseInfo


@dataclass
class SimulationResult:
    """Aggregated artifacts produced by a simulation run."""

    report: Dict
    report_path: Path
    trace_path: Path
    output_dir: Path


class SimulationPipeline:
    """
    High-level orchestrator for the GEMM → dataflow gen → execute → trace → report flow.

    This wraps the lower-level ``CaseInfo`` class so downstream projects can
    trigger a Python-only simulation without interacting with the extensive test harness.
    """

    def __init__(self, config_template: str | Path, output_root: str | Path | None = None):
        repo_dir = Path(__file__).resolve().parents[1]

        config_template = Path(config_template)
        if not config_template.is_absolute():
            config_template = (repo_dir / config_template).resolve()
        self.config_template = config_template
        self.arch = self._detect_arch(self.config_template)

        if output_root is None:
            output_root = repo_dir / "out" / "nova_lite"
        else:
            output_root = Path(output_root)
            if not output_root.is_absolute():
                output_root = (repo_dir / output_root).resolve()
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run_gemm(
        self,
        shape: Iterable[int],
        dtype: DType = DType.FP16,
        bench_version: int = 5,
        quant_type: Optional[str] = None,
        enable_cache: bool = False,
        topo: TOPO = TOPO.STANDALONE,
        force_rerun: bool = True,
        tags: Optional[List[str]] = None,
    ) -> SimulationResult:
        """
        Execute a GEMM benchmark end-to-end and collect the generated artifacts.

        Parameters
        ----------
        shape:
            The GEMM tensor shape in ``[B, M, K, N]`` order.
        dtype:
            Datatype for the computation (defaults to ``FP16``).
        bench_version:
            Controls the GEMM kernel flavour. Versions ``5`` and ``6`` map to the
            shared and local micro-architectures respectively. Other versions fall
            back to the classic GEMM configuration.
        quant_type:
            Optional quantisation mode (matches existing ``bench_gemm_quant_type`` values).
        enable_cache:
            Toggles the cache service inside the executor.
        topo:
            Target topology (defaults to standalone).
        force_rerun:
            If ``True`` (default) the simulation ignores any cached report under ``output_root``.
        tags:
            Optional tag list carried through metadata for downstream tooling.
        """

        shape = list(shape)
        case_dir = self._build_case_dir(dtype, shape, bench_version)
        self._ensure_file_logging(case_dir)
        dataflow_config = {"bench_gemm_op_version": bench_version}
        if quant_type:
            dataflow_config["bench_gemm_quant_type"] = quant_type

        case = CaseInfo(
            optype="gemm",
            shape=shape,
            dtype=dtype,
            config=str(self.config_template),
            outdir=str(case_dir),
            tag=tags or ["nova-lite"],
            dataflow_config=dataflow_config,
            enable_cache=enable_cache,
            topo=topo,
        )
        case.do_sim(force_rerun=force_rerun)
        return self._collect_results(case_dir)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _build_case_dir(self, dtype: DType, shape: List[int], bench_version: int) -> Path:
        shape_token = "-".join(str(dim) for dim in shape)
        arch_token = f"_{self.arch.lower()}" if self.arch else ""
        dirname = f"gemm_v{bench_version}_{dtype.name.lower()}_{shape_token}{arch_token}"
        case_dir = self.output_root / dirname
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _detect_arch(self, config_template: Path) -> str | None:
        try:
            with open(config_template) as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.lower().startswith("arch"):
                        parts = stripped.split(":")
                        if len(parts) >= 2:
                            val = parts[1].strip()
                            return val or None
        except Exception:
            pass
        return None

    def _collect_results(self, case_dir: Path) -> SimulationResult:
        report_files = sorted(case_dir.glob("gcu*/report.yaml"))
        if not report_files:
            raise FileNotFoundError(f"No report.yaml produced under {case_dir}")
        report_path = report_files[0]
        with open(report_path) as f:
            report = yaml.safe_load(f) or {}

        trace_path = report_path.parent / "trace.perfetto-trace"
        return SimulationResult(
            report=report,
            report_path=report_path,
            trace_path=trace_path,
            output_dir=case_dir,
        )

    def _ensure_file_logging(self, case_dir: Path) -> None:
        """Attach a file handler under the case directory to capture INFO logs."""
        log_path = case_dir / "nova-lite.log"
        root = logging.getLogger()
        # Avoid adding duplicate handlers for the same file
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
                break
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
