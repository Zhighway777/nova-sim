from pathlib import Path
import shutil

from nova_lite import SimulationPipeline
from nova_platform.base_model import DType


def test_pipeline_end_to_end():
    repo_dir = Path(__file__).resolve().parent
    config_path = Path("config") / "libra_1DIE_3.2TB_24SIP_256OST.yaml"
    pipeline = SimulationPipeline(config_path)
    result = None

    try:
        result = pipeline.run_gemm(
            shape=[1, 16, 64, 64],
            dtype=DType.FP16,
            bench_version=5,
            force_rerun=True,
        )

        assert result.report_path.exists()
        assert result.trace_path.exists()
        assert "total_latency" in result.report
        assert result.output_dir.is_relative_to(pipeline.output_root)
    finally:
        if result and result.output_dir.exists():
            shutil.rmtree(result.output_dir)
        out_root = pipeline.output_root
        if out_root.is_dir():
            try:
                out_root.rmdir()
            except OSError:
                pass
