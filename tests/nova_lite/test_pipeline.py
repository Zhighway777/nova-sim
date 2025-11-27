import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOVA_PLATFORM_ROOT = REPO_ROOT / "nova-platform"
if str(NOVA_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(NOVA_PLATFORM_ROOT))

from nova_lite import SimulationPipeline  # noqa: E402
from nova_platform.base_model import DType  # noqa: E402


def test_python_pipeline_generates_report_and_trace():
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
        assert str(result.output_dir).startswith(str(pipeline.output_root))
    finally:
        keep_artifacts = os.getenv("NOVALITE_KEEP_ARTIFACTS")
        if not keep_artifacts:
            if result and result.output_dir.exists():
                shutil.rmtree(result.output_dir)
            for directory in (pipeline.output_root, pipeline.output_root.parent):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
