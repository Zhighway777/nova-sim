
from glob import glob
from pathlib import Path

import pytest
from bossa_nova.data_visual.trace_post_processor import BossaNovaTraceProcessor

base_path = "./__output__/test_allgather_gemm/test_allgather_gemm_benchmark_p/topo_TOPO.SUPERNODE8"


@pytest.mark.parametrize("base_path", [base_path])
def test_trace_postprocess(base_path):
    visited_case = {}
    for f in glob(f"{base_path}/**/report.yaml", recursive=True):
        case_root = Path(f).parent.parent
        if str(case_root) in visited_case:
            continue
        trace_path = f"{case_root}/gcu00/trace.perfetto-trace"
        if not Path(trace_path).exists():
            continue
        print(f"Processing {trace_path}")
        tp = BossaNovaTraceProcessor(trace_path=trace_path)
        qr = tp.get_esl_bw_stat()
        rst = next(qr)

        visited_case[str(case_root)] = rst

    for case, rst in visited_case.items():
        print(case, rst)
