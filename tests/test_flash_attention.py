import pytest

from bossa_nova.base_model import DType
import logging
from tests.test_base import CaseInfo


logger = logging.getLogger(__name__)

shape = [
    [[1, 12, 4096, 128], [1, 12, 4096, 128]],
    [[1, 48, 4096, 128], [1, 48, 4096, 128]],
    [[1, 64, 2048, 128], [1, 64, 2048, 128]],
    [[1, 64, 1024, 128], [1, 64, 1024, 128]],
    [[1, 16, 4096, 128], [1, 16, 4096, 128]],
    [[1, 64, 1024, 128], [1, 64, 1024, 128]],
    [[1, 16, 8192, 88], [1, 16, 8192, 88]],
    [[1, 96, 3072, 128], [1, 8, 3072, 128]],
    [[1, 96, 1, 128], [1, 8, 3584, 128]],
    [[1, 96, 1, 128], [1, 8, 28672, 128]],
    [[1, 64, 3072, 128], [1, 8, 3072, 128]],
    [[1, 64, 1, 128], [1, 8, 3584, 128]],
    [[1, 64, 1, 128], [1, 8, 28672, 128]],
    [[1, 16, 6144, 88], [1, 8, 6144, 88]],
]


@pytest.mark.timeout(30 * 60)  # timeout 30mins
@pytest.mark.parametrize(
    "config",
    [
        # "libra_1DIE_3.2TB_24SIP_256OST.yaml",
        "eltanin_D60-A.yaml",
        "eltanin_D60-B.yaml",
        "eltanin_D80-A.yaml",
        "eltanin_D80-B.yaml",
        "eltanin_D90-A.yaml",
        "eltanin_D90-B.yaml",
    ],
)
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("shape", shape)
@pytest.mark.parametrize("dropout", [True, False])
# @pytest.mark.parametrize("with_dropout", [False])
def test_flash_attention_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape, dropout):
    shape.append(dropout)
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "sdpa",
            "dtype": dtype,
            "shape": shape,
            "enable_cache": False,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)
