import pytest

from bossa_nova.base_model import DType
import logging
from tests.test_base import CaseInfo, TOPO

logger = logging.getLogger(__name__)

shape = [
    ([1, 128 * 32, 128, 128], [1, 128, 128, 128]),
]


config = [
    "eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml",
    # "libra_1DIE_3.2TB_24SIP_256OST.yaml"
]


@pytest.mark.parametrize("config", config)
@pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("shape", shape)
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("topo", [TOPO.SUPERNODE8])
def test_allgather_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape, topo):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "allgather",
            "dtype": dtype,
            "shape": shape,
            "enable_cache": False,
            "topo": topo,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)
