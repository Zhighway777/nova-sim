import pytest

from bossa_nova.base_model import DType
from bossa_nova.config import BossaNovaConfig
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config


import logging

from tests.test_base import CaseInfo

logger = logging.getLogger(__name__)


@pytest.mark.ci
@pytest.mark.parametrize("config", ["libra_1DIE_3.2TB_24SIP_256OST.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP32])
# @pytest.mark.parametrize("shape", [[128, 4096], [4096, 4096]])
@pytest.mark.parametrize("shape", [[1, 1024, 1024], [12, 1024, 1024]])
def test_add_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(**{
        "config": config,
        "optype": "add",
        "dtype": dtype,
        "shape": shape,
        "enable_cache": False,
        "outdir": outdir
    })
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.ci
@pytest.mark.parametrize("config", ["libra_1DIE_3.2TB_24SIP_256OST.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP32])
@pytest.mark.parametrize("shape", [[128, 4096]])
def test_mul_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(**{
        "config": config,
        "optype": "mul",
        "dtype": dtype,
        "shape": shape,
        "enable_cache": False,
        "outdir": outdir
    })

    case_info.do_sim(force_rerun, rank_id, device)
