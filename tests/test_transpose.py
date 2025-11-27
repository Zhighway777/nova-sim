import pytest

from bossa_nova.base_model import DType
from tests.test_base import CaseInfo
from bossa_nova.config import BossaNovaConfig
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config


import logging


logger = logging.getLogger(__name__)


@pytest.mark.ci
@pytest.mark.parametrize("config", ["libra_1DIE_3.2TB_24SIP_256OST.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("shape", [([32, 7, 7, 512], [1, 2, 0, 3])])
def test_transpose_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape):
    optype = "transpose"
    case_info = CaseInfo(**{
        "outdir": outdir,
        "config": config,
        "optype": optype,
        "dtype": dtype,
        "shape": shape,
    })
    case_info.do_sim(force_rerun, rank_id, device)
