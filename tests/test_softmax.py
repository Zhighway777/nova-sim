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
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("softmax_shape", [[12*4096, 4096]])
def test_softmax_benchmark(outdir, force_rerun, rank_id, device, config, dtype, softmax_shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": 'softmax',
            "dtype": dtype,
            "shape": softmax_shape,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.ci
@pytest.mark.parametrize("config", ["eltanin_v0.6_48SIP_1.55GHz_512KB.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("softmax_shape", [[12*4096, 4096]])
def test_softmax_eltanin_benchmark(outdir, force_rerun, rank_id, device, config, dtype, softmax_shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": 'softmax',
            "dtype": dtype,
            "shape": softmax_shape,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.ci
@pytest.mark.parametrize("config", ["libra_1DIE_3.2TB_24SIP_256OST.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP16])
@pytest.mark.parametrize("softmax_shape", [[12*4096, 4096]])
def test_softmax_backward_benchmark(outdir, force_rerun, rank_id, device, config, dtype, softmax_shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": 'softmaxbackward',
            "dtype": dtype,
            "shape": softmax_shape,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)
