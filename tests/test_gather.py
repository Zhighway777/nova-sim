import pytest

from bossa_nova.base_model import DType
from bossa_nova.config import BossaNovaConfig
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config

from tests.test_base import CaseInfo

import logging

logger = logging.getLogger(__name__)


# @pytest.mark.ci # these cases are running slow in cpu version cache. disable until gpu ci runner ready
@pytest.mark.parametrize("config", ["eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml"])
@pytest.mark.parametrize("dtype", [DType.FP32])
# for cpu cache model, 4096 takes more than one hour
# @pytest.mark.parametrize("shape", [[1, 4096, 51200, 12288]])
@pytest.mark.parametrize("shape", [([1, 192, 51200, 12288])])
# @pytest.mark.parametrize("shape", [([1, 4096, 32000, 4096])])
@pytest.mark.parametrize("mu", [2e-8, 3e-8, 4e-8])
# for [1, 4096, 51200, 12288], mu= , hit rate=
# 1e-9  89.5%
# 1e-8  74.7%
# 2e-8  61.9%
# 3e-8  53.2%
# 4e-8  45.9%
# 5e-8  41.9%
# 6e-8  38.2%
# 7e-8  35.0%
# 8e-8  32.6%
# 1e-7  29.9%
# 5e-7  13.8%
# 1e-6   9.7%
# 0.1   0.03%
# 1     0.02%
def test_gather_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape, mu):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gather",
            "dtype": dtype,
            "shape": shape,
            # "enable_cache": True,
            "mu": mu
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


MEM_16G = 16 * 2**30


dtype = DType.FP32
bpe = dtype.get_bpe()


@pytest.mark.parametrize("config", ["eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml"])
@pytest.mark.parametrize("shape,mu", [
    # embedding_dim=1024/bpe
    # mu = -4.100 => skip_m=4, seq_num=100, seq_num支持3位数
    ([1, 64*200, int(MEM_16G/1024), int(1024/bpe)], -1.001),
    ([1, 64*200, int(MEM_16G/64), int(64/bpe)], -1.001),
    ([1, 64*200, int(MEM_16G/1024), int(1024/bpe)], -4.010),
    ([1, 64*200, int(MEM_16G/64), int(64/bpe)], -4.010),
])
def test_gather_for_mmu(outdir, force_rerun, rank_id, device, config, shape, mu):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gather",
            "dtype": dtype,
            "shape": shape,
            "mu": mu,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)
