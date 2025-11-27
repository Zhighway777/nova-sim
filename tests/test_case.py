import pytest
import functools
from pathlib import Path
from bossa_nova.base_model import DType
from bossa_nova.config import BossaNovaConfig
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.dataflow.action.diag_action import parse_shape_tile_info
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config
from bossa_nova.utils.cuda_utils import get_gpu_count
from bossa_nova.perfetto_protobuf._tgen import TraceGenerator
import logging

from tests.test_base import CaseInfo
logger = logging.getLogger(__name__)


def test_split_info_parser():
    file_path = "tests/resources/batch_gemm_fp16__0_gemm_fp16_1_4096_12288_1536__grid_shape_list.txt"
    split_info = parse_shape_tile_info(file_path)
    print(split_info)


ci_list = [
    ("libra_1DIE_3.2TB_24SIP_256OST.yaml", "matmul", [1, 4096, 12288, 1536], 596.10e-6),
    ("libra_1DIE_3.2TB_24SIP_256OST.yaml", "matmul", [1, 4096, 12288, 6144], 2346.20e-6),
    ("libra_1DIE_3.2TB_24SIP_256OST.yaml", "matmul", [12, 4096, 128, 4096], 220.9e-6),
]


@pytest.mark.ci
@pytest.mark.parametrize("config,optype,op_shape, expected_res", ci_list)
def test_ci(outdir, config, optype, op_shape, expected_res, rank_id, device, enable_cache=False):
    case_info = CaseInfo(optype, op_shape,
                         expected_res, config=config, enable_cache=enable_cache)
    test_end2end_solution(case_info, outdir, force_rerun=True,
                          rank_id=rank_id, device=device)


def power_scale_bs(shape, power_list):
    for pow in power_list:
        yield [2**pow, *shape[1:]]


def step_n_2048_256_4096(shape):
    for n in range(2048, 4096+256, 256):
        yield [*shape[0:3], n]


def step_k_2048_256_4096(shape):
    for k in range(2048, 4096+256, 256):
        yield [*shape[0:2], k, shape[3]]


pow0_256 = functools.partial(
    power_scale_bs, power_list=list(range(9)))


fp8_quant_config = {"bench_gemm_quant_type": "Wf8t_Af8t"}
fp4_quant_config = {"bench_gemm_quant_type": "Wf4g_Af8t"}

dsm_shared_gemm = {"bench_gemm_op_version": 5}
dsm_local_gemm = {"bench_gemm_op_version": 6}
fp8_dsm_shared_gemm = {**dsm_shared_gemm, **fp8_quant_config}
fp8_dsm_local_gemm = {**dsm_local_gemm, **fp8_quant_config}
fp4_dsm_shared_gemm = {**dsm_shared_gemm, **fp4_quant_config}
fp4_dsm_local_gemm = {**dsm_local_gemm, **fp4_quant_config}

# fmt: off
op_shapes_expected_res = [
    #---------------------------------------------------
    # examples [B, M, K, N]
    #---------------------------------------------------
    # CaseInfo("matmul", [16, 4096, 192, 4096], None,
    #         None, "dev", ["inference generate"], enable_cache=True),

    #===================================================
    # 靶点模型算子形状列表
    #===================================================
    #---------------------------------------------------
    # train
    #---------------------------------------------------
    # from deepseekv2-TP8
#     CaseInfo("linear", [1, 512, 5120, 192], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 512, 1536, 3072], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 512, 5120, 72], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 512, 512, 4096], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("matmul", [16, 4096, 192, 4096], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("matmul", [16, 4096, 4096, 128], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 512, 5120, 3072], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 512, 5120, 160], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),

#     # from deepseekv2-TP1
#     CaseInfo("linear", [1, 4096, 5120, 1536], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 1536, 24576], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 5120, 576], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 512, 32768], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [128, 4096, 192, 4096], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [128, 4096, 4096, 128], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 5120, 24576], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 5120, 160], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 16384, 5120], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 4096, 3072, 5120], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),

#     # from largest shape, always TO, :(
#     CaseInfo("matmul", [128, 32768, 7168, 7168], None,
#             None, "deepseekv2", ["train", "largest shape"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     # from SD
#     CaseInfo("linear", [1, 262144, 2560, 2560], None,
#             None, "SD", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),

#     # from best dot
#     CaseInfo("linear", [1, 32768, 27392, 4096], None,
#             None, "LAMMA2", ["inference prefill", "best dot"], enable_cache=False, dataflow_config=dsm_shared_gemm),

    #---------------------------------------------------
    # inference
    #---------------------------------------------------
    # from deepseekv2-TP8
#     CaseInfo("linear", [16, 1, 5120, 192], None,
#             pow0_256, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [16, 1, 1536, 3072], None,
#             pow0_256, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [16, 1, 5120, 72], None,
#             pow0_256, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [16, 1, 512, 4096], None,
#             pow0_256, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("matmul", [256, 1, 192, 2048], None,
#             step_n_2048_256_4096, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("matmul", [256, 1, 2048, 128], None,
#             step_k_2048_256_4096, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [16, 1, 5120, 3072], None,
#             None, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [16, 1, 5120, 160], None,
#             None, "deepseekv2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),

    # from LAMMA2
    CaseInfo("linear", [1, 129, 12288, 4096], None,
            None, "LAMMA2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("linear", [1, 129, 4096, 4096], None,
            None, "LAMMA2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("linear", [1, 129, 22016, 4096], None,
            None, "LAMMA2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("linear", [1, 129, 4096, 11008], None,
            None, "LAMMA2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("matmul", [16, 1, 12288, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("matmul", [16, 1, 4096, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("matmul", [16, 1, 22016, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    CaseInfo("matmul", [16, 1, 4096, 11008], None,
            None, "LAMMA2", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),

    # 1D OPs, date type according to OP excel
    # from deepseekv2
#     CaseInfo("add", [1, 4096, 1, 5120], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [1, 4096, 1, 3072], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [1, 4096, 1, 1536], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("softmax", [1 * 4096 * 1, 160], None,      # [1, 4096, 1, 160] -> [batch_size, reduce维度]
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP32),
#     CaseInfo("softmax", [128 * 4096 * 1, 4096], None,   # [128, 4096, 1, 4096]
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP32),
#     CaseInfo("silu", [1, 4096, 1, 3072], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [1, 4096, 1, 1536], None,
#             None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [1, 512, 1, 12288], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("gelu", [1, 4096, 1, 6144], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),

    # 推荐系统embedding lookup，P0形状
    # gather             [bs， 次数， 行数， 每行的大小]
#     CaseInfo("gather", [1, 262144, 20971520, 11],  None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
#     CaseInfo("gather", [1, 102400, 20971520, 11],  None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
#     CaseInfo("gather", [1, 26624,  20971520, 11],  None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
#     CaseInfo("gather", [1, 262144, 20971520, 128], None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
#     CaseInfo("gather", [1, 102400, 20971520, 128], None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
#     CaseInfo("gather", [1, 26624,  20971520, 128], None, None, "SGT", ["inference generate"], enable_cache=False, mu=4e-8, dtype=DType.FP32),

    # Cape微信模型 dtype=DType.FP16, P0跑，其他的暂时注释掉
#     CaseInfo("linear", [1,  1, 256,  256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [1,  1, 1024, 256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [1,  1, 1024, 1024], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [1,  1, 256,  256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),
#     CaseInfo("linear", [1,  1, 1024, 256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),
#     CaseInfo("linear", [1,  1, 1024, 1024], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),
#     CaseInfo("linear", [32, 1, 256,  256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [32, 1, 1024, 256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [32, 1, 1024, 1024], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
#     CaseInfo("linear", [32, 1, 256,  256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),
#     CaseInfo("linear", [32, 1, 1024, 256], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),
#     CaseInfo("linear", [32, 1, 1024, 1024], None,
#             None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP32),

    ##CaseInfo("linear", [1, 96, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 96, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 96, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 96, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 128, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 128, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 128, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 128, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 192, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 192, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 192, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 192, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 256, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 256, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 256, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [1, 256, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 96, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 96, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 96, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 96, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 128, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 128, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 128, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 128, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 192, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 192, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 192, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 192, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 256, 256, 1024], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 256, 1024, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 256, 256, 768], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
    ##CaseInfo("linear", [32, 256, 256, 256], None,
    ##        None, "WeChat", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),

    # from LLAMA3-70b-train
#     CaseInfo("linear", [1, 32768, 8192, 1024], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 8192, 128], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 1024, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 8192, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 3584, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], False], None, None,       # 只有ChatGPT是带dropout
#              "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),

#     # CaseInfo("rms_norm	fp16	1	4096	1	8192    ###，应该用rms_norm
#     CaseInfo("layernorm", [1*4096*1, 8192], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [1, 4096, 1, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [1, 32768, 1, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [1, 32768, 1, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),

#     CaseInfo("linear", [1, 65536, 8192, 1024], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 65536, 8192, 128], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 65536, 1024, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 65536, 8192, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 65536, 3584, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128], False], None, None,       # 只有ChatGPT是带dropout
#              "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
 
#     # CaseInfo("rms_norm	fp16	1	8192	1	8192###
#     CaseInfo("layernorm", [1*8192*1, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [1, 8192, 1, 8192], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [1, 65536, 1, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [1, 65536, 1, 3584], None,
#             None, "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
    
#     # from LLAM3-405b-train
#     CaseInfo("linear", [1, 32768, 16384, 1024], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 16384, 64], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 1024,  16384], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 16384, 3328], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [1, 32768, 3328,  16384], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),

#     # CaseInfo("Flashattn	fp16 #重复
    
#     # CaseInfo("rms_norm	fp16	1	2048	1	16384#
#     CaseInfo("layernorm", [1*2048*1, 16384], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [1, 2048, 1, 16384], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [1, 32768, 1, 3328], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [1, 32768, 1, 3328], None,
#             None, "LLAMA3-405B", ["train"], enable_cache=False, dtype=DType.FP16),
    
    # from LLAM3-70b-inference, 32k-prefill
#     CaseInfo("linear", [4, 24576, 8192,  1024], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 24576, 8192,  128], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 24576, 1024,  8192], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 24576, 8192,  28672], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 24576, 28672, 8192], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128], False], None, None, 
#              "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

    # CaseInfo("rms_norm	fp16	4	24576	1	8192##
#     CaseInfo("layernorm", [4*24576*1, 8192], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [4, 24576, 1, 8192], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [4, 24576, 1, 28672], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [4, 24576, 1, 28672], None,
#             None, "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
    
    # from LLAM3-70b-inference, 32k-decode
#     CaseInfo("linear", [4, 1, 8192,  1024], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 1, 8192,  128], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 1, 1024,  8192], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 1, 8192,  28672], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("linear", [4, 1, 28672, 8192], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm),
#     CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], False], None, None, 
#              "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    # CaseInfo("rms_norm	fp16	4	1	1	8192 #???
#     CaseInfo("layernorm", [4*1*1, 8192], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("add", [4, 1, 1, 8192], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("silu", [4, 1, 1, 28672], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),
#     CaseInfo("mul", [4, 1, 1, 28672], None,
#             None, "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    #---------------------------------------------------
    # 1D OPs, date type according to OP excel
    #---------------------------------------------------
    # from test
    CaseInfo("relu", [12 * 4096, 4096], None,
            None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    CaseInfo("sigmoid", [12 * 4096, 4096], None,
            None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    CaseInfo("layernorm", [512, 12288], None,
            None, "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),

    # gather             [bs， 次数， 行数， 每行的大小]
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=1,    dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=1e-6, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=5e-7, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=1e-7, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=6e-8, dtype=DType.FP32),
    CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=4e-8, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=2e-8, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=1e-8, dtype=DType.FP32),

#     # more gather cases
#     # lookup参数    datatype    batchsize   特征数量    dim size    table memory内存占用
#     # gather	    fp32/fp16	1024	    256         128/64/32	10GB
#     # gather	    fp32/fp16	1024	    100	        128/64/32	10GB
#     # gather	    fp32/fp16	1024	    26	        128/64/32	10GB
#     # gather	    fp32/fp16	2048	    256	        128/64/32	10GB
#     # gather	    fp32/fp16	2048	    100	        128/64/32	10GB
#     # gather	    fp32/fp16	2048	    26	        128/64/32	10GB
#     # gather	    fp32/fp16	5120	    256	        128/64/32	10GB
#     # gather	    fp32/fp16	5120	    100	        128/64/32	10GB
#     # gather	    fp32/fp16	5120	    26	        128/64/32	10GB

    # for Power
    # CaseInfo("matmul", [12, 4096, 128, 4096], None,
    #       None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm),

    #---------------------------------------------------
    # Flash Attention
    #---------------------------------------------------
    ## fro gpt175b with dropout
    #CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],True], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],True], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],True], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    ## fro gpt175b without dropout
    #CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],False], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],False], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],False], None, None, 
    #         "gpt175b", ["train"], enable_cache=False, dtype=DType.FP16),

    ## new ops with dropout
    #CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], True], None, None,
    #         "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128],True], None, None,
    #         "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128] ,True], None, None,
    #         "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], True], None, None,
    #         "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    ## new ops without dropout
    #CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], False], None, None,
    #         "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128], False], None, None,
    #         "LLAMA3-70B", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128] ,False], None, None,
    #         "LLAMA3-70B", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], False], None, None,
    #         "LLAMA3-70B", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    ## for llama3_70b, train with dropout
    #CaseInfo("sdpa", [[1, 64, 2048, 128], [1, 64, 2048, 128], True], None, None, 
    #         "llama3", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 8, 32768, 128], True], None, None, 
    #         "llama3", ["train"], enable_cache=False, dtype=DType.FP16),
    ## for llama3_70b, train without dropout
    #CaseInfo("sdpa", [[1, 64, 2048, 128], [1, 64, 2048, 128], False], None, None, 
    #         "llama3", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 8, 32768, 128], False], None, None, 
    #         "llama3", ["train"], enable_cache=False, dtype=DType.FP16),

    ## for mixtral8x70b, train with dropout
    #CaseInfo("sdpa", [[1, 64, 1024, 128], [1, 64, 1024, 128], True], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 16, 4096, 128], [1, 16, 4096, 128], True], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 4, 32768, 128], [1, 4, 32768, 128], True], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),
    ## for mixtral8x70b, train without dropout
    #CaseInfo("sdpa", [[1, 64, 1024, 128], [1, 64, 1024, 128], False], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 16, 4096, 128], [1, 16, 4096, 128], False], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),
    #CaseInfo("sdpa", [[1, 4, 32768, 128], [1, 4, 32768, 128], False], None, None, 
    #         "mixtral", ["train"], enable_cache=False, dtype=DType.FP16),

    ## for deepseekv2, train with dropout
    #CaseInfo("sdpa", [[1, 128, 4096, 192], [1, 128, 4096, 192], True], None, None, 
    #         "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),
    ## for deepseekv2, train without dropout
    #CaseInfo("sdpa", [[1, 128, 4096, 192], [1, 128, 4096, 192], False], None, None, 
    #         "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16),

    ## for hydit, train with dropout
    #CaseInfo("sdpa", [[1, 16, 8192, 88], [1, 16, 8192, 88], True], None, None, 
    #         "hydit", ["train"], enable_cache=False, dtype=DType.FP16),
    ## for hydit, train without dropout
    #CaseInfo("sdpa", [[1, 16, 8192, 88], [1, 16, 8192, 88], False], None, None, 
    #         "hydit", ["train"], enable_cache=False, dtype=DType.FP16),

    ## for gpt175b, inference
    #CaseInfo("sdpa", [[1, 96, 3072, 128], [1, 8, 3072, 128], False], None, None, 
    #         "gpt175b", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 96, 1, 128], [1, 8, 3584, 128], False], None, None, 
    #         "gpt175b", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 96, 24576, 128], [1, 8, 24576, 128], False], None, None, 
    #         "gpt175b", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 96, 1, 128], [1, 8, 28672, 128], False], None, None, 
    #         "gpt175b", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    ## for llama3, inference
    #CaseInfo("sdpa", [[1, 64, 3072, 128], [1, 8, 3072, 128], False], None, None, 
    #         "llama3", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 64, 1, 128], [1, 8, 3584, 128], False], None, None, 
    #         "llama3", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 64, 24576, 128], [1, 8, 24576, 128], False], None, None, 
    #         "llama3", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

    #CaseInfo("sdpa", [[1, 64, 1, 128], [1, 8, 28672, 128], False], None, None, 
    #         "llama3", ["inference generate"], enable_cache=False, dtype=DType.FP16),

    ## for hydit, inference
    #CaseInfo("sdpa", [[1, 16, 6144, 88], [1, 8, 6144, 88], False], None, None, 
    #         "hydit", ["inference prefill"], enable_cache=False, dtype=DType.FP16),

#####===================================================
##### new_op_shape
#####===================================================
#####---------------------------------------------------
##### train
#####---------------------------------------------------
##### llama3-70b-train 32K
####    CaseInfo("linear", [1, 32768, 8192, 1024], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 8192, 128],  None, None, "llama3-70b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 1024, 8192], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 8192, 3584], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 3584, 8192], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # Flashattn	fp16, input Q K V，只用指定Q K的形状即可，output是根据input固定的
####    # Flash attn的input=[[1, 8, 32768, 128], [1, 1, 32768, 128],[1, 1, 32768, 128]]
####    # Flash attn的output=[1, 8, 32768, 128]
####    CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], False], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [1*4096*1, 8192],    None, None, "llama3-70b-32K", ["train"], enable_cache=False, dtype=DType.FP16),      # 应该用rms_norm, [1, 4096, 1, 8192] -> [1 * 4096 * 1, 8192]
####    CaseInfo("add",       [1, 4096, 1, 8192],  None, None, "llama3-70b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 32768, 1, 3584], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 32768, 1, 3584], None, None, "llama3-70b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####
####    # llama3-70b-train 64K			
####    CaseInfo("linear", [1, 65536, 8192, 1024], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 65536, 8192, 128],  None, None, "llama3-70b-64K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 65536, 1024, 8192], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 65536, 8192, 3584], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 65536, 3584, 8192], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # Flashattn	fp16				
####    # Flash attn的input=[[1, 8, 65536, 128], [1, 1, 65536, 128], [1, 1, 65536, 128]]
####    # Flash attn的output=[1, 8, 65536, 128]
####    CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128], False], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [1*8192*1, 8192],    None, None, "llama3-70b-64K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("add",       [1, 8192, 1, 8192],  None, None, "llama3-70b-64K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 65536, 1, 3584], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 65536, 1, 3584], None, None, "llama3-70b-64K", ["train"], enable_cache=False, dtype=DType.FP16),
####
####    # llama3-405b-train	32K
####    CaseInfo("linear", [1, 32768, 16384, 1024], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 16384, 64],   None, None, "llama3-405b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 1024, 16384], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 16384, 3328], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32768, 3328, 16384], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # Flashattn	fp16 shape重复不给了
####
####    CaseInfo("layernorm", [1*2048*1, 16384],   None, None, "llama3-405b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("add",       [1, 2048, 1, 16384], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 32768, 1, 3328], None, None, "llama3-405b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 32768,1, 3328],  None, None, "llama3-405b-32K", ["train"], enable_cache=False, dtype=DType.FP16),
####
####    #---------------------------------------------------
####    # inference
####    #---------------------------------------------------
####    # llama3-70b-inference, 32K-prefill
####    CaseInfo("linear", [4, 24576, 8192, 1024],  None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 24576, 8192, 128],   None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 24576, 1024, 8192],  None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 24576, 8192, 28672], None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 24576, 28672, 8192], None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # Flashattn	fp16				
####    # Flash attn的input=[[4, 64, 24576, 128], [1, 1, 24576, 128],[1, 1, 24576, 128]]
####    # Flash attn的output=[4, 64, 24576, 128]
####    CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128], False], None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####						
####    CaseInfo("layernorm", [4*24576*1, 8192],    None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),     # 应该用layernorm
####    CaseInfo("add",       [4, 24576, 1, 8192],  None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [4, 24576, 1, 28672], None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [4, 24576, 1, 28672], None, None, "llama3-70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####						
####    # llama3-70b-inference, 32K-decode
####    CaseInfo("linear", [4, 1, 8192, 1024],      None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 1, 8192, 128],       None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 1, 1024, 8192],      None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 1, 8192, 28672],     None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [4, 1, 28672, 8192],     None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # Flashattn	fp16				
####    # Flash attn的input=[[4, 64, 1, 128], [4, 8, 28672, 128],[4,8, 28672, 128]]
####    # Flash attn的output=[4, 64, 1, 128] 
####    CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], False], None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [4*1*1, 8192],        None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("add",       [4, 1, 1, 8192],      None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [4, 1, 1, 28672],     None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [4, 1, 1, 28672],     None, None, "llama3-70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####
####    # mixtral_8x70b inference, 32K-prefill
####    CaseInfo("linear", [12, 24576, 8192, 2560], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [12, 24576, 2048, 8192], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 294912, 8192, 7168], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 294912, 8192, 8],    None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 294912, 7168, 8192], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # pagedattention	# 应该用paged attention
####    # paged attn的input=[[12, 16, 24576, 128], [12, 2, 24576, 128], [12, 2, 24576, 128]]
####    # paged attn的output=[12, 16, 24576, 128]
####    CaseInfo("sdpa", [[12, 16, 24576, 128], [12, 2, 24576, 128], False], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [12*24576*1, 8192],   None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 294912, 1, 7168], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 294912, 1, 7168], None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("softmax",	  [1 * 294912 * 1, 8],  None, None, "mixtral_8x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP32),  # [1, 294912, 1, 8] -> [batch_size, reduce维度]
####
####    # mixtral_8x70b inference 32K-decode
####    CaseInfo("linear", [12, 1, 8192, 2560],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [12, 1, 2048, 8192],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 12, 8192, 7168],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 12, 8192, 8],        None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 12, 7168, 8192],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # pagedattention					
####    # paged attn的input=[[12, 16, 1, 128], [12,2, 28672, 128], [12, 2, 28672, 128]]
####    # paged attn的output=[12, 16, 1, 128] 
####    CaseInfo("sdpa", [[12, 16, 1, 128], [12,2, 28672, 128], False], None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [12*1*1, 8192],       None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 12, 1, 7168],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 12, 1, 7168],     None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("softmax",   [1 * 12 * 1, 7168],   None, None, "mixtral_8x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP32),
####    
####    # mixtral_32x70b inference 32K-prefill
####    CaseInfo("linear", [32, 24576, 8192, 640],  None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [32, 24576, 512, 8192],  None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 786432, 8192, 1792], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 786432, 8192, 32],   None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 786432, 1792, 8192], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # pagedattention					
####    # paged attn的input=[[32, 4, 24576, 128], [32, 1, 24576, 128], [32,1, 24576, 128]]
####    # paged attn的output=[32, 4, 24576, 128] 
####    CaseInfo("sdpa", [[32, 4, 24576, 128], [32, 1, 24576, 128], False], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####
####    CaseInfo("layernorm", [32*24576*1, 8192],   None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 786432, 1, 1792], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 786432, 1, 1792], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("softmax",   [1 * 786432 * 1, 32], None, None, "mixtral_32x70b-32K", ["inference prefill"], enable_cache=False, dtype=DType.FP32),
####
####    # mixtral_32x70b inference 32K-decode
####    CaseInfo("linear", [32, 1, 8192, 640],      None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [32, 1, 512, 8192],      None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32, 8192, 32],       None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32, 8192, 1792],     None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####    CaseInfo("linear", [1, 32, 1792, 8192],     None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dataflow_config=dsm_shared_gemm, dtype=DType.FP16),
####
####    # pagedattention					
####    # paged attn的input=[[32, 4, 1, 128], [32, 1, 28672, 128],[32,1, 28672, 128]]
####    # paged attn的output=[32, 4, 1, 128]
####    CaseInfo("sdpa", [[32, 4, 1, 128],[32, 1, 28672, 128], False], None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    
####    CaseInfo("layernorm", [32*1*1, 8192],       None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("silu",      [1, 32, 1, 1792],     None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("mul",       [1, 32, 1, 1792],     None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP16),
####    CaseInfo("softmax",   [1 * 32 * 1, 32],     None, None, "mixtral_32x70b-32K", ["inference generate"], enable_cache=False, dtype=DType.FP32),
]

config_list = [
   #"libra_1DIE_3.2TB_24SIP_256OST.yaml",
   #"libra_v1.0_24SIP_3.2TB_7MB.yaml",
   #"eltanin_v0.3_64SIP_12TB_4MB.yaml",
   #"eltanin_v0.6_48SIP_1.55GHz_512KB.yaml",
   #"eltanin_v0.6_48SIP_1.55GHz_512KB_1.6GHz.yaml",
   #"eltanin_v0.6_48SIP_1.55GHz_512KB_2WR.yaml",
   #"eltanin_D60-A.yaml",
   #"eltanin_D60-B.yaml",
   #"eltanin_D80-A.yaml",
   #"eltanin_D80-B.yaml",
   #"eltanin_D90-A.yaml",
   #"eltanin_D90-B.yaml",
   ##"libra_1DIE_3.2TB_24SIP_256OST.yaml",
   ##"eltanin_1DIE_5.4TB_48SIP_384OST.yaml",
   #"eltanin_2DIE_4TB_64SIP_256OST.yaml",
   #"eltanin_2DIE_4TB_64SIP_384OST.yaml",
   #"eltanin_2DIE_6TB_64SIP_256OST.yaml",
   ##"eltanin_2DIE_6TB_64SIP_384OST.yaml",
   ##"eltanin_2DIE_6TB_64SIP_512OST.yaml",
   #"eltanin_2DIE_8TB_64SIP_256OST.yaml",
   #"eltanin_2DIE_8TB_64SIP_384OST.yaml",
   ##"eltanin_v0.7_2DIE_D90-A_DSM_Shared_256OST.yaml",
   ##"eltanin_v0.7_2DIE_D90-A_DSM_Shared_384OST.yaml",
   ##"eltanin_v0.7_2DIE_D90-A_DSM_Shared_512OST.yaml",
   #"eltanin_v0.7_2DIE_D90-A_DSM_Local_256OST.yaml",
   #"eltanin_v0.7_2DIE_D90-A_DSM_Local_384OST.yaml",
   #"eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml",

   "libra_1DIE_3.2TB_24SIP_256OST.yaml",
#   "eltanin_v0.8_PlanA_D90-A_DSM_Local_512OST_LLC_2MB.yaml",
#    "eltanin_v0.8_PlanA_D90-A_DSM_Shared2_512OST_LLC_2MB.yaml",
#    "eltanin_v0.8_PlanA_D90-A_DSM_Shared4_512OST_LLC_2MB.yaml",
#   "eltanin_v0.8_PlanB_D90-A_DSM_Local_512OST_LLC_3MB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_512OST_LLC_3MB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_512OST_LLC_3MB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_512OST_LLC_4MB.yaml",

#   "eltanin_v0.8_PlanB_D90-A_DSM_Local_72SIP_D2D_100GB.yaml",
#   "eltanin_v0.8_PlanB_D90-A_DSM_Local_72SIP_D2D_90GB.yaml",
#   "eltanin_v0.8_PlanB_D90-A_DSM_Local_80SIP_D2D_100GB.yaml",
#   "eltanin_v0.8_PlanB_D90-A_DSM_Local_80SIP_D2D_90GB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_72SIP_D2D_100GB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_72SIP_D2D_90GB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_80SIP_D2D_100GB.yaml",
#    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_80SIP_D2D_90GB.yaml",
]
# fmt: on


def get_test_case():
    for config in config_list:
        for param in op_shapes_expected_res:
            if not param.fun:
                extend_case = CaseInfo(**param.__dict__)
                extend_case.config = config
                yield extend_case
            else:
                for shape in param.fun(param.shape):
                    extend_case = CaseInfo(**param.__dict__)
                    extend_case.config = config
                    extend_case.shape = shape
                    yield extend_case


@pytest.mark.timeout(30*60)  # timeout 30mins
@pytest.mark.arch_libra
@pytest.mark.parametrize("case_info", get_test_case())
def test_end2end_solution(case_info: CaseInfo, outdir, force_rerun, rank_id, device):
    case_info.outdir = outdir
    case_info.do_sim(force_rerun, rank_id, device)
