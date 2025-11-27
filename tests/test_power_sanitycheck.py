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


def power_scale_bs(shape, power_list):
    for pow in power_list:
        yield [2**pow, *shape[1:]]


def step_n_2048_256_4096(shape):
    for n in range(2048, 4096+256, 256):
        yield [*shape[0:3], n]


def step_k_2048_256_4096(shape):
    for k in range(2048, 4096+256, 256):
        yield [*shape[0:2], k, shape[3]]

# === B M K N


def step_m_1_8192(shape):
    for m in [1, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3073, 4096, 5120, 6144, 7168, 8192]:
        yield [*shape[0:1], m, *shape[2:4]]


def step_k_1_8192(shape):
    for k in [1, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3073, 4096, 5120, 6144, 7168, 8192]:
        yield [*shape[0:2], k, shape[3]]


def step_n_1_8192(shape):
    for n in [1, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3073, 4096, 5120, 6144, 7168, 8192]:
        yield [*shape[0:3], n]


pow0_256 = functools.partial(
    power_scale_bs, power_list=list(range(9)))

dsm_shared_gemm = {"bench_gemm_op_version": 2}
dsm_local_gemm = {"bench_gemm_op_version": 4}


# ===quant config
fp8_quant_config = {"bench_gemm_quant_type": "Wf8t_Af8t"}
fp4_quant_config = {"bench_gemm_quant_type": "Wf4t_Af8t"}

new_dsm_shared_gemm = {"bench_gemm_op_version": 5}
new_dsm_local_gemm = {"bench_gemm_op_version": 6}

fp8_dsm_shared_gemm = {**new_dsm_shared_gemm, **fp8_quant_config}
fp4_dsm_shared_gemm = {**new_dsm_shared_gemm, **fp4_quant_config}

fp8_dsm_local_gemm = {**new_dsm_local_gemm, **fp8_quant_config}
fp4_dsm_local_gemm = {**new_dsm_local_gemm, **fp4_quant_config}

# == config dsm type and op data type
op_gemm_dtype = DType.FP16
dsm_config_type = dsm_local_gemm
dsm_local_config_type = new_dsm_local_gemm
dsm_share_config_type = new_dsm_shared_gemm

# ================================================================================
# =========================== DSM Local Test =====================================
# ================================================================================
# fmt: off
op_shapes_expected_res_dsm_local = [

    #------------------------------------------------------
    # check M/N/K 
    #------------------------------------------------------
    CaseInfo("linear", [1, 256, 256, 256], None,
            step_m_1_8192, "deepseekv2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 256, 256, 256], None,
            step_k_1_8192, "deepseekv2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 256, 256, 256], None,
            step_n_1_8192, "deepseekv2", ["inference prefill"], enable_cache=False, dataflow_config=dsm_config_type),

    #---------------------------------------------------
    # examples [B, M, K, N]
    #---------------------------------------------------
    CaseInfo("matmul", [16, 4096, 192, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_config_type),
    CaseInfo("matmul", [16, 4096, 4096, 128], None,
            None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_config_type),

    CaseInfo("linear", [128, 4096, 192, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_config_type),
    CaseInfo("linear", [128, 4096, 4096, 128], None,
            None, "deepseekv2", ["train"], enable_cache=False, dataflow_config=dsm_config_type),


# #     # from test
#     CaseInfo("relu", [12 * 4096, 4096], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
#     CaseInfo("sigmoid", [12 * 4096, 4096], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
#     CaseInfo("layernorm", [512, 12288], None,
#             None, "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),

#     # gather             [bs， 次数， 行数， 每行的大小]
#     CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=False, mu=4e-8, dtype=DType.FP32, dataflow_config=dsm_shared_gemm),

    #---------------------------------------------------
    # Flash Attention
    #---------------------------------------------------
    #fro gpt175b with dropout
    CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],True], None, None, 
             "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],True], None, None, 
             "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],True], None, None, 
            "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    # fro gpt175b without dropout
    CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],False], None, None, 
             "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],False], None, None, 
             "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],False], None, None, 
            "gpt175b", ["train"], enable_cache=False, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),





]
op_shapes_expected_res_dsm_local = [

    #------------------------------------------------------
    # check M/N/K 
    #------------------------------------------------------
#     CaseInfo("linear", [1, 256, 256, 256], None,
#             step_m_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
#     CaseInfo("linear", [1, 256, 256, 256], None,
#             step_k_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
#     CaseInfo("linear", [1, 256, 256, 256], None,
#             step_n_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
#     CaseInfo("linear", [1, 128, 128, 128], None,
#             step_m_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
#     CaseInfo("linear", [1, 128, 128, 128], None,
#             step_k_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
#     CaseInfo("linear", [1, 128, 128, 128], None,
#             step_n_1_8192, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    #===K=256, M=128, Shmoo N ==
    CaseInfo("linear", [2, 1024, 256, 128], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

]


op_shapes_expected_res_dsm_local  += [
    #---------------------------------------------------
    # examples [B, M, K, N]
    #--------------------------------------------------- 
    #===== FIX Batch=32,K=128, Shmoo M=[256,512,1024]
    CaseInfo("linear", [32, 128, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 128, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 128, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [32, 256, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [32, 512, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 512, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 512, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [32, 1024, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 1024, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 1024, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    #===== FIX Batch=32,K=256, Shmoo M=[256,512,1024]
    CaseInfo("linear", [32, 256, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [32, 512, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 512, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 512, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [32, 1024, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 1024, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 1024, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

   # === FIX Batch=64,K=128, Shmoo M=[128,256,512,1024]
    CaseInfo("linear", [64, 128, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 128, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 128, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 256, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 256, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 256, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 512, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 512, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 512, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 1024, 128, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 1024, 128, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 1024, 128, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
   # === FIX Batch=64,K=256, Shmoo M=[128,256,512,1024]
    CaseInfo("linear", [64, 128, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 128, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 128, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 256, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 256, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 256, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 512, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 512, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 512, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),

    CaseInfo("linear", [64, 1024, 256, 256], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 1024, 256, 512], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),
    CaseInfo("linear", [64, 1024, 256, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dataflow_config=dsm_config_type),



]

op_shapes_expected_res_dsm_local  = [
    #---------------------------------------------------
    # examples [B, M, K, N]
    #--------------------------------------------------- 
    CaseInfo("linear", [4, 384, 1024, 4096], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [4, 384, 1024, 2048], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [4, 384, 1024, 1024], None,
            None, "perf-per-watt", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),

    CaseInfo("linear", [1, 4096, 3072, 5120], None,
            None, "train", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 4096, 512, 32768], None,
            None, "train", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype,dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 512, 5120, 3072], None,
            None, "train", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 4096, 5120, 1536], None,
            None, "train", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 4096, 5120, 3072], None,
            None, "train", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),


]



#==================================== All OPs ====
# fmt: off
op_shapes_expected_res = [
    #---------------------------------------------------
    # examples [B, M, K, N]
    #---------------------------------------------------
    # CaseInfo("matmul", [16, 4096, 192, 4096], None,
    #         None, "dev", ["inference generate"], enable_cache=True),
    # CaseInfo("matmul", [16, 4096, 192, 4096], None,
    #         None, "dev", ["inference generate"], enable_cache=True),
    # CaseInfo("linear", [1,  4096, 12288, 1536],
    #              596.10e-6, None, None, ["ci case", "libra_zebu"]),
    # CaseInfo("linear", [1,  4096, 12288, 6144],
    #              2346.20e-6, None, None, ["ci case", "libra_zebu"]),
    # CaseInfo("matmul", [12, 4096,  128, 4096],
    #              220.90e-6, None, None, ["ci case", "libra_zebu"]),
    # # IO bound?
    # # CaseInfo(None, [1, 64, 4096, 65024], None, None, None, ["IO bound"]),

    #---------------------------------------------------
    # train
    #---------------------------------------------------
    # from deepseekv2-TP8
    CaseInfo("linear", [1, 512, 5120, 192], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 1536, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 5120, 72], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 512, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 4096, 192, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 4096, 4096, 128], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 5120, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 512, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 512, 5120, 160], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),

    # from deepseekv2-TP1
    CaseInfo("linear", [1, 4096, 5120, 1536], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 1536, 24576], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 5120, 576], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 512, 32768], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [128, 4096, 192, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [128, 4096, 4096, 128], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 5120, 24576], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 5120, 160], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 16384, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 4096, 3072, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),


    #---------------------------------------------------
    # inference ????prefill和generate的准确性????
    #---------------------------------------------------
    ## deepseekv2-TP8
    CaseInfo("linear", [16, 1, 5120, 192], None,
            pow0_256, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [16, 1, 1536, 3072], None,
            pow0_256, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [16, 1, 5120, 72], None,
            pow0_256, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [16, 1, 512, 4096], None,
            pow0_256, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [256, 1, 192, 2048], None,
            step_n_2048_256_4096, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [256, 1, 2048, 128], None,
            step_k_2048_256_4096, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [16, 1, 5120, 3072], None,
            None, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [16, 1, 5120, 160], None,
            None, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),

    ## from deepseekv2
    CaseInfo("matmul", [1, 4096, 1, 3072], None,
            None, "deepseekv2", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),


    # from LAMMA2 generate
    CaseInfo("linear", [1, 129, 12288, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 129, 4096, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 129, 22016, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 129, 4096, 11008], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 1, 12288, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 1, 4096, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 1, 22016, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("matmul", [16, 1, 4096, 11008], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),

    #---------------------------------------------------
    # Cape微信模型
    #---------------------------------------------------
    CaseInfo("linear", [1, 96, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 96, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 96, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 96, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 128, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 128, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 128, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 128, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 192, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 192, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 192, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 192, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 256, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 256, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 256, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 256, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 96, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 96, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 96, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 96, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 128, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 128, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 128, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 128, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 192, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 192, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 192, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 192, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 256, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 256, 1024, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 256, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [32, 256, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),

    #---------------------------------------------------
    # from LLAMA3-70b-train
    #---------------------------------------------------
    CaseInfo("linear", [1, 32768, 8192, 1024], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 8192, 128], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 1024, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 8192, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 3584, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),

    # CaseInfo("rms_norm	fp16	1	4096	1	8192    ###
    CaseInfo("add", [1, 4096, 1, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [1, 32768, 1, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [1, 32768, 1, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),    

    CaseInfo("linear", [1, 65536, 8192, 1024], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 65536, 8192, 128], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 65536, 1024, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 65536, 8192, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 65536, 3584, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),

    # CaseInfo("rms_norm	fp16	1	8192	1	8192###
    CaseInfo("add", [1, 8192, 1, 8192], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [1, 65536, 1, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [1, 65536, 1, 3584], None,
            None, "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype),

    # from LLAM3-405b-train
    CaseInfo("linear", [1, 32768, 16384, 1024], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 16384, 64], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 1024,  16384], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 16384, 3328], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [1, 32768, 3328,  16384], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dataflow_config=dsm_local_gemm),

     # CaseInfo("Flashattn	fp16 #重复

    # CaseInfo("rms_norm	fp16	1	2048	1	16384#
    CaseInfo("add", [1, 2048, 1, 16384], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [1, 32768, 1, 3328], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [1, 32768, 1, 3328], None,
            None, "LLAMA3-405B", ["train"], enable_cache=True, dtype=op_gemm_dtype),

    # from LLAM3-70b-inference, 32k-prefill
    CaseInfo("linear", [4, 24576, 8192,  1024], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 24576, 8192,  128], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 24576, 1024,  8192], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 24576, 8192,  28672], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 24576, 28672, 8192], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dataflow_config=dsm_local_gemm),

    # CaseInfo("rms_norm	fp16	4	24576	1	8192##
    CaseInfo("add", [4, 24576, 1, 8192], None,
            None, "LLAMA3-70B", ["trinference prefillain"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [4, 24576, 1, 28672], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [4, 24576, 1, 28672], None,
            None, "LLAMA3-70B", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype),

    # from LLAM3-70b-inference, 32k-decode
    CaseInfo("linear", [4, 1, 8192,  1024], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 1, 8192,  128], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 1, 1024,  8192], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 1, 8192,  28672], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),
    CaseInfo("linear", [4, 1, 28672, 8192], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dataflow_config=dsm_local_gemm),


    # CaseInfo("rms_norm	fp16	4	1	1	8192 #???
    CaseInfo("add", [4, 1, 1, 8192], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [4, 1, 1, 28672], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [4, 1, 1, 28672], None,
            None, "LLAMA3-70B", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype),

    #---------------------------------------------------
    # 1D OPs, date type according to OP excel
    #---------------------------------------------------
    # from deepseekv2
    CaseInfo("add", [1, 4096, 1, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("add", [1, 512, 1, 12288], None,
            None, "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("mul", [1, 4096, 1, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True),
    CaseInfo("mul", [1, 4096, 1, 1536], None,
            None, "deepseekv2", ["train"], enable_cache=True),
    CaseInfo("softmax", [1 * 4096 * 1, 160], None,      # [1, 4096, 1, 160] -> [batch_size, reduce维度]
            None, "deepseekv2", ["train"], enable_cache=True, dtype=DType.FP32),
    CaseInfo("softmax", [128 * 4096 * 1, 4096], None,   # [128, 4096, 1, 4096]
            None, "deepseekv2", ["train"], enable_cache=True, dtype=DType.FP32),
    CaseInfo("silu", [1, 4096, 1, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("silu", [1, 4096, 1, 1536], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("gelu", [1, 4096, 1, 6144], None,
            None, "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype),

    # from test
    CaseInfo("relu", [12 * 4096, 4096], None,
            None, "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("sigmoid", [12 * 4096, 4096], None,
            None, "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype),
    CaseInfo("layernorm", [512, 12288], None,
            None, "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype),

    # gather             [bs， 次数， 行数， 每行的大小]
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=1,    dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=1e-6, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=5e-7, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=1e-7, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=6e-8, dtype=DType.FP32),
    CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=4e-8, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=2e-8, dtype=DType.FP32),
    #CaseInfo("gather", [1, 4096, 51200, 12288], None, None, "gpt175b", ["train"], enable_cache=True, mu=1e-8, dtype=DType.FP32),

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
    #       None, "deepseekv2", ["train"], enable_cache=True, dataflow_config=dsm_shared_gemm),

    #---------------------------------------------------
    # Flash Attention
    #---------------------------------------------------
#     # fro gpt175b with dropout
#     CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],True], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],True], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],True], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     # fro gpt175b without dropout
#     CaseInfo("sdpa", [[1, 48, 4096, 128], [1, 48, 4096, 128],False], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 12, 16384, 128], [1, 12, 16384, 128],False], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 6, 32768, 128], [1, 6, 32768, 128],False], None, None, 
#             "gpt175b", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # new ops with dropout
#     CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], True], None, None,
#             "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128],True], None, None,
#             "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128] ,True], None, None,
#             "LLAMA3-70B", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], True], None, None,
#             "LLAMA3-70B", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # new ops without dropout
#     CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 1, 32768, 128], False], None, None,
#             "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 8, 65536, 128], [1, 1, 65536, 128], False], None, None,
#             "LLAMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[4, 64, 24576, 128], [1, 1, 24576, 128] ,False], None, None,
#             "LLAMA3-70B", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[4, 64, 1, 128], [4, 8, 28672, 128], False], None, None,
#             "LLAMA3-70B", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for llama3_70b, train with dropout
#     CaseInfo("sdpa", [[1, 64, 2048, 128], [1, 64, 2048, 128], True], None, None, 
#             "llama3", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 8, 32768, 128], True], None, None, 
#             "llama3", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     # for llama3_70b, train without dropout
#     CaseInfo("sdpa", [[1, 64, 2048, 128], [1, 64, 2048, 128], False], None, None, 
#             "llama3", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 8, 32768, 128], [1, 8, 32768, 128], False], None, None, 
#             "llama3", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for mixtral8x70b, train with dropout
#     CaseInfo("sdpa", [[1, 64, 1024, 128], [1, 64, 1024, 128], True], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 16, 4096, 128], [1, 16, 4096, 128], True], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 4, 32768, 128], [1, 4, 32768, 128], True], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     # for mixtral8x70b, train without dropout
#     CaseInfo("sdpa", [[1, 64, 1024, 128], [1, 64, 1024, 128], False], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 16, 4096, 128], [1, 16, 4096, 128], False], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     CaseInfo("sdpa", [[1, 4, 32768, 128], [1, 4, 32768, 128], False], None, None, 
#             "mixtral", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for deepseekv2, train with dropout
#     CaseInfo("sdpa", [[1, 128, 4096, 192], [1, 128, 4096, 192], True], None, None, 
#             "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     # for deepseekv2, train without dropout
#     CaseInfo("sdpa", [[1, 128, 4096, 192], [1, 128, 4096, 192], False], None, None, 
#             "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for hydit, train with dropout
#     CaseInfo("sdpa", [[1, 16, 8192, 88], [1, 16, 8192, 88], True], None, None, 
#             "hydit", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#     # for hydit, train without dropout
#     CaseInfo("sdpa", [[1, 16, 8192, 88], [1, 16, 8192, 88], False], None, None, 
#             "hydit", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),


#     # for gpt175b, inference
#     CaseInfo("sdpa", [[1, 96, 3072, 128], [1, 8, 3072, 128], False], None, None, 
#             "gpt175b", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 96, 1, 128], [1, 8, 3584, 128], False], None, None, 
#             "gpt175b", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 96, 24576, 128], [1, 8, 24576, 128], False], None, None, 
#             "gpt175b", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 96, 1, 128], [1, 8, 28672, 128], False], None, None, 
#             "gpt175b", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for llama3, inference
#     CaseInfo("sdpa", [[1, 64, 3072, 128], [1, 8, 3072, 128], False], None, None, 
#             "llama3", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 64, 1, 128], [1, 8, 3584, 128], False], None, None, 
#             "llama3", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 64, 24576, 128], [1, 8, 24576, 128], False], None, None, 
#             "llama3", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     CaseInfo("sdpa", [[1, 64, 1, 128], [1, 8, 28672, 128], False], None, None, 
#             "llama3", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),

#     # for hydit, inference
#     CaseInfo("sdpa", [[1, 16, 6144, 88], [1, 8, 6144, 88], False], None, None, 
#             "hydit", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_gemm),
#    #============================== end of test list

]


#========== check  power  >  tdp 1200w ================
op_shapes_expected_res_dsm_local  = [
    #=== K Vs. llc hit rate
    CaseInfo("linear", [32, 256, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 256, 768], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 256, 512], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 256, 256, 256], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),


    CaseInfo("linear", [32, 192, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 128, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [32, 96, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),


    # from LAMMA2 generate
    CaseInfo("linear", [1, 129, 22016, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 129, 12288, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 129, 4096, 4096], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 129, 4096, 11008], None,
            None, "LAMMA2", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    # LLAMA3-70B
    CaseInfo("linear", [4, 24576, 8192, 128], None,
            None, "LAMMA3-70B", ["inference prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),    
    CaseInfo("linear", [1, 32768, 8192, 128], None,
            None, "LAMMA3-70B", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),    

    # deepseekv2
    CaseInfo("matmul", [16, 4096, 192, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("matmul", [16, 4096, 4096, 128], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 4096, 5120, 1536], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),
    CaseInfo("linear", [1, 4096, 3072, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),     
    CaseInfo("linear", [1, 4096, 512, 32768], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),    
    CaseInfo("linear", [1, 512, 5120, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),   
    CaseInfo("linear", [1, 4096, 5120, 576], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),   
    CaseInfo("linear", [1, 512, 1536, 3072], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type),   
    CaseInfo("linear", [1, 4096, 5120, 160], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type), 
    CaseInfo("linear", [1, 512, 512, 4096], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type), 
    CaseInfo("linear", [256, 1, 1536, 3072], None,
            None, "deepseekv2", ["inference_prefill"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_config_type), 


]

def get_test_case(config_list,op_shapes_expected_res):
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

#===========================
#========== check Sanity ================
op_shapes_expected_res_dsm_local  = [
    CaseInfo("linear", [1, 4096, 3072, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_config_type),     

    CaseInfo("linear", [32, 192, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_local_config_type), 
]

#==config
config_list_dsm_local = [

     #"eltanin_v0.7_2DIE_D90-A_DSM_Local_1.6G.yaml",
     #"eltanin_v0.7_2DIE_D90-A_DSM_Local_1.5G.yaml",
     #"eltanin_v0.7_2DIE_D90-A_DSM_Local_1.4G.yaml",
    #"eltanin_v0.7_2DIE_D90-A_DSM_Local_Cache_1.6G.yaml",
     # "eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml",
     #"eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST_VF1.4G.yaml",
     "eltanin_v0.8_PlanA_D90-A_DSM_Local_512OST_LLC_2MB.yaml",
     "eltanin_v0.8_PlanB_D90-A_DSM_Local_512OST_LLC_3MB.yaml",
     "eltanin_v0.8_PlanB_D90-A_DSM_Local_72SIP_D2D_100GB.yaml",
     "eltanin_v0.8_PlanB_D90-A_DSM_Local_72SIP_D2D_90GB.yaml",
     "eltanin_v0.8_PlanB_D90-A_DSM_Local_80SIP_D2D_100GB.yaml",
     "eltanin_v0.8_PlanB_D90-A_DSM_Local_80SIP_D2D_90GB.yaml",
]

# ========= RUN TEST ===================================
# fmt: on
def get_test_case(config_list,op_shapes_expected_res):
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

#===== Define Test here
@pytest.mark.timeout(200 * 60)  # timeout 90mins
@pytest.mark.parametrize("case_info", get_test_case(config_list_dsm_local, op_shapes_expected_res_dsm_local))
#@pytest.mark.parametrize("case_info", get_test_case(config_list_dsm_local, op_shapes_expected_res))
def test_power_sanitycheck_newdataflow_FP16(case_info: CaseInfo, outdir, force_rerun, rank_id, device):
    case_info.outdir = outdir
    case_info.do_sim(force_rerun, rank_id, device)

#========== check Sanity ================
op_shapes_expected_res_dsm_share  = [
    CaseInfo("linear", [1, 4096, 3072, 5120], None,
            None, "deepseekv2", ["train"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_share_config_type),     

    CaseInfo("linear", [32, 192, 256, 1024], None,
            None, "WeChat", ["inference generate"], enable_cache=True, dtype=op_gemm_dtype, dataflow_config=dsm_share_config_type), 
]
#==config
config_list_dsm_share = [ 
    "eltanin_v0.8_PlanA_D90-A_DSM_Shared2_512OST_LLC_2MB.yaml",

    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_512OST_LLC_3MB.yaml",
    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_512OST_LLC_3MB.yaml",

    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_72SIP_D2D_100GB.yaml",
    "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_72SIP_D2D_90GB.yaml",

    "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_80SIP_D2D_100GB.yaml",
    "eeltanin_v0.8_PlanB_D90-A_DSM_Shared4_80SIP_D2D_90GB.yaml",


]

#========= 

# fmt: on
# ========= RUN TEST ===================================

#===== Define Test here
@pytest.mark.timeout(200 * 60)  # timeout 90mins
@pytest.mark.parametrize("case_info", get_test_case(config_list_dsm_share, op_shapes_expected_res_dsm_share))
#@pytest.mark.parametrize("case_info", get_test_case(config_list_dsm_local, op_shapes_expected_res))
def test_power_sanitycheck_newdataflow_FP16(case_info: CaseInfo, outdir, force_rerun, rank_id, device):
    case_info.outdir = outdir
    case_info.do_sim(force_rerun, rank_id, device)

