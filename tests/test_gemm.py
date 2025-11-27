import pytest

from nova_platform.base_model import DType

import logging
from tests.test_base import CaseInfo


logger = logging.getLogger(__name__)

# gemm_shapes = [
#     [1, 1024, 12288, 1536],
#     [1, 2048, 12288, 1536],
#     [1, 4096, 12288, 1536],
#     # [1, 32000, 12288, 1536],
#     [1, 1024, 4096, 1024],
#     [1, 2048, 4096, 1024],
#     [1, 4096, 4096, 1024],
#     # [1, 32000, 4096, 1024],
#     [1, 512, 5120, 192],
#     [1, 512, 1536, 3072],
#     [1, 512, 512, 4096],
#     # [16, 4096, 4096, 128],
#     [1, 512, 5120, 3072],
#     [1, 512, 5120, 160],
#     [16, 1, 4096, 4096],
# ]

gemm_shapes = [
    [1, 512, 5120, 192],
    # [1, 512, 1536, 3072],
    # [1, 512, 5120, 72],
    # [1, 512, 512, 4096],
    # [16, 4096, 192, 4096],
    # [16, 4096, 4096, 128],
    # [1, 512, 5120, 3072],
    # [1, 512, 5120, 160],
    # [1, 4096, 3072, 5120],
    # [1, 16, 5120, 72],
    # [1, 16, 5120, 192],
    # [1, 16, 1536, 3072],
    # [1, 129, 4096, 4096],
    # [1, 256, 256, 1024],
    # [1, 256, 256, 768],
    # [1, 96, 256, 1024],
    # [1, 128, 5120, 72],
]


# gemm_shapes = []
# for i in range(5, 14):
#     m = pow(2, i)
#     gemm_shapes.append([1, m, 8192, 1024])
#     gemm_shapes.append([1, m, 8192, 128])
#     gemm_shapes.append([1, m, 1024, 8192])
#     # gemm_shapes.append([1, m, 8192, 28672])
#     # gemm_shapes.append([1, m, 28672, 8192])
#     gemm_shapes.append([1, m, 8192, 2560])
#     gemm_shapes.append([1, m, 2048, 8192])
#     # gemm_shapes.append([1, m, 8192, 7168])
#     gemm_shapes.append([1, m, 8192, 8])
#     # gemm_shapes.append([1, m, 7168, 8192])

dtype = [DType.FP16]

config = [
    "libra_1DIE_3.2TB_24SIP_256OST.yaml",
    # "eltanin_v0.8_PlanA_D90-A_DSM_Shared2_512OST_LLC_2MB.yaml",
    # "eltanin_v0.8_PlanA_D90-A_DSM_Shared4_512OST_LLC_2MB.yaml",
    # "eltanin_v0.8_PlanB_D90-A_DSM_Shared2_72SIP_D2D_100GB.yaml",
    # "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_72SIP_D2D_100GB.yaml",
    # "eltanin_v0.8_PlanB_D90-A_DSM_Shared4_80SIP_D2D_100GB.yaml",
]


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
def test_gemm_benchmark_1(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gemm",
            "dtype": dtype,
            "shape": shape,
            "dataflow_config": {"bench_gemm_op_version": 1},
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
def test_gemm_benchmark_2(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gemm",
            "dtype": dtype,
            "shape": shape,
            "dataflow_config": {"bench_gemm_op_version": 2},
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
def test_gemm_benchmark_4(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gemm",
            "dtype": dtype,
            "shape": shape,
            "dataflow_config": {"bench_gemm_op_version": 4},
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
def test_gemm_benchmark_5(outdir, force_rerun, rank_id, device, config, dtype, shape):
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": "gemm",
            "dtype": dtype,
            "shape": shape,
            "dataflow_config": {"bench_gemm_op_version": 5},
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)

quant_config = ["No_Quant", "Wf4g_Af8t", "Wf8t_Af8t"]


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
@pytest.mark.parametrize("quant_type", quant_config)
def test_quant_gemm_benchmark(outdir, force_rerun, rank_id, device, config, dtype, shape, quant_type):
    optype = "gemm.local" if "Local" in config else "gemm.shared"
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": optype,
            "dtype": dtype,
            "shape": shape,
            "dataflow_config": {"bench_gemm_quant_type": quant_type},
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)


@pytest.mark.parametrize("config", config)
# @pytest.mark.timeout(90 * 60)  # timeout 90mins
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("shape", gemm_shapes)
def test_gemm_benchmark_analysis(outdir, force_rerun, rank_id, device, config, dtype, shape):
    optype = "gemm.local" if "Local" in config else "gemm.shared"
    case_info = CaseInfo(
        **{
            "outdir": outdir,
            "config": config,
            "optype": optype,
            "dtype": dtype,
            "shape": shape,
        }
    )
    case_info.do_sim(force_rerun, rank_id, device)
