import pytest
from bossa_nova.base_model import DType
from tests.test_base import TOPO, CaseInfo

# [1, 262144, 2560, 2560]
dsm_shared_gemm = {"bench_gemm_op_version": 1}
dsm_local_gemm = {"bench_gemm_op_version": 4}
case_list = [
    # CaseInfo("matmul", [1, 4096, 12288, 1536], 596.10e-6, None,
    #          "deepseekv2", ["train"],    config="libra_1DIE_3.2TB_24SIP_256OST.yaml"),
    # CaseInfo("matmul", [1, 4096, 12288, 1536], None, None,
    #  "deepseekv2", ["train"],    config="eltanin_D90-A.yaml"),
    # CaseInfo("linear", [1, 512, 5120, 192], None,
    #          None, "deepseekv2", ["train"], config="libra_1DIE_3.2TB_24SIP_256OST.yaml"),  # enable_cache=False, config="eltanin_D90-A.yaml"),
    # CaseInfo("matmul", [12, 4096, 128, 4096],
    #          config="libra_1DIE_3.2TB_24SIP_256OST.yaml")
    # CaseInfo("add", [1, 4096, 1, 5120], None,
    #          None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16,
    #          config="libra_1DIE_3.2TB_24SIP_256OST.yaml"),
    # CaseInfo("add", [1, 4096, 1, 5120], None,
    #          None, "deepseekv2", ["train"], enable_cache=False, dtype=DType.FP16,
    #          config="libra_1DIE_3.2TB_24SIP_256OST_dual_die.yaml")
    # CaseInfo("matmul", [12, 4096, 128, 4096],
    #          config="libra_1DIE_3.2TB_24SIP_256OST.yaml"),
    # CaseInfo("matmul", [12, 4096, 128, 4096],
    #          config="libra_1DIE_3.2TB_24SIP_256OST_dual_die.yaml"),
    # CaseInfo("matmul", [1, 512, 5120, 192], dtype=DType.FP8,
    #          config="eltanin_D90-B.yaml"),


    # CaseInfo("matmul", [1, 4096, 192, 4096],
    #  config="libra_1DIE_3.2TB_24SIP_256OST.yaml"),
    # CaseInfo("matmul", [1, 4096, 192, 4096],
    #          config="eltanin_v0.7_2DIE_D90-A_DSM_Local_384OST.yaml", enable_cache=True),

    # CaseInfo("add", [1, 4096, 1, 5120],
    #          config="eltanin_v0.7_2DIE_D90-A_DSM_Local_384OST.yaml"),
    # CaseInfo("matmul", [16, 4096, 192, 4096],
    #          config="eltanin_1DIE_5.4TB_48SIP_256OST.yaml"),
    # CaseInfo("matmul", [16, 4096, 192, 4096],
    #          config="eltanin_2DIE_8TB_64SIP_384OST.yaml"),

    # CaseInfo('silu', [4, 1, 1, 28672],
    #          config='eltanin_v0.7_2DIE_D90-A_DSM_Local_384OST.yaml')
    # CaseInfo('matmul', [16, 1, 12288, 4096], enable_cache=True,
    #          config='eltanin_v0.7_2DIE_D90-A_DSM_Local_384OST.yaml', dataflow_config=dsm_shared_gemm),
    # CaseInfo('matmul', [1, 1, 12288, 4096], enable_cache=True,
    #  config='eltanin_2DIE_8TB_64SIP_384OST.yaml', dataflow_config=dsm_local_gemm)
    # CaseInfo("matmul", [1, 512, 5120, 192], dtype=DType.FP8,
    #          config="eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml", dataflow_config=dsm_local_gemm),

    CaseInfo(
        **{
            "optype": "allreduce",
            "shape": [1, 128, 128, 128],
            # "optype": "gather",
            # "shape": [1, 192, 51200, 12288],
            "dtype": DType.FP16,
            "enable_cache": False,
            "config": "eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml",
            "mu": 2e-8
        }
    )
]


@pytest.mark.parametrize("case_info", case_list)
def test_dev4(case_info: CaseInfo, outdir, force_rerun, rank_id, device):

    # case_info = CaseInfo("matmul", [1, 512, 5120, 192], 261.76e-6,
    #                      None, "deepseekv2", ["train"])
    case_info.outdir = outdir
    case_info.topo = TOPO.FULLMESH8
    case_info.do_sim(force_rerun, rank_id, device)
