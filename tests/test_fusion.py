import pytest
import functools
from pathlib import Path
from bossa_nova.base_model import DType
from bossa_nova.config import BossaNovaConfig, TOPO
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.dataflow.action.diag_action import parse_shape_tile_info
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config
from bossa_nova.utils.cuda_utils import get_gpu_count
from bossa_nova.perfetto_protobuf._tgen import TraceGenerator
import logging

from tests.test_base import CaseInfo
from tests.test_base import FusionCaseInfo
from bossa_nova.utils.fusion_utils import get_caseinfo_list, display_fusion_info

logger = logging.getLogger(__name__)



#get fusion caseinfo and address list from optrace file
#TODO: needs to add new optrace file
#-----------------------------------------------------------
# optrace_addr = "tests/resources/gpt175b_inference_tp8_optrace_modified.txt"
# optrace_addr = "tests/resources/small_optrace.txt"
optrace_addr = "tests/resources/simple_optrace.txt"
# optrace_addr = "tests/resources/sgd_syn_optrace.txt"
# optrace_addr = "tests/resources/gpt_tp_train.txt"
#-----------------------------------------------------------
#TODO: needs to set topo
topo_global = TOPO.SUPERNODE8
cases_addr_list, op_info_list = get_caseinfo_list(optrace_addr,topo=topo_global, enable_cache=True)

    #---------------------------------------------------
    # examples [CaseInfo, input_addr_list, output_addr_list]
    #---------------------------------------------------
    # [
    #   [
    #       CaseInfo("add", [1, 4096, 1, 5120], None, 
    #           None, "deepseekv2", ["train"], enable_cache=True, dtype=DType.FP16), 
    #       [input_addr_list], [output_addr_list]
    #   ],

    #   [
    #       CaseInfo("add", [1, 4096, 1, 5120], None,
    #           None, "deepseekv2", ["inference generate"], enable_cache=True, dtype=DType.FP16),
    #       [input_addr_list], [output_addr_list]
    #   ],
    #   ...
    # ]



config_list = [
    "eltanin_v0.7_2DIE_D90-A_DSM_Local_512OST.yaml",
    # "libra_1DIE_3.2TB_24SIP_256OST.yaml"
]


# Fusion cases
def get_test_fusion_case():
    fusion_case_list = []
    for config in config_list:
        fusion_case = FusionCaseInfo(optype="", shape=[], cases=[])
        # may change src of enumerate(CaseInfo_list)
        for idx, param in enumerate(cases_addr_list):
            if not param.fun:
                extend_case = CaseInfo(**param.__dict__)
                extend_case.config = config
                fusion_case.config = config
                fusion_case.add_case(extend_case)
            else:
                for shape in param.fun(param.shape):
                    extend_case = CaseInfo(**param.__dict__)
                    extend_case.config = config
                    extend_case.shape = shape
                    fusion_case.add_case(extend_case)
        fusion_case_list.append(fusion_case)
    return fusion_case_list

@pytest.mark.fusion
@pytest.mark.parametrize("topo", [topo_global])
@pytest.mark.parametrize("fusion_case_info", get_test_fusion_case())
def test_end2end_solution_fusion(fusion_case_info: FusionCaseInfo, outdir, force_rerun, rank_id, device, topo):
    fusion_case_info.outdir = outdir
    fusion_case_info.topo = topo
    display_fusion_info(outdir, op_info_list)
    # fusion_case_info.tgen = TraceGenerator(f"{outdir}/fusion_trace.perfetto-trace")
    fusion_case_info.do_sim(force_rerun, rank_id, device)
