from dataclasses import dataclass, field
from typing import Generator, List, Tuple, Dict
from pprint import pprint

from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
    DataflowActionComputeStat,
)
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.config import BossaNovaConfig
from nova_platform.benchmark.op_base import Workload, Operand, list_product
from nova_platform.benchmark.batch_gemm_local import BatchGemmGridShape, tile_local_gemm_workload, dsm_local_gemm_kernel
from nova_platform.benchmark.batch_gemm_shared import tile_shared_gemm_workload, dsm_shared_gemm_kernel, QuantType

import logging

logger = logging.getLogger(__name__)


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    gemm_shape: Dict[str, int] = field(default_factory=dict)
    grid_shape: List[int] = field(default_factory=list)
    block_shape: List[int] = field(default_factory=list)
    thread_shape: List[int] = field(default_factory=list)
    subthread_shape: List[int] = field(default_factory=list)
    grid_dim: List[int] = field(default_factory=list)
    block_dim: List[int] = field(default_factory=list)
    thread_dim: List[int] = field(default_factory=list)


@dataclass
class CoreCost(BaseCoreStat):
    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)


def action_tensor_to_operand(tensor):

    def reversed_dims(t):
        return tuple(reversed(t))

    return Operand(
        dim=reversed_dims(tensor.dims[:3]),
        addr=tensor.addr,
        bpe=tensor.bpe,
        dim_offset=reversed_dims(tensor.offsets[:3]),
        dim_stride=reversed_dims(tensor.stride_dims[:3]),
    )


@dataclass
class XPUBaseGemmAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_kernel_idx(self):
        return self.data[0], self.data[1]

    def get_chip_workload(self):
        lhs = action_tensor_to_operand(self.inputs[0].tensor[0])
        rhs = action_tensor_to_operand(self.inputs[1].tensor[0])
        res = action_tensor_to_operand(self.outputs[0].tensor[0])
        inputs = [lhs, rhs]
        quant_type = self.dataflow_config.get("bench_gemm_quant_type", QuantType.No_Quant)
        if quant_type == QuantType.Wf4g_Af8t:
            scaling = action_tensor_to_operand(self.inputs[2].tensor[0])
            inputs.append(scaling)
        attr = {"b": lhs.dim[0], "m": lhs.dim[1], "k": lhs.dim[2], "n": rhs.dim[2], "quant_type": quant_type}
        return Workload(
            inputs=inputs,
            outputs=[res],
            attr=attr,
            dtype=self.get_dtype(),
        )

    def get_instruction_info(self, chip_workload: Workload, best_shape: BatchGemmGridShape):
        def _get_shape(dim_stride):
            return {"b": dim_stride[2], "m": dim_stride[1], "n": dim_stride[0], "k": best_shape.calc_ceil_K_l2}

        return InstructionInfo(
            dtype=self.get_dtype(),
            gemm_shape=chip_workload.attr,
            grid_shape=_get_shape(best_shape.grid_dims_stride()),
            block_shape=_get_shape(best_shape.block_dims_stride()),
            thread_shape=_get_shape(best_shape.thread_dims_stride()),
            subthread_shape=_get_shape(best_shape.subthread_dims_stride()),
            grid_dim=best_shape.grid_dims,
            block_dim=best_shape.block_dims,
            thread_dim=best_shape.thread_dims,
        )


@dataclass
class XPULocalGemmAction(XPUBaseGemmAction):

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        sic_id, sip_id = self.get_kernel_idx()
        total_latency = 0
        chip_workload = self.get_chip_workload()
        workload, best_shape = tile_local_gemm_workload(self.config, self.get_chip_workload())
        self.core_cost.instruction_info = self.get_instruction_info(chip_workload, best_shape)
        if sic_id not in workload or sip_id not in workload[sic_id]:
            self.core_cost.latency = 0
            return
        # pprint(workload[sic_id][sip_id])
        total_latency = yield from dsm_local_gemm_kernel(
            workload[sic_id][sip_id], self.config, best_shape.subthread_cnt_in_thread(), 0
        )
        self.core_cost.latency = total_latency

    def _basic_stat_info(self):
        self.core_cost.tensor_dtype = self.get_dtype()


@dataclass
class XPUSharedGemmAction(XPUBaseGemmAction):
    # 统计信息
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        sic_id, sip_id = self.get_kernel_idx()
        total_latency = 0
        chip_workload = self.get_chip_workload()
        workload, best_shape = tile_shared_gemm_workload(self.config, chip_workload)
        self.core_cost.instruction_info = self.get_instruction_info(chip_workload, best_shape)
        if sic_id not in workload:
            self.core_cost.latency = 0
            return
        # pprint(workload[sic_id])
        total_latency = yield from dsm_shared_gemm_kernel(
            workload, best_shape, self.config, f"{self.case_id}_{self.config.gcu_id}", sic_id, sip_id, 0, self.ref
        )
        self.core_cost.latency = total_latency

    def _basic_stat_info(self):
        self.core_cost.tensor_dtype = self.get_dtype()
