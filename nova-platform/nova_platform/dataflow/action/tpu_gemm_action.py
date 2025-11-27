from dataclasses import dataclass, field
from typing import Generator

from nova_platform.base_model import DType, DataflowActionType
from nova_platform.benchmark.op_base import Workload, Operand
from nova_platform.benchmark.batch_gemm_tpu import tile_tpu_gemm_workload, BatchGemmGridShape
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.diag_action import DiagTensor
from nova_platform.dataflow.action.xpu_action import XPUAction


def _action_tensor_to_operand(tensor: DiagTensor) -> Operand:
    # 现有 diag tensor 采用 reversed 维度，保持与 XPU 行为一致的适配
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
class TPUCoreCost(BaseCoreStat):
    instruction_info: dict = field(default_factory=dict)


@dataclass
class TPUGemmAction(XPUAction):
    """
    极简 TPU GEMM action：使用占位 tiler，记录基本 shape，暂不建模访存/延迟。
    """

    core_cost: TPUCoreCost = field(default_factory=TPUCoreCost)

    def get_kernel_idx(self):
        return self.data[0], self.data[1]

    def get_chip_workload(self) -> Workload:
        lhs = _action_tensor_to_operand(self.inputs[0].tensor[0])
        rhs = _action_tensor_to_operand(self.inputs[1].tensor[0])
        res = _action_tensor_to_operand(self.outputs[0].tensor[0])
        inputs = [lhs, rhs]
        attr = {"b": lhs.dim[0], "m": lhs.dim[1], "k": lhs.dim[2], "n": rhs.dim[2]}
        return Workload(inputs=inputs, outputs=[res], attr=attr, dtype=self.get_dtype())

    def get_memory_stat(self) -> Generator:
        chip_workload = self.get_chip_workload()
        workloads, best_shape = tile_tpu_gemm_workload(self.config, chip_workload)
        # 仅记录基本形状信息；占位，不返回具体访存事件
        self.core_cost.instruction_info = {
            "grid_dims": best_shape.grid_dims,
            "block_dims": best_shape.block_dims,
            "thread_dims": best_shape.thread_dims,
        }
        self.core_cost.latency = 0
        yield from ()

    def _basic_stat_info(self):
        self.core_cost.tensor_dtype = self.get_dtype()
