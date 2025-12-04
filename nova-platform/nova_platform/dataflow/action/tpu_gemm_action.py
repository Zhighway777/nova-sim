from dataclasses import dataclass, field
from typing import Generator

from nova_platform.base_model import (
    AddrDomain,
    DataflowActionComputeStat,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    DType,
)
from nova_platform.benchmark.op_base import Workload, Operand, list_product
from nova_platform.benchmark.batch_gemm_tpu import tile_tpu_gemm_workload, BatchGemmGridShape
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.diag_action import DiagTensor
from nova_platform.dataflow.action.xpu_action import XPUAction

from nova_platform.benchmark.op_base import list_product

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
class TpuCoreCost(BaseCoreStat):
    instruction_info: dict = field(default_factory=dict)


@dataclass
class TpuGemmAction(XPUAction):
    """
    TPU GEMM action：使用 TPU tiler 输出，生成基本访存/计算统计和延迟。
    """

    core_cost: TpuCoreCost = field(default_factory=TpuCoreCost)

    def get_kernel_idx(self):
        if len(self.data) >= 2:
            return self.data[0], self.data[1]
        return self.data[0], 0

    def get_trace_label(self) -> str:
        # TPU 阵列轨道名称使用 SA 前缀
        return "sa"

    def get_chip_workload(self) -> Workload:    
        lhs = _action_tensor_to_operand(self.inputs[0].tensor[0])
        rhs = _action_tensor_to_operand(self.inputs[1].tensor[0])
        res = _action_tensor_to_operand(self.outputs[0].tensor[0])
        inputs = [lhs, rhs]
        attr = {"b": lhs.dim[0], "m": lhs.dim[1], "k": lhs.dim[2], "n": rhs.dim[2]}
        return Workload(inputs=inputs, outputs=[res], attr=attr, dtype=self.get_dtype())

    def _array_flops(self):
        freq_core = getattr(getattr(self.config, "freq", None), "CORE", 1.0)
        compute = getattr(self.config, "compute", None)
        arr_m = getattr(getattr(compute, "tpu", None), "ARRAY_M", 64) if compute else 64
        arr_n = getattr(getattr(compute, "tpu", None), "ARRAY_N", 64) if compute else 64
        return arr_m * arr_n * freq_core * 1e9  # mac/s

    def _hbm_bw(self):
        bw = getattr(self.config, "bw", None)
        freq = getattr(self.config, "freq", None)
        freq_mc = getattr(freq, "MC", getattr(freq, "CORE", 1.0)) if freq else 1.0
        num_die = getattr(getattr(self.config, "inst_num", None), "NUM_OF_DIE", 1)
        hbm_bw_cfg = getattr(getattr(getattr(bw, "mc", None), "l3", None), "bw", 64) if bw else 64
        return hbm_bw_cfg * freq_mc * 1e9 * num_die  # bytes/s

    def _vmem_bw(self):
        # 简化：使用 HBM 带宽作为 VMEM 带宽占位
        return self._hbm_bw()

    def _vregs_bw(self):
        # 寄存器层假设比 HBM 快 4 倍
        return self._hbm_bw() * 4

    def _single_access(self, base_addr: int, size: int, rw: str) -> list:
        return [DataflowActionMemoryAccess(base_addr=base_addr, size=size, rw=rw)]

    def _calc_time(self, bytes_total: float, bw_bytes_per_s: float) -> float:
        if bw_bytes_per_s <= 0:
            return 0.0
        return bytes_total / bw_bytes_per_s

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        sic_id, sip_id = self.get_kernel_idx()
        chip_workload = self.get_chip_workload()
        workloads, best_shape = tile_tpu_gemm_workload(self.config, chip_workload)
        self.core_cost.instruction_info = {
            "grid_dims": best_shape.grid_dims,
            "block_dims": best_shape.block_dims,
            "thread_dims": best_shape.thread_dims,
            "tile_k": best_shape.calc_ceil_K_l2,
        }
        if sic_id not in workloads or sip_id not in workloads[sic_id]:
            self.core_cost.latency = 0
            return

        array_flops = self._array_flops()
        hbm_bw = self._hbm_bw()
        vmem_bw = self._vmem_bw()
        vregs_bw = self._vregs_bw()
        total_ref = 0.0

        for idx, wl in enumerate(workloads[sic_id][sip_id]):
            bytes_lhs = wl.attr.get("bytes_lhs") or sum(list_product(tuple(t.dim)) * t.bpe for t in wl.inputs if t)
            bytes_rhs = wl.attr.get("bytes_rhs") or 0
            bytes_res = wl.attr.get("bytes_res") or sum(list_product(tuple(t.dim)) * t.bpe for t in wl.outputs)
            bytes_in = bytes_lhs + bytes_rhs

            # HBM -> VMEM (映射为 LOCAL->L3，与 XPU 路由兼容)
            mem_hbm = DataflowActionMemoryStat(
                total_count=int(bytes_in),
                master=DataflowActionType.CDTE,
                src=AddrDomain.LOCAL,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=total_ref,
                memory_access_list=self._single_access(wl.inputs[0].addr if wl.inputs else 0, int(bytes_in), "r"),
                name=f"hbm->vmem-{idx}",
            )
            yield mem_hbm
            hbm_time = self._calc_time(bytes_in, hbm_bw)

            # VMEM -> VREGS (映射为 L0->LOCAL，与 XPU 路由兼容)
            mem_vmem = DataflowActionMemoryStat(
                total_count=int(bytes_in),
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.LOCAL,
                rw="r",
                relative_ts=total_ref + hbm_time,
                memory_access_list=self._single_access(wl.inputs[0].addr if wl.inputs else 0, int(bytes_in), "r"),
                name=f"vmem->vregs-{idx}",
            )
            yield mem_vmem
            vmem_time = self._calc_time(bytes_in, vmem_bw)

            # Compute
            macs = wl.attr["m"] * wl.attr["n"] * wl.attr["k"]
            compute_ref = total_ref + hbm_time + vmem_time
            compute_stat = DataflowActionComputeStat(
                name="tpu_gemm_mac",
                compute_2d_ops={self.get_dtype(): macs * 2},
                relative_ts=compute_ref,
            )
            yield compute_stat
            compute_time = (macs * 2) / (array_flops or 1.0)

            # VREGS -> HBM (写回)
            out_ref = compute_ref + compute_time
            mem_out = DataflowActionMemoryStat(
                total_count=int(bytes_res),
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=out_ref,
                memory_access_list=self._single_access(wl.outputs[0].addr if wl.outputs else 0, int(bytes_res), "w"),
                name=f"vregs->hbm-{idx}",
            )
            yield mem_out
            out_time = self._calc_time(bytes_res, hbm_bw)

            total_ref = out_ref + out_time

        self.core_cost.latency = total_ref

    def _basic_stat_info(self):
        self.core_cost.tensor_dtype = self.get_dtype()
