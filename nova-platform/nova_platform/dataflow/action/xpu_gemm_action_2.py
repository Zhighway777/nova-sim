from typing import Generator
from dataclasses import dataclass, field
from typing import Generator, List, Tuple

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
from nova_platform.benchmark.op_base import Workload, Operand, list_product


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    grid_dim: List[int] = field(default_factory=list)
    block_dim: List[int] = field(default_factory=list)
    thread_dim: List[int] = field(default_factory=list)
    subthread_dim: List[int] = field(default_factory=list)

    def subthread_cnt_in_thread(self):
        return list_product(tuple(self.thread_dim))


@dataclass
class CoreCost(BaseCoreStat):
    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)
    sip_workloads: list = field(default_factory=list)


def reversed_dims(t):
    return tuple(reversed(t))


@dataclass
class XPUGemmAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        valid_m = self.data[4] & (0xFFFF)
        valid_n = self.data[4] >> 16
        valid_k = self.data[5]
        return [valid_m, valid_n, valid_k]

    def get_instruction_info(self):
        return InstructionInfo(
            dtype=self.get_dtype(),
            grid_dim=reversed_dims(self.tile_info.cube_dim),
            block_dim=reversed_dims(self.tile_info.grid_dim),
            thread_dim=reversed_dims(self.tile_info.block_dim),
            subthread_dim=self.tile_info.tile_shape,
        )

    def get_sip_workload(self):

        def action_tensor_to_operand(tensor, shape):

            return Operand(
                dim=shape,
                addr=tensor.addr,
                bpe=tensor.bpe,
                dim_offset=(0, 0, 0),
                dim_stride=reversed_dims(tensor.stride_dims[:3]),
            )

        lhs = self.inputs[0].tensor[0]
        rhs = self.inputs[1].tensor[0]
        res = self.outputs[0].tensor[0] if len(self.outputs) else None
        m, n, k = self.get_valid_shape()
        return Workload(
            inputs=[action_tensor_to_operand(
                lhs, (1, m, k)), action_tensor_to_operand(rhs, (1, k, n))],
            outputs=[action_tensor_to_operand(res, (1, m, n))] if res else [],
            dtype=self.get_dtype(),
        )

    def get_minimum_iter_k(self, b, m, n):
        throughput = self.config.compute.thread_2d_throughput[self.get_dtype()]
        l1_latency = self.config.bw.l0.local.pre_latency
        k = 0
        mac_cycle = 0
        while mac_cycle < l1_latency:
            k += 32
            mac_cycle = b * m * n * k / throughput
        return k

    def _iter_tensor_addr(self, tensor: Operand, rw) -> Generator[DataflowActionMemoryAccess, None, None]:
        mem_access = tensor.get_contiguous_mem_accesses()
        for addr, size in mem_access:
            yield DataflowActionMemoryAccess(addr, size, rw)

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        in_m, in_n, in_k = self.get_valid_shape()
        in_b = 1
        self.core_cost.instruction_info = self.get_instruction_info()
        self.core_cost.sip_workloads.append((in_b, in_m, in_n, in_k))
        dtype = self.get_dtype()
        self.core_cost.dtype = dtype
        bpe = dtype.get_bpe()
        sip_workload = self.get_sip_workload()

        dtype = self.core_cost.instruction_info.dtype
        lhs_gen = self._iter_access_gen(
            list(self._iter_tensor_addr(sip_workload.inputs[0], "r")))
        next(lhs_gen)

        rhs_gen = self._iter_access_gen(
            list(self._iter_tensor_addr(sip_workload.inputs[1], "r")))
        next(rhs_gen)

        if len(sip_workload.outputs) > 0:
            out_gen = self._iter_access_gen(
                list(self._iter_tensor_addr(sip_workload.outputs[0], "r")))
            next(out_gen)
        iter_ref = 0
        iter_k = self.get_minimum_iter_k(in_b, in_m, in_n)
        iter_k = min(in_k, iter_k)
        self.core_cost.tensor_macs[dtype] = 0
        for i in range(0, in_k, iter_k):
            k = min(iter_k, in_k - i)
            l0_lhs_size = in_m * k * bpe
            l0_lhs_read = DataflowActionMemoryStat(
                total_count=l0_lhs_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.SHARED,
                rw="r",
                relative_ts=iter_ref,
                memory_access_list=lhs_gen.send(l0_lhs_size),
                name=f"lhs:l1->l0",
            )
            yield l0_lhs_read

            l0_rhs_size = in_n * k * bpe
            l0_rhs_read = DataflowActionMemoryStat(
                total_count=l0_rhs_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.SHARED,
                rw="r",
                relative_ts=iter_ref,
                memory_access_list=rhs_gen.send(l0_rhs_size),
                name=f"rhs:l1->l0",
            )
            yield l0_rhs_read

            compute_ref = iter_ref + \
                max(l0_lhs_read.leading_latency, l0_rhs_read.leading_latency)
            scalar_ops = 4 * self.core_cost.instruction_info.subthread_cnt_in_thread()

            compute_stat1 = DataflowActionComputeStat(
                name=f"scalar",
                compute_scalar_cycle=scalar_ops,
                relative_ts=compute_ref,
            )
            yield compute_stat1
            scalar_cost = compute_stat1.latency

            compute_mac = in_m * in_n * k
            compute_stat2 = DataflowActionComputeStat(
                name=f"mac",
                compute_2d_ops={dtype: compute_mac * 2},
                relative_ts=compute_ref,
            )
            yield compute_stat2
            self.core_cost.tensor_macs[dtype] += compute_mac
            VMM = 10
            compute_vmm = DataflowActionComputeStat(
                name=f"vmm",
                compute_nop_cycle=VMM,
                relative_ts=compute_ref + compute_stat2.latency,
            )
            yield compute_vmm
            compute_latency = max(
                compute_ref + compute_stat2.latency + compute_vmm.latency,
                iter_ref + max(l0_lhs_read.latency, l0_rhs_read.latency),
            )
            iter_ref = compute_latency - \
                max(l0_lhs_read.leading_latency, l0_rhs_read.leading_latency)
            out_ref = compute_ref

        out_latency = 0
        if len(sip_workload.outputs) > 0:
            if self.core_cost.instruction_info.dtype == DType.FP8:
                scaling_ref = compute_ref + compute_stat2.latency + compute_vmm.latency
                compute_scaling = DataflowActionComputeStat(
                    name="scaling",
                    compute_1d_ops={DType.FP32: in_m * in_n},
                    relative_ts=scaling_ref,
                )
                yield compute_scaling
                compute_latency = max(
                    compute_latency, scaling_ref + compute_scaling.latency)
            out_size = in_m * in_n * bpe
            out_wirte = DataflowActionMemoryStat(
                total_count=out_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.SHARED,
                rw="w",
                relative_ts=out_ref,
                memory_access_list=out_gen.send(out_size),
                name="out",
            )
            yield out_wirte
            out_latency = out_ref + out_wirte.latency
        total_latency = max(out_latency, compute_latency)
        self.core_cost.latency = total_latency

    def _basic_stat_info(self):
        # TODO: need review
        pass
