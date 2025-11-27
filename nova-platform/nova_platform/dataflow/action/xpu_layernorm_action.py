from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

from nova_platform.base_model import (
    AddrDomain,
    DType,
    DataflowActionComputeStat,
    DataflowActionMemoryStat,
    DataflowActionType,
)
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction

import logging
logger = logging.getLogger(__file__)

BYPTES_PER_OA = 512


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    ld_ins_num: int = 0
    scalar_ins_num: int = 0
    st_ins_num: int = 0
    compute_cycle_num: int = 0
    compute_1d_ins_num: int = 0
    compute_sfu_ins_num: int = 0
    compute_ins_shape: int = 0


def layernorm_kernel_fp16(dim0_size: int, dim1_size: int):
    compute_cycle_num = 0
    compute_1d_ins_num = 0
    st_ins_num = 0
    scalar_ins_num = 18
    ld_ins_num = 0
    compute_sfu_ins_num = 0
    elements_per_OA = int(BYPTES_PER_OA / 2)
    unroll_num = 8
    for i in range(dim0_size):
        scalar_ins_num += 3
        scalar_ins_num += 19
        for j in range(0, dim1_size, elements_per_OA * unroll_num):
            scalar_ins_num += 3
            ld_ins_num += unroll_num
            compute_1d_ins_num += unroll_num  # mop_cvt
            compute_1d_ins_num += unroll_num * 2  # mop_mul
            compute_1d_ins_num += unroll_num * 4  # mop_add
            compute_cycle_num += unroll_num
        compute_1d_ins_num += 14
        scalar_ins_num += 1
        for j in range(0, dim1_size, elements_per_OA * unroll_num):
            scalar_ins_num += 3
            ld_ins_num += unroll_num
            compute_1d_ins_num += unroll_num  # mop_cvt
            compute_1d_ins_num += unroll_num * 2  # mop_sub
            compute_1d_ins_num += unroll_num * 2  # mop_mul
            compute_1d_ins_num += unroll_num * 2  # mop_add
            compute_1d_ins_num += unroll_num  # mop_cvt
            st_ins_num += 1

    return InstructionInfo(
        dtype=DType.FP16,
        ld_ins_num=ld_ins_num,
        scalar_ins_num=scalar_ins_num,
        st_ins_num=st_ins_num,
        compute_cycle_num=compute_cycle_num,
        compute_1d_ins_num=compute_1d_ins_num,
        compute_sfu_ins_num=compute_sfu_ins_num,
        compute_ins_shape=elements_per_OA,
    )


def layernorm_kernel(size: int, dtype: DType = DType.FP16):
    compute_cycle_num = 0
    compute_1d_ins_num = 0
    st_ins_num = 0
    scalar_ins_num = 18
    ld_ins_num = 0
    compute_sfu_ins_num = 0
    elements_per_OA = int(BYPTES_PER_OA / dtype.get_bpe())
    for i in range(0, size, elements_per_OA):
        ld_ins_num += 2
        compute_1d_ins_num += 20
        st_ins_num += 1
        compute_cycle_num += 1

    return InstructionInfo(
        dtype=dtype,
        ld_ins_num=ld_ins_num,
        scalar_ins_num=scalar_ins_num,
        st_ins_num=st_ins_num,
        compute_cycle_num=compute_cycle_num,
        compute_1d_ins_num=compute_1d_ins_num,
        compute_sfu_ins_num=compute_sfu_ins_num,
        compute_ins_shape=elements_per_OA,
    )


@dataclass
class CoreCost(BaseCoreStat):
    oa_occupation: int = 0  # Bytes
    main_body_length: int = 0
    main_body_num: int = 0
    unroll_num: int = 8
    subthread_num: int = 8

    data_preparation_cost: int = 0
    compute_cost: int = 0

    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)


@dataclass
class XPULayernormAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        return self.outputs[0].tensor[0].dims

    def _get_l0_occupation(self):  # step 3
        self.core_cost.oa_occupation = 3 * BYPTES_PER_OA  # 2 for inputs, 1 for outputs

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = self.core_cost.unroll_num * \
            self.core_cost.subthread_num

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        size = self.get_valid_shape()
        size = reduce(lambda a, b: a * b, size, 1)
        self.core_cost.instruction_info = layernorm_kernel(
            size, dtype=self.get_dtype())
        # core_cost.instruction_info = layernorm_kernel_fp16(size[1], size[0])
        self._get_l0_occupation()
        self._get_main_body_length()
        stat_ref = 0
        total_latency = 0
        total_compute_latency = 0
        dtype = self.core_cost.instruction_info.dtype
        throughput_1d = self.config.compute.thread_1d_throughput[dtype]
        throughput_sfu = self.config.compute.thread_sfu_throughput

        lhs1_tensor = self.inputs[0].tensor[0]
        lhs1_access = self._iter_addr(lhs1_tensor, "r")
        lhs2_tensor = self.inputs[0].tensor[0]
        lhs2_access = self._iter_addr(lhs2_tensor, "r")
        out_tensor = self.outputs[0].tensor[0]
        out_access = self._iter_addr(out_tensor, "w")
        input1_gen = self._iter_access_gen([lhs1_access])
        next(input1_gen)
        input2_gen = self._iter_access_gen([lhs2_access])
        next(input2_gen)
        output_gen = self._iter_access_gen([out_access])
        next(output_gen)

        for i in range(0, self.core_cost.instruction_info.compute_cycle_num, self.core_cost.main_body_length):
            inst_num = min(self.core_cost.main_body_length,
                           self.core_cost.instruction_info.compute_cycle_num - i)
            lhs, out = [inst_num * BYPTES_PER_OA] * 2

            lhs_read_1 = DataflowActionMemoryStat(
                total_count=lhs,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=stat_ref,
                memory_access_list=input1_gen.send(lhs),
            )
            yield lhs_read_1
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - lhs_read_1.latency:{lhs_read_1.latency}, lhs_read_1.leading_latency:{lhs_read_1.leading_latency}"
            )

            compute_1d_ops = (
                inst_num
                * (
                    self.core_cost.instruction_info.compute_1d_ins_num
                    / self.core_cost.instruction_info.compute_cycle_num
                )
                * self.core_cost.instruction_info.compute_ins_shape
            )
            compute_sfu_ops = (
                inst_num
                * (
                    self.core_cost.instruction_info.compute_sfu_ins_num
                    / self.core_cost.instruction_info.compute_cycle_num
                )
                * self.core_cost.instruction_info.compute_ins_shape
            )
            compute_scalar_cycle = (
                inst_num
                * self.core_cost.instruction_info.scalar_ins_num
                / self.core_cost.instruction_info.compute_cycle_num
            )
            compute_cycle = (
                compute_1d_ops / throughput_1d
                + compute_sfu_ops / throughput_sfu
                + compute_scalar_cycle
            )
            compute_ref = stat_ref + lhs_read_1.leading_latency
            # compute_latency = compute_cycle / \
            #     (self.config.freq.CORE_CLOCK_DOMAIN * 1e9)

            # total_compute_latency += compute_latency

            compute_stat1_1d = DataflowActionComputeStat(
                compute_1d_ops={
                    dtype: compute_1d_ops/2
                },
                relative_ts=compute_ref,
            )
            yield compute_stat1_1d
            compute_stat1_sfu = DataflowActionComputeStat(
                compute_msf_ops=compute_sfu_ops/2,
                relative_ts=compute_ref+compute_stat1_1d.latency,
            )
            yield compute_stat1_sfu
            compute_stat1_scalar = DataflowActionComputeStat(
                compute_scalar_cycle=compute_scalar_cycle/2,
                relative_ts=compute_ref+compute_stat1_1d.latency+compute_stat1_sfu.latency,
            )
            yield compute_stat1_scalar
            # note: compute_latency = half of the latency computed by compute_cycle / throughput / freq
            compute_latency1 = compute_stat1_1d.latency + \
                compute_stat1_sfu.latency+compute_stat1_scalar.latency
            total_compute_latency += compute_latency1
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - compute_cycle: {compute_cycle/2}, compute latency:{compute_latency1}"
            )

            # lhs_read_2_ref = lhs_read_1.leading_latency + compute_latency / 2
            lhs_read_2_ref = lhs_read_1.leading_latency + compute_latency1
            lhs_read_2_ref = max(
                lhs_read_2_ref, lhs_read_1.latency - lhs_read_1.leading_latency)
            lhs_read_2 = DataflowActionMemoryStat(
                total_count=lhs,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=stat_ref + lhs_read_2_ref,
                memory_access_list=input2_gen.send(lhs),
            )
            yield lhs_read_2
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - lhs_read_2.latency:{lhs_read_2.latency}, lhs_read_2.leading_latency:{lhs_read_2.leading_latency}"
            )

            compute_ref = stat_ref + lhs_read_2_ref + lhs_read_2.leading_latency
            compute_stat2_1d = DataflowActionComputeStat(
                compute_1d_ops={
                    dtype: compute_1d_ops/2
                },
                relative_ts=compute_ref,
            )
            yield compute_stat2_1d
            compute_stat2_sfu = DataflowActionComputeStat(
                compute_msf_ops=compute_sfu_ops/2,
                relative_ts=compute_ref+compute_stat2_1d.latency,
            )
            yield compute_stat2_sfu
            compute_stat2_scalar = DataflowActionComputeStat(
                compute_scalar_cycle=compute_scalar_cycle/2,
                relative_ts=compute_ref+compute_stat2_1d.latency+compute_stat2_sfu.latency,
            )
            yield compute_stat2_scalar
            compute_latency2 = (
                compute_stat2_1d.latency
                + compute_stat2_sfu.latency
                + compute_stat2_scalar.latency
            )
            total_compute_latency += compute_latency2

            # out_ref = lhs_read_2_ref + lhs_read_2.leading_latency + compute_latency / 2
            out_ref = lhs_read_2_ref + lhs_read_2.leading_latency + compute_latency2
            out_write = DataflowActionMemoryStat(
                total_count=out,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=stat_ref + out_ref,
                memory_access_list=output_gen.send(out),
            )
            yield out_write
            out_latency, out_leading_latency = out_write.latency, out_write.leading_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - out_latency:{out_latency}, out_leading_latency:{out_leading_latency}"
            )

            total_latency = max(
                total_latency,
                stat_ref + lhs_read_1.latency,  # vld 1 latency
                stat_ref + lhs_read_1.leading_latency + \
                # compute_latency / 2,  # compute 1 latency
                compute_latency1,
                stat_ref + lhs_read_2_ref + lhs_read_2.latency,  # vld 2 latency
                stat_ref + lhs_read_2_ref + lhs_read_2.leading_latency + \
                # compute_latency / 2,  # compute 2 latency
                compute_latency2,
                stat_ref + out_ref + out_latency,  # vst latency
            )
            # stat_ref = total_latency - vld_leading_latency  # update ref
            stat_ref = (
                stat_ref
                + out_ref
                + out_write.leading_latency
                + (out_write.latency - out_write.leading_latency) /
                self.core_cost.subthread_num
            )  # update ref
        print(
            f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - total_latency:{total_latency}")
        self.core_cost.latency = total_latency
        self.core_cost.compute_cost = total_compute_latency

    def _basic_stat_info(self):
        info = self.core_cost.instruction_info
        dtype = self.core_cost.instruction_info.dtype
        bpe = dtype.get_bpe()
        self.core_cost.vector_dtype = dtype
        self.core_cost.vector_ops = info.compute_ins_shape * info.compute_1d_ins_num
        self.core_cost.ld_l0_l3 = (
            info.compute_ins_shape * bpe) * info.ld_ins_num
        self.core_cost.st_l0_l3 = (
            info.compute_ins_shape * bpe) * info.st_ins_num
