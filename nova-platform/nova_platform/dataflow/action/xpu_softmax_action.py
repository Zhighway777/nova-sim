from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

from nova_platform.base_model import AddrDomain, DType, DataflowActionComputeStat, DataflowActionMemoryStat, DataflowActionType
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction

import logging
logger = logging.getLogger(__name__)


BYPTES_PER_OA = 512


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    ld_ins_num: int = 0
    scalar_ins_num: int = 0
    st_ins_num: int = 0
    compute_1d_ins_num: int = 0
    compute_sfu_ins_num: int = 0
    compute_ins_shape: int = 0


def softmax_kernel(size: int, dtype: DType = DType.FP16):
    compute_1d_ins_num = 0
    compute_sfu_ins_num = 0
    st_ins_num = 0
    scalar_ins_num = 12
    ld_ins_num = 0
    elements_per_OA = int(BYPTES_PER_OA / dtype.get_bpe())

    for i in range(0, size, elements_per_OA):
        ld_ins_num += 1
        compute_1d_ins_num += 5
        compute_sfu_ins_num += 1
        scalar_ins_num += 1
        st_ins_num += 1

    return InstructionInfo(
        dtype=dtype,
        ld_ins_num=ld_ins_num,
        scalar_ins_num=scalar_ins_num,
        st_ins_num=st_ins_num,
        compute_1d_ins_num=compute_1d_ins_num,
        compute_sfu_ins_num=compute_sfu_ins_num,
        compute_ins_shape=elements_per_OA,
    )


@dataclass
class CoreCost(BaseCoreStat):
    oa_occupation: int = 0  # Bytes
    main_body_length: int = 0
    main_body_num: int = 0

    data_preparation_cost: int = 0
    compute_cost: int = 0

    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)


@dataclass
class XPUSoftmaxAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def _get_l0_occupation(self):  # step 3
        self.core_cost.oa_occupation = 2 * BYPTES_PER_OA  # 2 for inputs, 1 for outputs

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = int(
            self.config.memory.l0.OA_SIZE / self.core_cost.oa_occupation)
        # 64 = 4 thread * 16 per thread
        self.core_cost.main_body_length = 64
        self.main_body_num = float(
            self.core_cost.instruction_info.ld_ins_num) / self.core_cost.main_body_length

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        size = self.get_valid_shape()
        size = reduce(lambda a, b: a * b, size, 1)
        self.core_cost.instruction_info = softmax_kernel(
            size, dtype=self.get_dtype())
        self._get_l0_occupation()
        self._get_main_body_length()
        stat_ref = 0
        total_latency = 0
        total_compute_latency = 0
        dtype = self.core_cost.instruction_info.dtype

        throughput_1d = self.config.compute.thread_1d_throughput[dtype]
        throughput_sfu = self.config.compute.thread_sfu_throughput

        input_tensor = self.inputs[0].tensor[0]
        input_access = self._iter_addr(input_tensor, 'r')
        output_tensor = self.outputs[0].tensor[0]
        output_access = self._iter_addr(output_tensor, 'w')
        input_gen = self._iter_access_gen([input_access])
        next(input_gen)
        output_gen = self._iter_access_gen([output_access])
        next(output_gen)

        for i in range(0, self.core_cost.instruction_info.ld_ins_num, self.core_cost.main_body_length):
            inst_num = min(self.core_cost.main_body_length,
                           self.core_cost.instruction_info.ld_ins_num - i)
            lhs, out = [inst_num * BYPTES_PER_OA] * 2

            lhs_read = DataflowActionMemoryStat(
                total_count=lhs,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=stat_ref,
                memory_access_list=input_gen.send(lhs)
            )
            yield lhs_read
            lhs_latency, lhs_leading_latency = lhs_read.latency, lhs_read.leading_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - lhs_latency:{lhs_latency}, lhs_leading_latency:{lhs_leading_latency}"
            )

            vld_latency = lhs_latency
            vld_leading_latency = lhs_leading_latency

            compute_1d_ops = (
                inst_num
                * (self.core_cost.instruction_info.compute_1d_ins_num / self.core_cost.instruction_info.ld_ins_num)
                * self.core_cost.instruction_info.compute_ins_shape
            )
            compute_sfu_ops = (
                inst_num
                * (self.core_cost.instruction_info.compute_sfu_ins_num / self.core_cost.instruction_info.ld_ins_num)
                * self.core_cost.instruction_info.compute_ins_shape
            )
            # 0.8 means 1D || SFU
            compute_cycle = (
                max(compute_1d_ops / throughput_1d, compute_sfu_ops / throughput_sfu) / 0.8 +
                + inst_num * self.core_cost.instruction_info.scalar_ins_num /
                self.core_cost.instruction_info.ld_ins_num
            )
            compute_ref = stat_ref+lhs_leading_latency
            # compute_latency = compute_cycle / \
            #     (self.config.freq.CORE_CLOCK_DOMAIN * 1e9)
            # print("comp[ute_latency:", compute_1d_ops / throughput_1d, compute_sfu_ops / throughput_sfu, inst_num * core_cost.instruction_info.scalar_ins_num / core_cost.instruction_info.ld_ins_num, compute_latency)

            compute_stat = DataflowActionComputeStat(
                compute_1d_ops={
                    dtype: compute_1d_ops
                },
                compute_msf_ops=compute_sfu_ops,
                relative_ts=compute_ref,
            )
            yield compute_stat

            compute_latency = compute_stat.latency

            total_compute_latency += compute_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - compute_cycle: {compute_cycle}, compute latency:{compute_latency}"
            )

            out_write = DataflowActionMemoryStat(
                total_count=out,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=stat_ref + vld_leading_latency+compute_latency/4,
                memory_access_list=output_gen.send(out))
            yield out_write
            out_latency, out_leading_latency = out_write.latency, out_write.leading_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - out_latency:{out_latency}, out_leading_latency:{out_leading_latency}"
            )

            total_latency = max(
                total_latency,
                stat_ref + vld_leading_latency + compute_latency,  # compute latency,
                stat_ref + lhs_latency,
                stat_ref + vld_leading_latency+compute_latency / 4 + out_latency,
            )
            stat_ref = total_latency - vld_leading_latency  # update ref
        logger.debug(
            f"{self.get_engine_id()}:{self.get_engine_sub_id()}  - total_latency:{total_latency}")
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
