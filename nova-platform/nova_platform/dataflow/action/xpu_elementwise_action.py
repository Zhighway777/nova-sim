from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

from nova_platform.base_model import AddrDomain, DType, DataflowActionComputeStat, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType
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
    compute_ins_num: int = 0
    compute_ins_shape: int = 0


def elementwise_vector_kernel(size: int, dtype: DType = DType.FP16):
    compute_ins_num = 0
    st_ins_num = 0
    scalar_ins_num = 18
    ld_ins_num = 0
    elements_per_OA = int(BYPTES_PER_OA / dtype.get_bpe())
    enroll_time = 16
    for i in range(0, size, elements_per_OA):
        ld_ins_num += 2
        compute_ins_num += 1
        scalar_ins_num += 3
        st_ins_num += 1

    return InstructionInfo(dtype, ld_ins_num, scalar_ins_num, st_ins_num, compute_ins_num, elements_per_OA)


@dataclass
class CoreCost(BaseCoreStat):
    oa_occupation: int = 0  # Bytes
    main_body_length: int = 0
    main_body_num: int = 0
    unroll_num: int = 16
    subthread_num: int = 4
    data_preparation_cost: int = 0
    compute_cost: int = 0

    instruction_info: InstructionInfo = None
    elements_per_cycle: float = 0


@dataclass
class XPUElementWiseAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_memory_access(self) -> Generator[DataflowActionMemoryAccess, None, None]:
        lhs_tensor = self.inputs[0].tensor[0]
        lhs_access = self._iter_addr(lhs_tensor, 'r')
        rhs_tensor = self.inputs[1].tensor[0]
        rhs_access = self._iter_addr(rhs_tensor, 'r')
        out_tensor = self.outputs[0].tensor[0]
        out_access = self._iter_addr(out_tensor, 'w')
        yield from self._iter_access_gen([lhs_access, rhs_access, out_access])

    def _get_l0_occupation(self):  # step 3
        self.core_cost.oa_occupation = 3 * BYPTES_PER_OA  # 2 for inputs, 1 for outputs

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = int(
            self.config.memory.l0.OA_SIZE / self.core_cost.oa_occupation)
        self.core_cost.main_body_length = self.core_cost.unroll_num * \
            self.core_cost.subthread_num
        self.core_cost.main_body_num = float(
            self.core_cost.instruction_info.compute_ins_num) / self.core_cost.main_body_length

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        size = self.get_valid_shape()
        size = reduce(lambda a, b: a * b, size, 1)
        self.core_cost.instruction_info = elementwise_vector_kernel(
            size, dtype=self.get_dtype())
        self._get_l0_occupation()
        self._get_main_body_length()
        stat_ref = 0
        total_latency = 0
        total_compute_latency = 0

        dtype = self.core_cost.instruction_info.dtype

        throughput = self.config.compute.thread_1d_throughput[dtype]

        lhs_tensor = self.inputs[0].tensor[0]
        lhs_access = self._iter_addr(lhs_tensor, 'r')
        rhs_tensor = self.inputs[1].tensor[0]
        rhs_access = self._iter_addr(rhs_tensor, 'r')
        out_tensor = self.outputs[0].tensor[0]
        out_access = self._iter_addr(out_tensor, 'w')
        input_gen = self._iter_access_gen([lhs_access, rhs_access])
        next(input_gen)
        output_gen = self._iter_access_gen([out_access])
        next(output_gen)

        for i in range(0, self.core_cost.instruction_info.compute_ins_num, self.core_cost.main_body_length):
            inst_num = min(self.core_cost.main_body_length,
                           self.core_cost.instruction_info.compute_ins_num - i)
            lhs, rhs, out = [inst_num * BYPTES_PER_OA] * 3

            lhs_read = DataflowActionMemoryStat(
                total_count=lhs*2,
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

            # rhs_ref = stat_ref+lhs_latency-lhs_leading_latency
            # rhs_read = DataflowActionMemoryStat(
            #     rhs, DataflowActionType.XPU, AddrDomain.L0, AddrDomain.L3, "r", rhs_ref)
            # yield rhs_read
            # rhs_latency, rhs_leading_latency = rhs_read.latency, rhs_read.leading_latency
            # logger.debug(
            #     f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - rhs_latency:{rhs_latency}, rhs_leading_latency:{rhs_leading_latency}"
            # )

            # vld_latency = max(lhs_latency, rhs_latency)
            # vld_leading_latency = max(lhs_leading_latency, rhs_leading_latency)

            vld_leading_latency = lhs_leading_latency

            compute_ops = inst_num * self.core_cost.instruction_info.compute_ins_shape
            compute_cycle = (
                compute_ops / throughput
                + inst_num * self.core_cost.instruction_info.scalar_ins_num /
                self.core_cost.instruction_info.compute_ins_num
            )
            compute_ref = stat_ref+vld_leading_latency
            # compute_latency = compute_cycle / \
            #     (self.config.freq.CORE_CLOCK_DOMAIN * 1e9)

            compute_stat = DataflowActionComputeStat(
                compute_1d_ops={
                    dtype: compute_ops
                },
                relative_ts=compute_ref,
            )
            yield compute_stat

            compute_latency = compute_stat.latency
            total_compute_latency += compute_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - compute_cycle: {compute_cycle}, compute latency:{compute_latency}"
            )

            out_ref = compute_ref + compute_latency
            out_write = DataflowActionMemoryStat(
                total_count=out,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=out_ref,
                memory_access_list=output_gen.send(out)
            )
            yield out_write
            out_latency, out_leading_latency = out_write.latency, out_write.leading_latency
            logger.debug(
                f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - out_latency:{out_latency}, out_leading_latency:{out_leading_latency}"
            )

            total_latency = max(
                total_latency,
                stat_ref + lhs_latency,
                compute_ref + compute_latency,  # compute latency,
                out_ref + out_latency,  # vst latency
            )
            stat_ref = total_latency - vld_leading_latency  # update ref
            # stat_ref = total_latency-out_leading_latency  # update ref
        print(
            f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - total_latency:{total_latency}")
        self.core_cost.latency = total_latency
        self.core_cost.compute_cost = total_compute_latency
        yield  # using yield instead of return to avoid from StopIterator exception

    def _basic_stat_info(self):
        info = self.core_cost.instruction_info
        dtype = self.core_cost.instruction_info.dtype
        bpe = dtype.get_bpe()
        self.core_cost.vector_dtype = dtype
        self.core_cost.vector_ops = info.compute_ins_shape * info.compute_ins_num
        self.core_cost.ld_l0_l3 = 2 * \
            (info.compute_ins_shape * bpe) * info.compute_ins_num
        self.core_cost.st_l0_l3 = (info.compute_ins_shape *
                                   bpe) * info.compute_ins_num
        self.core_cost.elements_per_cycle = (
            info.compute_ins_shape * info.st_ins_num /
            (self.core_cost.latency * 1e9 * self.config.freq.CORE)
        )


class XPUAddAction(XPUElementWiseAction):
    pass


class XPUMulAction(XPUElementWiseAction):
    pass
