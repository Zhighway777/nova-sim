import random
from dataclasses import dataclass, field
from typing import Generator

from nova_platform.base_model import AddrDomain, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext
from nova_platform.dataflow.action.diag_action import DiagTensor
from nova_platform.dataflow.action.xpu_action import XPUAction

import logging

logger = logging.getLogger(__name__)

random.seed(22)

BYPTES_PER_OA = 512


@dataclass
class InstructionInfo:
    bpe: int = 2
    ld_index_l3_ins_num: int = 0
    ld_table_l3_ins_num: int = 0
    st_table_l3_ins_num: int = 0
    scalar_ins_num: int = 0


# length = embedding size
# index = index size


def scatter_instruction_mapping(embedding_size: int, index: int, bpe: int = 2):
    ld_index_L3_inst_cout = 0
    ld_table_L3_inst_cout = 0
    st_table_L3_inst_cout = 0
    scalar_inst_count = 0
    remain_index_size = index
    ELEMENTS_PER_OA = int(BYPTES_PER_OA / bpe)
    scalar_inst_count += 8
    for i in range(0, index, ELEMENTS_PER_OA):
        slice_index_size = ELEMENTS_PER_OA if remain_index_size > ELEMENTS_PER_OA else remain_index_size
        remain_index_size -= slice_index_size
        scalar_inst_count += 8
        scalar_inst_count += 16
        ld_index_L3_inst_cout += 1
        for j in range(0, slice_index_size):
            scalar_inst_count += 3
            scalar_inst_count += 6
            remain_length_size = embedding_size
            for k in range(0, embedding_size, ELEMENTS_PER_OA):
                slice_length_size = ELEMENTS_PER_OA if remain_length_size > ELEMENTS_PER_OA else remain_length_size
                remain_length_size -= slice_length_size
                scalar_inst_count += 3
                scalar_inst_count += 7
                ld_table_L3_inst_cout += 1
                st_table_L3_inst_cout += 1

    return InstructionInfo(
        bpe=bpe,
        ld_index_l3_ins_num=ld_index_L3_inst_cout,
        ld_table_l3_ins_num=ld_table_L3_inst_cout,
        st_table_l3_ins_num=st_table_L3_inst_cout,
        scalar_ins_num=scalar_inst_count,
    )


@dataclass
class CoreCost(BaseCoreStat):
    oa_occupation: int = 0  # Bytes
    main_body_length: int = 0
    main_body_num: int = 0
    subthread_num: int = 8
    unroll_num: int = 16
    data_preparation_cost: int = 0
    compute_cost: int = 0

    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)
    elements_per_cycle: float = 0


@dataclass
class XPUScatterAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        out_dim = self.inputs[2].tensor[0].dims
        return out_dim[0], out_dim[1]  # table size, index size

    def _iter_random_addr(self, tensor: DiagTensor, num, rw):
        bpe = tensor.bpe
        stride = tensor.stride_dims[0] * bpe
        width = tensor.dims[0] * bpe
        rows = tensor.dims[1]
        base_addr = tensor.addr

        for _ in range(num):
            addr = base_addr + random.randint(0, rows - 1) * stride
            yield DataflowActionMemoryAccess(addr, width, rw)

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        embedding_size, idx_size = self.get_valid_shape()
        bpe = self.inputs[0].tensor[0].bpe
        self.core_cost.instruction_info = scatter_instruction_mapping(
            embedding_size, idx_size, bpe=bpe)
        ld_table_vs_index = int(
            self.core_cost.instruction_info.ld_table_l3_ins_num /
            self.core_cost.instruction_info.ld_index_l3_ins_num
        )

        index_tensor = self.inputs[1].tensor[0]
        num_index = index_tensor.dims[0]
        index_access = DataflowActionMemoryAccess(
            index_tensor.addr, index_tensor.dims[0] * index_tensor.bpe, "r")
        input_gen = self._iter_access_gen([index_access])
        next(input_gen)

        update_tensor = self.inputs[2].tensor[0]
        update_access = self._iter_tensor_addr(
            update_tensor.addr, update_tensor, "w")
        update_gen = self._iter_access_gen(update_access)
        next(update_gen)

        output_tensor = self.outputs[0].tensor[0]

        output_access = self._iter_random_addr(output_tensor, num_index, "w")
        output_gen = self._iter_access_gen(output_access)
        next(output_gen)

        stat_ref = 0
        total_latency = 0
        for i in range(0, self.core_cost.instruction_info.ld_index_l3_ins_num, self.core_cost.subthread_num):
            ld_index_num = min(self.core_cost.subthread_num,
                               self.core_cost.instruction_info.ld_index_l3_ins_num - i)
            index_read = DataflowActionMemoryStat(
                total_count=ld_index_num * BYPTES_PER_OA,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=stat_ref,
                memory_access_list=input_gen.send(
                    ld_index_num * BYPTES_PER_OA),
            )
            yield index_read
            total_ld_table_num = ld_table_vs_index * ld_index_num
            table_ref = stat_ref + index_read.latency
            table_latency = 0
            for j in range(0, total_ld_table_num, self.core_cost.unroll_num * self.core_cost.subthread_num):
                ld_table_num = min(
                    self.core_cost.unroll_num * self.core_cost.subthread_num, total_ld_table_num - j)
                table_read = DataflowActionMemoryStat(
                    total_count=ld_table_num * BYPTES_PER_OA,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="r",
                    relative_ts=table_ref,
                    memory_access_list=update_gen.send(
                        ld_table_num * BYPTES_PER_OA),
                )
                yield table_read
                table_write = DataflowActionMemoryStat(
                    total_count=ld_table_num * BYPTES_PER_OA,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="w",
                    relative_ts=table_ref + table_read.leading_latency,
                    memory_access_list=output_gen.send(
                        ld_table_num * BYPTES_PER_OA),
                )
                yield table_write
                table_ref = table_ref + table_read.latency
                table_latency = max(
                    table_latency,
                    table_ref + table_read.latency,
                    table_ref + table_read.leading_latency + table_write.latency,
                )

            total_latency = max(
                total_latency,
                stat_ref + index_read.latency,  # index latency,
                table_latency,
            )
            stat_ref = total_latency - index_read.leading_latency
        print(
            f"{self.get_engine_id()}:{self.get_engine_sub_id()} {i} - total_latency:{total_latency}")
        self.core_cost.latency = total_latency

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = 16 * 4

        self.core_cost.main_body_num = (
            float(self.core_cost.instruction_info.st_table_l3_ins_num) /
            self.core_cost.main_body_length
        )

    def _basic_stat_info(self):
        info = self.core_cost.instruction_info
        self.core_cost.ld_l0_l3 = (
            info.ld_index_l3_ins_num + info.ld_table_l3_ins_num) * BYPTES_PER_OA
        self.core_cost.st_l0_l3 = info.st_table_l3_ins_num * BYPTES_PER_OA

    def compute(self, context: BossaNovaContext) -> Generator[DataflowActionMemoryStat, any, CoreCost]:
        self._get_main_body_length()
        yield from self.get_memory_stat()
        self._basic_stat_info()
        return self.core_cost
