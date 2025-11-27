import random
from dataclasses import dataclass, field
from typing import Generator

from nova_platform.base_model import AddrDomain, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext
from nova_platform.dataflow.action.diag_action import DiagTensor
from nova_platform.dataflow.action.xpu_action import XPUAction
try:
    from scipy.stats import invgauss, uniform
except ModuleNotFoundError:
    class _InvGauss:
        def rvs(self, mu):
            sigma = abs(mu) * 0.1 if mu else 1.0
            return max(random.gauss(mu if mu else 1.0, sigma), 0.0)

        def ppf(self, q, mu):
            # simple fallback based on gaussian assumption
            sigma = abs(mu) * 0.1 if mu else 1.0
            return max(mu + sigma, 0.0)

    class _Uniform:
        def __init__(self, low, width):
            self.low = low
            self.width = width

        def rvs(self):
            return random.uniform(self.low, self.low + self.width)

    invgauss = _InvGauss()

    def uniform(a, b):
        return _Uniform(a, b)

import logging
logger = logging.getLogger(__name__)

random.seed(22)

BYPTES_PER_OA = 512  # bytes
NOC_DATALINE_SIZE = 128  # bytes


def align_data_size(size, align=NOC_DATALINE_SIZE):
    return (size + align - 1) // align * align


@dataclass
class InstructionInfo:
    bpe: int = 2
    ld_index_l3_ins_num: int = 0
    ld_table_l3_ins_num: int = 0
    st_table_l3_ins_num: int = 0
    scalar_ins_num: int = 0

# length = embedding size
# index = index size


def gather_instruction_mapping(embedding_size: int, index: int, bpe: int = 2):
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
class XPUGatherAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        out_dim = self.outputs[0].tensor[0].dims
        return out_dim[0], out_dim[1]  # table size, index size

    def _iter_random_addr(self, tensor: DiagTensor, num):
        bpe = tensor.bpe
        stride = tensor.stride_dims[0]*bpe
        width = tensor.dims[0]*bpe
        mu = self.config.gather_mu
        width = align_data_size(width) if mu >= 0 else width
        rows = tensor.dims[1]
        base_addr = tensor.addr

        if mu < 0:
            # e.g. mu = -4.100  [3].[200]
            # skip_m = 4
            # seq_num = 200
            skip_m = -int(mu)
            seq_num = int(("%.03f" % (mu + skip_m))[3:6])
            assert num//seq_num == num/seq_num, "num should be divided by continuity"
            loop_num = num//seq_num
            # if mu <0, use uniform distribution
            uniform_gen = uniform(0, rows)
            for _ in range(loop_num):
                r = int(uniform_gen.rvs()) % (rows-(seq_num-1)*skip_m)
                for i in range(seq_num):
                    addr = base_addr + (r+i)*stride
                    yield DataflowActionMemoryAccess(addr, width, 'r')
        else:
            lx = invgauss.ppf(0.95, mu)
            scale = rows/lx
            for _ in range(num):
                # addr = base_addr + random.randint(0, rows-1)*stride
                # r = int(invgauss.rvs(rows-1))  # 1 0.5 0.25
                # r = int(invgauss.rvs(mu)) % rows
                r = int(invgauss.rvs(mu)*scale) % rows
                addr = base_addr + r*stride

                yield DataflowActionMemoryAccess(addr, width, 'r')

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        embedding_size, idx_size = self.get_valid_shape()
        bpe = self.inputs[0].tensor[0].bpe
        self.core_cost.instruction_info = gather_instruction_mapping(
            embedding_size, idx_size, bpe=bpe)
        ld_table_vs_index = int(
            self.core_cost.instruction_info.ld_table_l3_ins_num /
            self.core_cost.instruction_info.ld_index_l3_ins_num
        )

        index_tensor = self.inputs[1].tensor[0]
        num_lookup = index_tensor.dims[0]
        index_access = DataflowActionMemoryAccess(
            index_tensor.addr, index_tensor.dims[0]*index_tensor.bpe, 'r')
        input_gen = self._iter_access_gen([index_access])
        next(input_gen)

        lookup_tensor = self.inputs[0].tensor[0]
        mu = self.config.gather_mu
        lookup_access = self._iter_random_addr(
            lookup_tensor, num_lookup)
        lookup_gen = self._iter_access_gen(lookup_access)
        next(lookup_gen)

        output_tensor = self.outputs[0].tensor[0]
        output_access = self._iter_tensor_addr(
            output_tensor.addr, output_tensor, 'w')
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
                name="ld_index",
            )

            yield index_read
            total_ld_table_num = ld_table_vs_index * ld_index_num
            table_ref = stat_ref + index_read.latency
            table_latency = 0
            offset = self.core_cost.unroll_num * self.core_cost.subthread_num
            # if mu < 0:
            #     offset = total_ld_table_num
            for j in range(0, total_ld_table_num, offset):
                mu = self.config.gather_mu
                if mu >= 0:
                    ld_table_num = min(
                        self.core_cost.unroll_num * self.core_cost.subthread_num, total_ld_table_num - j)
                    table_datalane_size = min(
                        BYPTES_PER_OA, align_data_size(embedding_size * bpe))
                else:
                    # ld_table_num = total_ld_table_num
                    # table_datalane_size = embedding_size * bpe
                    ld_table_num = min(
                        self.core_cost.unroll_num * self.core_cost.subthread_num, total_ld_table_num - j)
                    table_datalane_size = min(
                        BYPTES_PER_OA, embedding_size * bpe)
                table_read = DataflowActionMemoryStat(
                    total_count=ld_table_num * table_datalane_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="r",
                    relative_ts=table_ref,
                    memory_access_list=lookup_gen.send(
                        ld_table_num * table_datalane_size),
                    name="ld_table",
                )
                yield table_read
                table_write = DataflowActionMemoryStat(
                    total_count=ld_table_num * table_datalane_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="w",
                    relative_ts=table_ref + table_read.leading_latency,
                    memory_access_list=output_gen.send(
                        ld_table_num * table_datalane_size),
                    name="st_table",
                )
                yield table_write
                table_latency = max(
                    table_latency,
                    table_ref + table_read.latency,
                    table_ref + table_read.leading_latency + table_write.latency,

                )
                table_ref = table_latency - table_read.leading_latency

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

        self.core_cost.main_body_num = float(
            self.core_cost.instruction_info.st_table_l3_ins_num) / self.core_cost.main_body_length

    def _basic_stat_info(self):
        info = self.core_cost.instruction_info
        self.core_cost.ld_l0_l3 = (info.ld_index_l3_ins_num +
                                   info.ld_table_l3_ins_num) * BYPTES_PER_OA
        self.core_cost.st_l0_l3 = info.st_table_l3_ins_num * BYPTES_PER_OA

    def compute(self, context: BossaNovaContext) -> Generator[DataflowActionMemoryStat, any, CoreCost]:
        self._get_main_body_length()
        yield from self.get_memory_stat()
        self._basic_stat_info()
        return self.core_cost
