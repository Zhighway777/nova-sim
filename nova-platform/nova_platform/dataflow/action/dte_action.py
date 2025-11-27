from dataclasses import dataclass
from functools import reduce
from typing import Generator, Tuple

from nova_platform.base_model import AddrDomain, BaseActionStat, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext
from nova_platform.dataflow.action.diag_action import DiagDataflowAction, DiagTensor
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseDTEStat(BaseActionStat):
    r_datasize: int = 0
    w_datasize: int = 0


@dataclass
class DTEAction(DiagDataflowAction):
    op: str
    param0: list[int]
    param1: list[int]
    param2: list[int]
    param3: list[int]
    core_cost: BaseDTEStat

    def get_local_engine_id(self):
        return self.engine_sub_id

    def compute(self, context: BossaNovaContext) -> Generator[DataflowActionMemoryStat, None, BaseCoreStat]:
        # cost_book = context.get_cost_book(self)
        self.core_cost = BaseDTEStat()
        latency = 0
        for stat in self.get_memory_stat():
            yield stat
            latency = max(latency, stat.latency+stat.relative_ts)
            if stat.rw == 'r':
                self.core_cost.r_datasize += stat.total_count
                # cost_book.r_datasize += stat.total_count
            else:
                self.core_cost.w_datasize += stat.total_count
                # cost_book.w_datasize += stat.total_count

        # cost_book.latency = latency
        self.core_cost.latency = latency
        return self.core_cost


class CDTESliceAction(DTEAction):

    def get_memory_access(self) -> Generator[DataflowActionMemoryAccess, None, None]:
        src: DiagTensor = self.inputs[0].tensor[0]
        dst: DiagTensor = self.outputs[0].tensor[0]
        src_addr_domain = AddrDomain.get_addr_domain(src.addr)
        dst_addr_domain = AddrDomain.get_addr_domain(dst.addr)
        if src_addr_domain == AddrDomain.SHARED and dst_addr_domain == AddrDomain.L3:
            param0 = self.param0
            tensor = DiagTensor(**src.__dict__)
            tensor.dims = param0
            tensor.stride_dims = dst.stride_dims
            yield from self._iter_tensor_addr(dst.addr, tensor, 'w')
        elif src_addr_domain == AddrDomain.L3 and dst_addr_domain == AddrDomain.SHARED:
            param0 = self.param0
            tensor = DiagTensor(**dst.__dict__)
            tensor.dims = param0
            tensor.stride_dims = src.stride_dims
            yield from self._iter_tensor_addr(src.addr, tensor, 'r')

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        if logger.level == logging.DEBUG:
            src_addr = self.inputs[0].tensor[0].addr
            dst_addr = self.outputs[0].tensor[0].addr
            src_addr_domain = AddrDomain.get_addr_domain(src_addr)
            dst_addr_domain = AddrDomain.get_addr_domain(dst_addr)
            assert src_addr_domain == AddrDomain.L3
            assert dst_addr_domain == AddrDomain.SHARED

        data_dims = self.param0
        data_size = reduce(lambda a, b: a*b, data_dims, 1) * \
            self.inputs[0].tensor[0].bpe
        yield DataflowActionMemoryStat(
            total_count=data_size,
            master=DataflowActionType.CDTE,
            src=AddrDomain.SHARED,
            dst=AddrDomain.L3,
            rw='r',
            memory_access_list=list(self.get_memory_access())
        )


class CDTEDesliceAction(CDTESliceAction):
    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        if logger.level == logging.DEBUG:
            src_addr = self.inputs[0].tensor[0].addr
            dst_addr = self.outputs[0].tensor[0].addr
            src_addr_domain = AddrDomain.get_addr_domain(src_addr)
            dst_addr_domain = AddrDomain.get_addr_domain(dst_addr)
            assert src_addr_domain == AddrDomain.SHARED
            assert dst_addr_domain == AddrDomain.L3

        data_dims = self.param0
        data_size = reduce(lambda a, b: a*b, data_dims, 1) * \
            self.inputs[0].tensor[0].bpe
        yield DataflowActionMemoryStat(
            total_count=data_size,
            master=DataflowActionType.CDTE,
            src=AddrDomain.SHARED,
            dst=AddrDomain.L3,
            rw='w',
            memory_access_list=list(self.get_memory_access())
        )


class CDTEReshapeAction(DTEAction):
    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        if logger.level == logging.DEBUG:
            src_addr = self.inputs[0].tensor[0].addr
            dst_addr = self.outputs[0].tensor[0].addr
            src_addr_domain = AddrDomain.get_addr_domain(src_addr)
            dst_addr_domain = AddrDomain.get_addr_domain(dst_addr)
            assert src_addr_domain == AddrDomain.L3
            assert dst_addr_domain == AddrDomain.L3
        data_size = reduce(
            lambda a, b: a*b, self.outputs[0].tensor[0].dims, 1) * self.outputs[0].tensor[0].bpe

        input_tensor = self.inputs[0].tensor[0]
        input_access = self._iter_tensor_addr(
            input_tensor.addr, input_tensor, 'r')
        input_gen = self._iter_access_gen(input_access)
        next(input_gen)

        output_tensor = self.outputs[0].tensor[0]
        output_access = self._iter_tensor_addr(
            output_tensor.addr, output_tensor, 'w')
        output_gen = self._iter_access_gen(output_access)
        next(output_gen)

        # read
        read_stat = DataflowActionMemoryStat(
            total_count=data_size,
            master=DataflowActionType.CDTE,
            src=AddrDomain.L3,
            dst=AddrDomain.L3,
            rw='r',
            memory_access_list=input_gen.send(data_size)
        )
        yield read_stat
        # write
        yield DataflowActionMemoryStat(
            total_count=data_size,
            master=DataflowActionType.CDTE,
            src=AddrDomain.L3,
            dst=AddrDomain.L3,
            rw='w',
            relative_ts=read_stat.leading_latency,
            memory_access_list=output_gen.send(data_size)
        )
