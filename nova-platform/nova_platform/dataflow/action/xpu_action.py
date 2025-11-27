from dataclasses import dataclass
from typing import Generator, List
from nova_platform.base_model import DType, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowOpType
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext
from nova_platform.dataflow.action.diag_action import DiagDataflowAction, DiagTensor


@dataclass
class XPUAction(DiagDataflowAction):
    code: str
    data: list
    core_cost: BaseCoreStat

    def get_local_engine_id(self):
        return self.engine_id % self.config.inst_num.NUM_OF_CORE_PER_CLUSTER

    def get_valid_shape(self):
        return [
            self.tile_info.tile_shape[0] * self.tile_info.block_dim[1],
            self.tile_info.tile_shape[1] * self.tile_info.block_dim[2],
            self.tile_info.tile_shape[2],
        ]

    def get_optype(self) -> DataflowOpType:
        return DataflowOpType(self.code.split("_")[0])

    def get_dtype(self) -> DType:
        return DType(self.code.split("_")[-1])

    def _basic_stat_info(self):
        raise NotImplementedError()

    def compute(self, context: BossaNovaContext) -> Generator[DataflowActionMemoryStat, any, BaseCoreStat]:
        yield from self.get_memory_stat()
        self._basic_stat_info()
        return self.core_cost

    def _iter_addr(self, tensor: DiagTensor, rw) -> DataflowActionMemoryAccess:
        data_num = self.data[0]
        base_addr = tensor.addr
        data_size = data_num*tensor.bpe
        return DataflowActionMemoryAccess(base_addr, data_size, rw)
