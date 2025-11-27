from dataclasses import dataclass
from typing import Generator, Tuple

from nova_platform.base_model import DataflowActionMemoryAccess, DataflowActionMemoryStat
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext, DataflowAction
from nova_platform.dataflow.action.diag_action import DiagDataflowAction
from nova_platform.dataflow.action.xpu_action import XPUAction


@dataclass
class XPUNopAction(XPUAction):
    def get_memory_access(self) -> Generator[DataflowActionMemoryAccess, None, None]:
        yield from ()

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        yield from ()

    def compute(self, context: BossaNovaContext) -> Generator[None, None, BaseCoreStat]:
        self.core_cost = BaseCoreStat()
        yield
        return self.core_cost
