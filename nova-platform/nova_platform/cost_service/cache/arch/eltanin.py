from collections import Counter
from cache_model.arch import AbstractGCU
from cache_model.memory import AbstractMemoryManager
from cache_model.memory.memory_manger import L1CManager, L3Manager, LLCManager
from cache_model.entity.model import HardwareConfig, ModelContext, HardwareConfig
from cache_model.type_util import *

from nova_platform.cost_service.cache.arch import RequestType, SimpleRequest


class GCU_Eltanin(AbstractGCU):
    def __init__(self, hardware: HardwareConfig) -> None:
        super().__init__(hardware)
        self.hardware = hardware
        self.context = ModelContext(self.hardware)

        def cache_selector(request: SimpleRequest, num_of_sets):
            slice_id = request.sip_id
            set_index = request.line_addr % num_of_sets
            return slice_id, set_index

        self.L3 = L3Manager(hardware.MEMORY.L3, self.context)
        self.LLC = LLCManager(hardware.MEMORY.LLC, self.context,
                              self.L3, lambda addr: (addr//128) % 96)
        self.L1C = L1CManager(hardware.MEMORY.L1C,
                              self.context, self.LLC, cache_selector)

    def get_last_memory_manager(self) -> AbstractMemoryManager:
        return self.LLC

    def process(self, request: SimpleRequest, timestamp: int):
        self.context.timestamp = timestamp
        if request.request_type == RequestType.LLC:
            self.LLC.process(request)
        elif request.request_type == RequestType.L1C:
            self.L1C.process(request)

    def post_process(self, timestamp):
        self.LLC.post_process(timestamp)
        self.L1C.post_process(timestamp)

    def stat_dict(self):
        _stat_dict = {
            "L1C": self.L1C.stat(),
            "LLC": self.LLC.stat(),
            "L3": self.L3.stat(),
        }
        return _stat_dict

    def histogram_dict(self) -> Counter:
        distance_dict = {
            "LLC": self.LLC.histogram()
        }
        return distance_dict
