from collections import Counter
from cache_model.arch import AbstractGCU
from cache_model.memory import AbstractMemoryManager
from cache_model.memory.addr_converter import addr_to_llc_index
from cache_model.memory.memory_manger import L3Manager, LLCManager
from cache_model.entity.model import HardwareConfig, ModelContext, HardwareConfig
from cache_model.type_util import *

from nova_platform.cost_service.cache.arch import RequestType, SimpleRequest


class GCU_Libra(AbstractGCU):
    def __init__(self, hardware: HardwareConfig) -> None:
        super().__init__(hardware)
        self.hardware = hardware
        self.context = ModelContext(self.hardware)

        self.L3 = L3Manager(hardware.MEMORY.L3, self.context)
        # TODO: addr_to_llc_index only support llc 8/16

        CACHE_SETS = int(hardware.MEMORY.LLC.CACHE_SIZE /
                         hardware.MEMORY.LLC.CACHE_LINE_SIZE/hardware.MEMORY.LLC.CACHE_WAYS)

        def addr_convert(addr):
            line_addr = addr//hardware.MEMORY.LLC.CACHE_LINE_SIZE
            llc_index = (line_addr//CACHE_SETS) % (hardware.MEMORY.LLC.NUM_OF_PARTITIONS *
                                                   hardware.MEMORY.LLC.NUM_OF_SLICES_PER_PARTITION)
            return llc_index
        self.LLC = LLCManager(hardware.MEMORY.LLC, self.context,
                              self.L3, addr_convert)

    def get_last_memory_manager(self) -> AbstractMemoryManager:
        return self.LLC

    def process(self, request: SimpleRequest, timestamp: int):
        self.context.timestamp = timestamp
        if request.request_type == RequestType.LLC:
            self.LLC.process(request)
        else:
            raise Exception(
                "request type %s is not supported in Libra", request.request_type)

    def post_process(self, timestamp):
        self.LLC.post_process(timestamp)

    def stat_dict(self):
        _stat_dict = {
            "LLC": self.LLC.stat(),
            "L3": self.L3.stat(),
        }
        return _stat_dict

    def histogram_dict(self) -> Counter:
        distance_dict = {
            "LLC": self.LLC.histogram()
        }
        return distance_dict
