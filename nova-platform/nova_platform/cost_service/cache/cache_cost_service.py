from collections import Counter
from typing import Dict, List
import logging

from nova_platform.base_model import DataflowActionMemoryAccess, DataflowActionType
from nova_platform.config import BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BossaNovaContext, DataflowAction
from nova_platform.dataflow.dataflow import Dataflow
from nova_platform.cost_service.cache.base_cache_model import BaseCacheCostService, CacheStat

logger = logging.getLogger(__name__)

try:
    from nova_platform.cost_service.cache.arch import RequestType, SimpleAccess, SimpleStrategy
    from nova_platform.cost_service.cache.arch.eltanin import GCU_Eltanin
    from nova_platform.cost_service.cache.arch.libra import GCU_Libra
    from cache_model.entity.model import HardwareConfig, L1C_Config, L3_Config, LLC_Config, Memory
    from cache_model.type_util import INF
    _CACHE_MODEL_AVAILABLE = True
except ModuleNotFoundError:
    RequestType = SimpleAccess = SimpleStrategy = None
    GCU_Eltanin = GCU_Libra = None
    INF = HardwareConfig = L1C_Config = L3_Config = LLC_Config = Memory = None
    _CACHE_MODEL_AVAILABLE = False
    logger.warning(
        "cache_model package not found; cache statistics will be disabled."
    )


if _CACHE_MODEL_AVAILABLE:

    class CacheCostService(BaseCacheCostService):
        def __init__(self, config: BossaNovaConfig):
            super().__init__(config)
            # TODO: pass cache directly, may need refactor
            arch_map = {
                "eltanin": GCU_Eltanin,
                "libra": GCU_Libra
            }
            if config.arch_name not in arch_map:
                raise Exception("arch %s not supported", config.arch_name)
            self.arch = arch_map[config.arch_name]

            self.cache_lvl_request_type_map = {
                GCU_Eltanin: {
                    DataflowActionType.CDTE: RequestType.LLC,
                    DataflowActionType.XPU: RequestType.L1C,
                },
                GCU_Libra: {
                    DataflowActionType.CDTE: RequestType.LLC,
                    DataflowActionType.XPU: RequestType.LLC,
                },
            }

            memory_config = Memory(
                LLC=LLC_Config(
                    CACHE_LINE_SIZE=config.memory.llc.CACHE_LINE_SIZE,
                    CACHE_WAYS=config.memory.llc.CACHE_WAYS,
                    CACHE_SIZE=config.memory.llc.CACHE_SIZE,
                    MEM_LATENCY=config.memory.llc.MEM_LATENCY,
                    NON_MEM_LATENCY=config.memory.llc.NON_MEM_LATENCY,
                    NUM_MSHR=config.memory.llc.NUM_MSHR if config.memory.llc.NUM_MSHR else INF,
                    NUM_OF_PARTITIONS=config.memory.llc.NUM_OF_PARTITIONS*config.inst_num.NUM_OF_DIE,
                    NUM_OF_SLICES_PER_PARTITION=config.memory.llc.NUM_OF_SLICES_PER_PARTITION,
                ),

                L3=L3_Config(
                    START_ADDR=0x500_0000_0000,
                    SIZE_PER_HBM=0x9_0000_0000,
                    NUM_OF_HBM=4,
                )
            )
            if config.memory.l1c:
                L1C = L1C_Config(
                    CACHE_LINE_SIZE=config.memory.l1c.CACHE_LINE_SIZE,
                    CACHE_WAYS=config.memory.l1c.CACHE_WAYS,
                    CACHE_SIZE=config.memory.l1c.CACHE_SIZE,
                    MEM_LATENCY=config.memory.l1c.MEM_LATENCY,
                    NON_MEM_LATENCY=config.memory.l1c.NON_MEM_LATENCY,
                    NUM_MSHR=config.memory.l1c.NUM_MSHR if config.memory.l1c.NUM_MSHR else INF,
                    NUM_OF_SIP=config.memory.l1c.NUM_OF_CORE*config.inst_num.NUM_OF_DIE,
                    CACHE_SIZE_PER_SIP=config.memory.l1c.CACHE_SIZE_PER_CORE,
                )
                memory_config.L1C = L1C

            hardware_config = HardwareConfig(
                MEMORY=memory_config
            )

            self.cache_model = SimpleStrategy(hardware_config, self.arch)
            # self.last_stat = self.cache_model.memory_device.stat_dict()

        def __deepcopy__(self, memo):
            # deepcopy逻辑，None表示跳过
            return None

        def _process_access(self, action: DataflowAction, access_list: List[DataflowActionMemoryAccess]):
            for access in access_list:
                base_addr = access.base_addr
                size = access.size
                # TODO: need refactor
                if base_addr < 5*2**40:
                    continue
                request_type = self.cache_lvl_request_type_map[self.arch][action.get_action_type(
                )]

                sip_id = action.get_engine_id()
                access = SimpleAccess(
                    0 if access.rw == 'r' else 1,
                    base_addr,
                    1,
                    size,
                    base_addr+size-1,
                    request_type=request_type,
                    sip_id=sip_id
                )
                self.cache_model.reuse_distance(access)
            self.cache_model.finish_queue()

        def get_cache_stat_dict(self):
            stat_dict = self.cache_model.memory_device.stat_dict()
            for k, counter in stat_dict.items():
                w_count = sum([v for rw, v in counter.items() if "write" in rw])
                w_hit = counter.get("write_hits", 0)
                w_hit_rate = w_hit/w_count if w_count else 1
                r_count = sum([v for rw, v in counter.items() if "read" in rw])
                r_hit = counter.get("read_hits", 0)
                r_hit_rate = r_hit/r_count if r_count else 1
                stat_dict[k] = CacheStat(
                    write_count=w_count,
                    write_hit_count=w_hit,
                    read_count=r_count,
                    read_hit_count=r_hit
                )
            return stat_dict

        def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):
            stat = self.cache_model.memory_device.stat_dict()
            report = {}
            for lvl, _stat in stat.items():
                report[lvl] = dict(_stat)
                read_count = sum([v for rw, v in _stat.items() if "read" in rw])
                read_hit_count = _stat.get("read_hits", 0)
                read_hit_rate = read_hit_count/read_count if read_count else 0
                write_count = sum([v for rw, v in _stat.items() if "write" in rw])
                write_hit_count = _stat.get("write_hits", 0)
                write_hit_rate = write_hit_count/write_count if write_count else 0
                report[lvl].update({
                    "read_hit_rate": read_hit_rate,
                    "write_hit_rate": write_hit_rate
                })

            return report

else:

    class CacheCostService(BaseCacheCostService):
        def __init__(self, config: BossaNovaConfig):
            super().__init__(config)
            self._has_l1c = bool(getattr(config.memory, "l1c", None))
            self._stat_template = {"LLC": CacheStat(), "L3": CacheStat()}
            if self._has_l1c:
                self._stat_template["L1C"] = CacheStat()
            logger.info("Using no-op CacheCostService; cache stats remain zero.")

        def __deepcopy__(self, memo):
            return None

        def _process_access(self, action: DataflowAction, access_list: List[DataflowActionMemoryAccess]):
            # cache_model unavailable: skip access simulation
            return

        def get_cache_stat_dict(self):
            return {level: CacheStat() for level in self._stat_template}

        def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):
            report = {}
            for level in self._stat_template:
                report[level] = {
                    "read_count": 0,
                    "read_hit_rate": 0,
                    "write_count": 0,
                    "write_hit_rate": 0,
                }
            return report
