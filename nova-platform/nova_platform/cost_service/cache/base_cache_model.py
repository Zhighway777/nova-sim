from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Dict, Generator, List

from nova_platform.base_model import DataflowActionMemoryAccess, DataflowActionMemoryStat
from nova_platform.config import BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BaseCostService
from nova_platform.cost_service.compute.base_compute_model import BossaNovaContext, DataflowAction

import logging
logger = logging.getLogger(__name__)


@dataclass
class CacheStat:
    write_count: int = 0
    write_hit_count: int = 0
    read_count: int = 0
    read_hit_count: int = 0

    @property
    def write_hit_rate(self):
        return self.write_hit_count/self.write_count if self.write_count > 0 else 0

    @property
    def read_hit_rate(self):
        return self.read_hit_count/self.read_count if self.read_count > 0 else 0

    def __sub__(self, b: 'CacheStat'):
        diff_write_count = self.write_count - b.write_count
        diff_write_hit_count = self.write_hit_count - b.write_hit_count

        diff_read_count = self.read_count - b.read_count
        diff_read_hit_count = self.read_hit_count - b.read_hit_count

        return CacheStat(
            write_count=diff_write_count,
            write_hit_count=diff_write_hit_count,
            read_count=diff_read_count,
            read_hit_count=diff_read_hit_count
        )

    def __add__(self, b: 'CacheStat'):
        diff_write_count = self.write_count + b.write_count
        diff_write_hit_count = self.write_hit_count + b.write_hit_count

        diff_read_count = self.read_count + b.read_count
        diff_read_hit_count = self.read_hit_count + b.read_hit_count

        return CacheStat(
            write_count=diff_write_count,
            write_hit_count=diff_write_hit_count,
            read_count=diff_read_count,
            read_hit_count=diff_read_hit_count
        )


class BaseCacheCostService(BaseCostService):
    def __init__(self, config: BossaNovaConfig):
        super().__init__(config)

    def get_raw_stat_dict(self) -> Dict[str, Counter]:
        raise NotImplementedError()

    def process(self, action: DataflowAction, context: BossaNovaContext, ref: float) -> Generator[None, DataflowActionMemoryStat | None, None]:
        action_stat = defaultdict(CacheStat)

        while True:
            mem_stat = yield
            if not mem_stat:
                break

            memory_access_list: List[DataflowActionMemoryAccess] = mem_stat.memory_access_list
            start_stat = self.get_cache_stat_dict()
            self._process_access(action, memory_access_list)
            end_stat = self.get_cache_stat_dict()
            stat = {}
            for k, v in end_stat.items():
                stat[k] = v-start_stat[k]

            for k, v in stat.items():
                action_stat[k] += v

            mem_stat.cache_stat = stat

            # log the mem access
            # logger.info("memory access list: %s", memory_access_list)
            # for access in memory_access_list:
            #     logging.info(
            #         "cache service - action: %s, memory_access: base_addr=%s, size=%s, rw=%s",
            #         action.get_action_id(),
            #         access.base_addr,
            #         access.size,
            #         access.rw
            #     )

            # # 打印 llc_stat 的详细信息
            # llc_stat = self.get_cache_stat_dict()["LLC"]
            # logging.info(
            #     "llc_stat: write_count=%s, write_hit_rate=%.2f, read_count=%s, read_hit_rate=%.2f, \n\n",
            #     llc_stat.write_count,
            #     llc_stat.write_hit_rate,
            #     llc_stat.read_count,
            #     llc_stat.read_hit_rate
            # )
        cost_book = context.get_cost_book(action)

        # action_stat = {}
        # for k, v in last_stat.items():
        #     action_stat[k] = v-action_begin_stat[k]

        cost_book.cache_stat_dict = dict(action_stat)
        logger.debug("cache service - action: %s, stat: %s",
                     action.get_action_id(), cost_book.cache_stat_dict)

    def process_old(self, action: DataflowAction, context: BossaNovaContext, ref: float) -> Generator[bool, DataflowActionMemoryStat | None, None]:
        action_begin_stat = self.get_raw_stat_dict()
        last_stat = action_begin_stat
        access_gen = action.get_memory_access()
        memory_access_queue: List[DataflowActionMemoryAccess] = []
        while True:
            local_counter = reduce(
                lambda acc, curr: acc+curr.size, memory_access_queue, 0)
            stat: DataflowActionMemoryStat | None = yield self.get_cache_stat_dict(last_stat)
            if not stat:
                break
            while True:
                if local_counter >= stat.total_count:
                    break
                else:
                    access = next(access_gen)
                    local_counter += access.size
                    memory_access_queue.append(access)

            if local_counter > stat.total_count:
                # split last access to match stat total count
                last_access = memory_access_queue[-1]
                # assume stat.total_count =10
                # access size: 5, 3, 6   local_counter: 14
                # last_size: 6 - (14-10) = 2
                # access size: 5, 3, 2 >>> sum=10 match stat.total_counter
                new_size = last_access.size - \
                    (local_counter-stat.total_count)
                curr_queue = memory_access_queue[:-1]

                curr_queue.append(
                    DataflowActionMemoryAccess(
                        last_access.base_addr, new_size, last_access.rw)
                )

                next_base_addr = last_access.base_addr+new_size
                next_size = last_access.size-new_size
                memory_access_queue = [DataflowActionMemoryAccess(
                    next_base_addr, next_size, last_access.rw)]
            else:
                curr_queue = memory_access_queue
                memory_access_queue = []
            self._process_access(action, curr_queue)
            curr_stat = self.get_raw_stat_dict()
            stat = {}
            for k, v in curr_stat.items():
                stat[k] = v-last_stat[k]

            last_stat = curr_stat

        cost_book = context.cost_dict[action.get_action_id()]

        action_stat = {}
        for k, v in last_stat.items():
            action_stat[k] = v-action_begin_stat[k]

        cost_book.cache_stat_dict[k] = self.get_cache_stat_dict(action_stat)
        logger.debug("cache service - action: %s, stat: %s",
                     action.get_action_id(), cost_book.cache_stat_dict[k])
        yield  # using yield instead of return to avoid from StopIterator exception

    def _process_access(self, action: DataflowAction, access_list: List[DataflowActionMemoryAccess]):
        raise NotImplementedError()

    def get_cache_stat_dict(self) -> Dict[str, CacheStat]:
        raise NotImplementedError()
