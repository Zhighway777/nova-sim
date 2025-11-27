from threading import Lock, Thread
from time import sleep
from dataclasses import dataclass
from functools import partial, partialmethod
from typing import Dict
import logging

from nova_platform.utils.base_utils import SingletonMeta
logger = logging.getLogger(__file__)


class BARRIER():
    def __init__(self, count):
        self.count = count
        self.barrier = [i for i in range(count)]
        self.barrier_map = {b: False for b in self.barrier}
        self.max_t = 0
        self.is_done = False
        self.lock = Lock()

    def get_barrier(self):
        b = self.barrier.pop()
        self.barrier_map[b] = False
        return partial(self.__wait, b)

    def __wait(self, barrier_obj, ref):
        self.lock.acquire()
        self.max_t = max(self.max_t, ref)
        del self.barrier_map[barrier_obj]
        self.lock.release()
        while True:
            if self.barrier_map:
                yield self
            else:
                self.is_done = True
                break
        return self.max_t

    def wait(self, ref):
        b = self.get_barrier()
        max_t = yield from b(ref)
        return max_t


_inst_map: Dict[any, BARRIER] = {}


class BarrierManager(metaclass=SingletonMeta):
    def __init__(self):
        self.lock = Lock()

    def get_barrier(self, uuid, count) -> BARRIER:
        if uuid not in _inst_map:
            self.lock.acquire()
            if uuid not in _inst_map:
                _inst_map[uuid] = BARRIER(count)
            self.lock.release()
        return _inst_map[uuid]


def check():
    inflight_count = 0
    done_count = 0
    if done_count == 0 and inflight_count == 0:
        return
    for uuid in list(_inst_map.keys()):
        barrier = _inst_map[uuid]
        if not barrier.is_done:
            logger.warning("barrier: %s, wait %d out of %d", uuid, len(
                barrier.barrier_map), barrier.count)
            inflight_count += 1
        else:
            done_count += 1

    logger.info("barrier: inflgiht: %d, done: %d", inflight_count, done_count)


def threaded_function():
    logger.info("barrier: check thread started")
    while True:
        check()
        sleep(2)


thread = Thread(target=threaded_function, args=(),
                daemon=True, name="Barrier Check Thread")
thread.start()
