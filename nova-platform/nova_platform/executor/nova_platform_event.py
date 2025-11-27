from typing import Dict
from nova_platform.utils.base_utils import SingletonMeta
import logging
from threading import Lock, Thread
from time import sleep

logger = logging.getLogger(__file__)


class BossaNovaEvent():

    def __init__(self, uuid):
        super().__init__()
        self.is_done = False
        self.max_t = 0
        self.uuid = uuid
        self.lock = Lock()

    def set(self, ref):
        assert self.is_done == False
        self.max_t = max(ref, self.max_t)
        self.is_done = True

    def wait(self, ref):
        self.max_t = ref
        while True:
            if not self.is_done:
                yield self
            else:
                break
        return self.max_t


_inst_map: Dict[any, BossaNovaEvent] = {}


class EventManager(metaclass=SingletonMeta):
    def __init__(self):
        self.Lock = Lock()

    def get_event(self, uuid) -> BossaNovaEvent:
        if uuid not in _inst_map:
            self.Lock.acquire()
            if uuid not in _inst_map:
                _inst_map[uuid] = BossaNovaEvent(uuid)
            self.Lock.release()

        return _inst_map[uuid]


def check():
    inflight_count = 0
    done_count = 0
    if done_count == 0 and inflight_count == 0:
        return
    for uuid in list(_inst_map.keys()):
        event = _inst_map[uuid]
        if not event.is_done:
            logger.warning("event: wait %s", uuid)
            inflight_count += 1
        else:
            done_count += 1

    logger.info("event: inflgiht: %d, done: %d", inflight_count, done_count)


def threaded_function():
    logger.info("event: check thread started")
    while True:
        check()
        sleep(2)


thread = Thread(target=threaded_function, args=(),
                daemon=True, name="Event Check Thread")
thread.start()
