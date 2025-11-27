import os
import atexit
from multiprocessing import Process
from multiprocessing import Manager
from queue import Queue
from threading import Thread

from nova_platform.utils.base_utils import check_gil_enabled
from ._core import _BaseTraceGenerator
from typing import Dict, Tuple
from dataclasses import asdict
import logging
logger = logging.getLogger(__name__)


class BaseTrack:
    def __init__(self, name, parent, uuid):
        self._parent: TraceGenerator = parent
        self._uuid = uuid
        self._name = name

    def submit(self, task_type, task_param):
        self._parent.submit(task_type, task_param)


class CounterTrack(BaseTrack):
    def count(self, ts, value):
        # return
        """ Add a count value on the track. """
        ts = int(ts*1e9)
        # pkt = self._parent._track_count(self._uuid, ts, value)
        self.submit('count', {
            "uuid": self._uuid,
            "ts": ts,
            "value": value
        })


class NormalTrack(BaseTrack):
    def __init__(self, name, parent, uuid):
        self._parent: TraceGenerator = parent
        self._uuid = uuid
        self._name = name

    def duration(self, ts, dur, annotation, kwargs=None, flow=[], category_list=[]):
        # kwargs = None
        if kwargs and hasattr(type(kwargs), '__dataclass_fields__'):
            kwargs = asdict(kwargs)

        ts = int(ts*1e9)
        # pkt1 = self._parent._track_open(
        #     self._uuid, ts, annotation, kwargs, flow, category_list=category_list)

        dur = ts+int(dur*1e9)
        # pkt2 = self._parent._track_close(self._uuid, ts, flow)
        self.submit('duration', {
            "uuid": self._uuid,
            "ts": ts,
            "dur": dur,
            "annotation": annotation,
            "kwargs": kwargs,
        })

    def instant(self, ts, annotation, kwargs=None, flow=[], category_list=[]):
        """ Record an instant event. """
        if kwargs and hasattr(type(kwargs), '__dataclass_fields__'):
            kwargs = asdict(kwargs)
        ts = int(ts*1e9)
        # pkt = self._parent._track_instant(
        #     self._uuid, ts, annotation, kwargs, flow, category_list=category_list)
        self.submit("instant", {
            "uuid": self._uuid,
            "ts": ts,
            "annotation": annotation,
            "kwargs": kwargs,
        })


class GroupTrack:
    def __init__(self, name, parent, uuid):
        self._parent: TraceGenerator = parent
        self._uuid = uuid
        self._name = name

    def create_track(self) -> NormalTrack:
        """ Create a child track for this track."""
        return self._parent._create_track(self._uuid, self._name, 0)


class Group:
    def __init__(self, name, parent, uuid):
        self._parent: TraceGenerator = parent
        self._uuid = uuid

    def create_track(self, track_name: str, tid) -> NormalTrack:
        """ Create a normal track for this track."""
        return self._parent._create_track(self._uuid, track_name, 0)

    def create_counter_track(self, track_name: str) -> CounterTrack:
        """ Create a counter track.  Counter tracks can be used for recording int values."""
        return self._parent._create_counter_track(self._uuid, track_name, 1)

    def create_group(self, track_name: str) -> GroupTrack:
        """ Create a group track.  Group tracks can be used for grouping normal tracks."""
        return self._parent._create_track(self._uuid, track_name, 2)


def tg_worker(queue: Queue, filename):
    tg = _BaseTraceGenerator()
    packet_list = []

    def _duration(uuid, ts, dur, annotation, kwargs):
        # _track_open(self, uuid, ts, annotation, kwargs, flow, caller=None, category_list=None):
        pkt1 = tg._track_open(uuid, ts, annotation, kwargs)
        # _track_close(self, uuid, ts, flow):
        pkt2 = tg._track_close(uuid, dur)
        return [pkt1, pkt2]

    class TGExit(Exception):
        pass

    def _exit():
        raise TGExit()

    task_handler_map = {
        'create_header': tg._header_packet,
        'duration': _duration,
        'count': lambda **kwargs: [tg._track_count(**kwargs)],
        'instant': lambda **kwargs: [tg._track_instant(**kwargs)],
        'create_group': lambda **kwargs: [tg._pid_packet(**kwargs)],
        'create_track': lambda **kwargs: [tg._tid_packet(**kwargs)],
        'exit': _exit,
    }

    def flush_packet_list():
        nonlocal packet_list
        trace = tg._make_trace(packet_list)
        f.write(trace.SerializeToString())
        packet_list = []

    with open(filename, 'ab') as f:
        try:
            while True:
                # count, duration, instant
                task_name, param = queue.get()
                handler = task_handler_map[task_name]
                pkts = handler(**param)
                packet_list.extend(pkts)

                if len(packet_list) > 1000:
                    flush_packet_list()
        except TGExit as e:
            if packet_list:
                flush_packet_list()
            logger.info('tg exit')


# manager = None
# if check_gil_enabled():
#     manager = Manager()  # GIL=1 时使用多进程manager


class TraceGenerator(_BaseTraceGenerator):
    def __init__(self, filename: str, gcu_id):
        """ Create a trace """
        super().__init__()
        self.__pid__ = 1
        self.tgen_cluster: Dict[Tuple[int, int], Group] = {}
        self.track_map = {}

        # if check_gil_enabled():  # GIL=1
        use_MP_env = os.getenv('BOSSANOVA_PROTOBUF_MP', '1') == '1'
        logger.info(f"use_MP: {use_MP_env}")
        self.manager = None
        use_mp_runtime = use_MP_env
        if use_mp_runtime:
            try:
                self.manager = Manager()
                self.queue = self.manager.Queue()
                self.worker = Process(target=tg_worker, args=(
                    self.queue, filename), name=f"TG-{gcu_id}")

                def cleanup():
                    try:
                        self.worker.terminate()
                    except Exception as e:
                        logger.error(e)
                    if self.manager:
                        try:
                            self.manager.shutdown()
                        except Exception as e:
                            logger.error(e)

                atexit.register(cleanup)
                self.worker.start()
            except Exception as exc:
                logger.warning(
                    "Falling back to thread-based TraceGenerator: %s", exc)
                self.manager = None
                use_mp_runtime = False

        if not use_mp_runtime:
            self.queue = Queue()
            self.worker = Thread(target=tg_worker, args=(
                self.queue, filename), name=f"TG-{gcu_id}", daemon=True)
            self.worker.start()
        self.submit('create_header', {})

    def __deepcopy__(self, memo):
        # deepcopy逻辑，None表示跳过
        return None
    
    def block_until_all_tasks_done(self):
        self.submit('exit', {})
        self.worker.join()
        # return

    def submit(self, task_type, task_param):
        self.queue.put((task_type, task_param))

    def _create_group(self, process_name: str, track_name: str = None):
        """ Create a group.  Each "group" comes with a default normal track (named track_name)."""
        pid = self.__pid__
        self.__pid__ += 1

        uuid = self.__uuid__
        self.__uuid__ += 1

        # pkt = self._pid_packet(uuid, pid, process_name, track_name)
        self.submit('create_group', {
            "uuid": uuid,
            "pid": pid,
            "process_name": process_name,
            "track_name": track_name
        })

        return Group(process_name, self, uuid)

    def _create_track(self, parent_uuid, track_name, ttype):
        tid = self.__pid__
        self.__pid__ += 1

        uuid = self.__uuid__
        self.__uuid__ += 1

        # uuid = 0
        # uuid, pkt = self._tid_packet(uuid, tid, parent_uuid, track_name, ttype)
        self.submit("create_track", {
            "uuid": uuid,
            "tid": tid,
            "parent_uuid": parent_uuid,
            "track_name": track_name,
            "ttype": ttype
        })
        if ttype == 0:
            return NormalTrack(track_name, self, uuid)
        elif ttype == 1:
            return CounterTrack(track_name, self, uuid)
        elif ttype == 2:
            return GroupTrack(track_name, self, uuid)
        else:
            assert False

    def _create_counter_track(self, parent_uuid, track_name, ttype) -> CounterTrack:
        k = (parent_uuid, track_name, ttype)
        if k not in self.track_map:
            self.track_map[k] = self._create_track(
                parent_uuid, track_name, ttype)
        return self.track_map[k]

    def get_cluster_tgen(self, die_id: int, cid: int) -> Group:
        if (die_id, cid) not in self.tgen_cluster:
            if die_id == None and cid == None:
                group_name = f"Global"
            elif cid == None:
                group_name = f"Die:{die_id}, Global"
            else:
                group_name = f"Die:{die_id}, Cluster:{cid}"
            self.tgen_cluster[(die_id, cid)] = self._create_group(
                group_name)
        return self.tgen_cluster[(die_id, cid)]
