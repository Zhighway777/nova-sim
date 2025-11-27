from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping
from cache_model.entity.model import HardwareConfig, Request, Access, HardwareConfig, No_MSHR
from cache_model.entity.report import DistanceCount
from cache_model.arch import AbstractGCU


class RequestType(str, Enum):
    L1C = "L1C"
    LLC = "LLC"


@dataclass
class SimpleRequest(Request):
    request_type: RequestType | None = None
    sip_id: int = 0


@dataclass
class SimpleAccess(Access):
    request_type: RequestType | None = None
    sip_id: int = 0


class SimpleStrategy():
    def __init__(self, hardware: HardwareConfig, _cls: AbstractGCU):
        self.hardware = hardware
        self.memory_device: AbstractGCU = _cls(self.get_hardware(hardware))
        self.fake_time = 0
        self.queue = []

    def get_hardware(self, hardware: HardwareConfig):
        return hardware

    def process_request(self, request: SimpleRequest):
        try:
            self.memory_device.process(
                request, self.fake_time)

        except No_MSHR as e:
            self.queue.append(request)

    def process(
        self,
        access: SimpleAccess
    ):
        if access.width != 0:
            start_line_addr = int(
                access.address/self.hardware.CACHE_LINE_SIZE)
            end_line_addr = int(
                access.end_address/self.hardware.CACHE_LINE_SIZE)
            for line_addr in range(start_line_addr, end_line_addr+1):
                request = SimpleRequest(
                    access.direction,
                    int(access.address) +
                    (line_addr-start_line_addr) *
                    self.hardware.CACHE_LINE_SIZE,
                    line_addr,
                    thread=None,
                    request_type=access.request_type,
                    sip_id=access.sip_id,
                )
                if self.queue:
                    self.queue.append(request)
                else:
                    self.process_request(request)

    def reuse_distance(
        self,
        access: SimpleAccess
    ):
        if self.queue:
            # TODO: unsafe for multi-threads
            # check performance when hit MSHR limit
            _queue = self.queue
            self.queue = []
            for i, _request in enumerate(_queue):
                if not self.queue:
                    self.process_request(_request)
                else:
                    self.queue.extend(_queue[i:])
                    break

        self.process(access)
        self.memory_device.post_process(self.fake_time)
        self.fake_time += 1

    def finish_queue(self):
        while self.queue:
            if self.queue:
                # TODO: unsafe for multi-threads
                # check performance when hit MSHR limit
                _queue = self.queue
                self.queue = []
                for i, _request in enumerate(_queue):
                    if not self.queue:
                        self.process_request(_request)
                    else:
                        self.queue.extend(_queue[i:])
                        break
            self.memory_device.post_process(self.fake_time)
            self.fake_time += 1

    def histogram(self) -> Mapping[str, List[DistanceCount]]:
        # distance = self.read_distance+self.write_distance
        distance_dict = self.memory_device.histogram_dict()
        his = {}
        for k, distance in distance_dict.items():
            v = [DistanceCount(k, v)
                 for k, v in sorted(distance.items(), reverse=True)]
            his[k] = v

        return his
