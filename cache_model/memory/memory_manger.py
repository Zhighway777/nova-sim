from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Callable, Tuple

from cache_model.entity.model import L1C_Config, LLC_Config, ModelContext, Request
from cache_model.memory import AbstractMemoryManager


class _LRUSet:
    def __init__(self, ways: int):
        self._ways = max(1, ways)
        self._lines: OrderedDict[int, None] = OrderedDict()

    def access(self, tag: int) -> bool:
        hit = tag in self._lines
        if hit:
            self._lines.move_to_end(tag)
        else:
            if len(self._lines) >= self._ways:
                self._lines.popitem(last=False)
            self._lines[tag] = None
        return hit


class _BaseCacheManager(AbstractMemoryManager):
    def __init__(self, config, context: ModelContext, next_level=None):
        super().__init__(config, context, next_level=next_level)
        self._stat = Counter()
        self._hist = Counter()
        self._last_access = {}

    def _record(self, request: Request, hit: bool):
        op = "read" if request.direction == 0 else "write"
        key = f"{op}_{'hits' if hit else 'misses'}"
        self._stat[key] += 1
        last = self._last_access.get((request.sip_id, request.line_addr))
        if last is not None and hit:
            distance = max(1, self.context.timestamp - last)
            self._hist[distance] += 1
        self._last_access[(request.sip_id, request.line_addr)] = self.context.timestamp

    def post_process(self, timestamp):
        super().post_process(timestamp)
        self.context.timestamp = timestamp

    def stat(self) -> Counter:
        return Counter(self._stat)

    def histogram(self) -> Counter:
        return Counter(self._hist)


class L3Manager(_BaseCacheManager):
    def __init__(self, config, context: ModelContext):
        super().__init__(config, context, next_level=None)

    def process(self, request: Request):
        # Model L3 as an always-hit terminal level.
        self._record(request, True)


class LLCManager(_BaseCacheManager):
    def __init__(
        self,
        config: LLC_Config,
        context: ModelContext,
        next_level: L3Manager,
        addr_convert: Callable[[int], int] | None = None,
    ):
        super().__init__(config, context, next_level=next_level)
        self._addr_convert = addr_convert or (lambda addr: 0)
        self._total_slices = max(
            1, config.NUM_OF_PARTITIONS * config.NUM_OF_SLICES_PER_PARTITION
        )
        sets_per_slice = max(
            1, config.CACHE_SIZE // config.CACHE_LINE_SIZE // config.CACHE_WAYS
        )
        self._sets = [
            [_LRUSet(config.CACHE_WAYS) for _ in range(sets_per_slice)]
            for _ in range(self._total_slices)
        ]
        self._sets_per_slice = sets_per_slice

    def process(self, request: Request):
        slice_idx = self._addr_convert(request.address) % self._total_slices
        set_idx = request.line_addr % self._sets_per_slice
        cache_set = self._sets[slice_idx][set_idx]
        hit = cache_set.access(request.line_addr)
        if not hit and self.next_level:
            self.next_level.process(request)
        self._record(request, hit)


class L1CManager(_BaseCacheManager):
    def __init__(
        self,
        config: L1C_Config,
        context: ModelContext,
        next_level: LLCManager,
        cache_selector: Callable[[Request, int], Tuple[int, int]],
    ):
        super().__init__(config, context, next_level=next_level)
        self._cache_selector = cache_selector
        self._num_sets = max(1, config.CACHE_SIZE // config.CACHE_LINE_SIZE)
        self._num_slices = max(1, config.NUM_OF_SIP)
        self._ways = max(1, config.CACHE_WAYS)
        self._sets = [
            [_LRUSet(self._ways) for _ in range(self._num_sets)]
            for _ in range(self._num_slices)
        ]

    def process(self, request: Request):
        slice_idx, set_idx = self._cache_selector(request, self._num_sets)
        slice_idx %= self._num_slices
        set_idx %= self._num_sets
        cache_set = self._sets[slice_idx][set_idx]
        hit = cache_set.access(request.line_addr)
        if not hit and self.next_level:
            self.next_level.process(request)
        self._record(request, hit)
 

__all__ = ["L1CManager", "L3Manager", "LLCManager"]
