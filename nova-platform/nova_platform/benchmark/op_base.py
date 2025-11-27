from dataclasses import dataclass, field
from typing import Generator, List, Tuple, Dict
from functools import lru_cache, reduce
import itertools
import operator

from nova_platform.config import BossaNovaConfig
from nova_platform.base_model import DType

import logging

logger = logging.getLogger(__name__)


@lru_cache
def list_product(input):
    return reduce(operator.mul, input, 1)


@lru_cache
def get_layout(instances):
    layout = []
    for b in range(1, instances + 1):
        for m in range(1, instances + 1):
            if b * m > instances:
                continue
            for n in range(1, instances + 1):
                if b * m * n == instances:
                    layout.append([b, m, n])
    return layout


@dataclass
class GridShape:
    grid_dims: list = field(default_factory=lambda: [1, 1, 1, 1])
    block_dims: list = field(default_factory=lambda: [1, 1, 1, 1])
    thread_dims: list = field(default_factory=lambda: [1, 1, 1, 1])
    subthread_dims: list = field(default_factory=lambda: [1, 1, 1, 1])
    block_row_major: bool = True
    subthread_row_major: bool = True
    block_traverse_idx: list = field(default_factory=list)
    thread_traverse_idx: list = field(default_factory=list)
    subthread_traverse_idx: list = field(default_factory=list)

    def block_cnt_in_grid(self):
        return list_product(tuple(self.grid_dims))

    def thread_cnt_in_block(self):
        return list_product(tuple(self.block_dims))

    def subthread_cnt_in_thread(self):
        return list_product(tuple(self.thread_dims))

    def subthread_dims_stride(self):
        return self.subthread_dims

    def thread_dims_stride(self):
        return [thread_dim * subthread_dim for thread_dim, subthread_dim in zip(self.thread_dims, self.subthread_dims)]

    def block_dims_stride(self):
        return [
            block_dim * thread_dim * subthread_dim
            for block_dim, thread_dim, subthread_dim in zip(self.block_dims, self.thread_dims, self.subthread_dims)
        ]

    def grid_dims_stride(self):
        return [
            grid_dim * block_dim * thread_dim * subthread_dim
            for grid_dim, block_dim, thread_dim, subthread_dim in zip(
                self.grid_dims, self.block_dims, self.thread_dims, self.subthread_dims
            )
        ]

    def gen_traverse_idx(self, traverse_total_size):
        shape_grid = self.grid_dims_stride()
        shape_block = self.block_dims_stride()
        gridLoopNum = [0, 0, 0, 0]
        grid_total = 1
        blockLoopNum = [0, 0, 0, 0]
        block_total = 1
        for i in range(4):
            gridLoopNum[i] = (traverse_total_size[i] + shape_grid[i] - 1) // shape_grid[i]
            grid_total *= gridLoopNum[i]
            blockLoopNum[i] = (traverse_total_size[i] + shape_block[i] - 1) // shape_block[i]
            block_total *= blockLoopNum[i]

        self.block_traverse_idx = [[] for i in range(self.block_cnt_in_grid())]
        # print("!!!!self.block_traverse_idx", self.block_traverse_idx)
        gridFlattenIdxX = 0
        gridFlattenIdxY = 0
        for i in range(gridLoopNum[self.block_traverse_dim_order[3]]):
            for j in range(gridLoopNum[self.block_traverse_dim_order[2]]):
                for k in range(gridLoopNum[self.block_traverse_dim_order[1]]):
                    for l in range(gridLoopNum[self.block_traverse_dim_order[0]]):
                        idx = [0, 0, 0, 0]
                        idx[self.block_traverse_dim_order[0]] = l
                        idx[self.block_traverse_dim_order[1]] = k
                        idx[self.block_traverse_dim_order[2]] = j
                        idx[self.block_traverse_dim_order[3]] = i
                        # snake-like
                        if self.block_traverse_dim_order[0] == 0 and (gridFlattenIdxY & 1):  # row major
                            idx[self.block_traverse_dim_order[0]] = (
                                gridLoopNum[self.block_traverse_dim_order[0]] - l - 1
                            )
                        if self.block_traverse_dim_order[3] == 0 and (gridFlattenIdxX & 1):  # col major
                            idx[self.block_traverse_dim_order[0]] = (
                                gridLoopNum[self.block_traverse_dim_order[0]] - l - 1
                            )
                            idx[self.block_traverse_dim_order[1]] = (
                                gridLoopNum[self.block_traverse_dim_order[1]] - k - 1
                            )
                            idx[self.block_traverse_dim_order[2]] = (
                                gridLoopNum[self.block_traverse_dim_order[2]] - j - 1
                            )
                        c_idx = 0
                        for n in range(self.grid_dims[3]):
                            for z in range(self.grid_dims[2]):
                                for y in range(self.grid_dims[1]):
                                    for x in range(self.grid_dims[0]):
                                        block_idx = [0, 0, 0, 0]
                                        block_idx[0] = idx[0] * self.grid_dims[0] + x
                                        block_idx[1] = idx[1] * self.grid_dims[1] + y
                                        block_idx[2] = idx[2] * self.grid_dims[2] + z
                                        block_idx[3] = idx[3] * self.grid_dims[3] + n
                                        if (
                                            block_idx[0] >= blockLoopNum[0]
                                            or block_idx[1] >= blockLoopNum[1]
                                            or block_idx[2] >= blockLoopNum[2]
                                            or block_idx[3] >= blockLoopNum[3]
                                        ):
                                            block_idx = [-1, -1, -1, -1]
                                        self.block_traverse_idx[c_idx].append(block_idx)
                                        c_idx += 1

                    gridFlattenIdxY += 1
            gridFlattenIdxX += 1

        # gen thread traverse index vec
        threadLoopNum = [0, 0, 0, 0]
        thread_total = 1
        for i in range(4):
            threadLoopNum[i] = self.block_dims[i]
            thread_total *= threadLoopNum[i]
        self.thread_traverse_idx = []
        for i in range(threadLoopNum[3]):
            for j in range(threadLoopNum[2]):
                for k in range(threadLoopNum[1]):
                    for l in range(threadLoopNum[0]):
                        self.thread_traverse_idx.append([l, k, j, i])

        # gen thread traverse index vec
        subthreadLoopNum = [0, 0, 0, 0]
        subthread_total = 1
        for i in range(4):
            subthreadLoopNum[i] = self.thread_dims[i]
            subthread_total *= subthreadLoopNum[i]
        self.subthread_traverse_idx = []
        subthreadFlattenIdxX = 0
        subthreadFlattenIdxY = 0
        for i in range(subthreadLoopNum[self.subthread_traverse_dim_order[3]]):
            for j in range(subthreadLoopNum[self.subthread_traverse_dim_order[2]]):
                for k in range(subthreadLoopNum[self.subthread_traverse_dim_order[1]]):
                    for l in range(subthreadLoopNum[self.subthread_traverse_dim_order[1]]):
                        idx = [0, 0, 0, 0]
                        idx[self.subthread_traverse_dim_order[0]] = l
                        idx[self.subthread_traverse_dim_order[1]] = k
                        idx[self.subthread_traverse_dim_order[2]] = j
                        idx[self.subthread_traverse_dim_order[3]] = i
                        if self.subthread_traverse_dim_order[0] == 0 and (subthreadFlattenIdxY & 1):  # row major
                            idx[self.subthread_traverse_dim_order[0]] = (
                                subthreadLoopNum[self.subthread_traverse_dim_order[0]] - l - 1
                            )
                        if self.subthread_traverse_dim_order[3] == 0 and (subthreadFlattenIdxX & 1):  # col major
                            idx[self.subthread_traverse_dim_order[0]] = (
                                subthreadLoopNum[self.subthread_traverse_dim_order[0]] - l - 1
                            )
                            idx[self.subthread_traverse_dim_order[1]] = (
                                subthreadLoopNum[self.subthread_traverse_dim_order[1]] - k - 1
                            )
                            idx[self.subthread_traverse_dim_order[2]] = (
                                subthreadLoopNum[self.subthread_traverse_dim_order[2]] - j - 1
                            )

                        self.subthread_traverse_idx.append(idx)


@dataclass
class Operand:
    dim: tuple = field(default_factory=tuple)
    addr: int = 0
    bpe: int = 2
    dim_offset: tuple = field(default_factory=tuple)
    dim_stride: tuple = field(default_factory=tuple)
    level: int = 3

    def __hash__(self):
        return hash((tuple(self.dim), tuple(self.dim_offset), tuple(self.dim_stride), self.level, self.addr, self.bpe))

    def get_phy_addr_by_offset(self, off_):
        assert all([o < s for o, s in zip(off_, self.dim_stride)])

        def calculate_dim_stride(shape):
            strides = []
            current_stride = 1
            for size in reversed(shape):
                strides.append(current_stride)
                current_stride *= size
            strides.reverse()
            return strides

        stride = calculate_dim_stride(self.dim_stride)
        offset = 0
        for o, s in zip(off_, stride):
            offset += o * s
        return self.addr + offset * self.bpe

    def get_contiguous_mem_accesses(self):
        if all(o == 0 for o in self.dim_offset) and all([d == s for d, s in zip(self.dim, self.dim_stride)]):
            return [(self.addr, list_product(tuple(self.dim)) * self.bpe)]
        mem_access = []

        size = self.dim[-1] * self.bpe
        dim = list(self.dim[:-1]) + [1]
        for idx in itertools.product(*(range(i) for i in dim)):
            offset = [i + o for i, o in zip(idx, self.dim_offset)]
            addr = self.get_phy_addr_by_offset(offset)
            mem_access.append((addr, size))
        return mem_access


@dataclass()
class Workload:
    inputs: List[Operand] = field(default_factory=list)
    outputs: List[Operand] = field(default_factory=list)
    dtype: DType = DType.FP16
    name: str = ""
    attr: dict = field(default_factory=dict)

    def __hash__(self):
        return hash((tuple(self.inputs), tuple(self.outputs), self.dtype, self.name))


class OpBase:
    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        self.config = config
        self.workload = workload
        self.dtype = self.workload.dtype
