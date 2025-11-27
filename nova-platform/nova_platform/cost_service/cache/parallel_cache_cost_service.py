from collections import Counter
import numpy as np
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Generator, List
from nova_platform.base_model import DataflowActionMemoryAccess
from nova_platform.cost_service.compute.base_compute_model import BossaNovaContext, DataflowAction
from nova_platform.config import AbstractCacheConfig, BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BaseCostService
from nova_platform.cost_service.cache.base_cache_model import BaseCacheCostService, CacheStat
from nova_platform.dataflow.dataflow import Dataflow

from pycuda.compiler import SourceModule
import pycuda.driver as cuda

import logging
logger = logging.getLogger(__name__)

cache_kernel = None
addr_convert_kernel = None


class ParallelCacheCostService(BaseCacheCostService):
    def __init__(self, config: BossaNovaConfig, device_id=0):
        super().__init__(config)
        os.environ['CUDA_DEVICE'] = str(device_id)
        import pycuda.autoinit

        # 编译并加载CUDA模块
        mod = SourceModule(
            Path("nova_platform/cost_service/cache/cache_kernal.cu").read_text())
        # 获取内核函数
        global cache_kernel, addr_convert_kernel
        cache_kernel = mod.get_function("cache_kernel")
        addr_convert_kernel = mod.get_function("addr_convert_kernel")

        # TODO: pass cache directly, may need refactor
        if config.arch_name != "libra":
            raise Exception("arch %s is not supported", config.arch_name)
        self.llc_config = self.config.memory.llc

        cfg = asdict(self.config.memory.llc)
        NUM_OF_PARTITIONS = cfg.pop("NUM_OF_PARTITIONS")
        NUM_OF_SLICES_PER_PARTITION = cfg.pop("NUM_OF_SLICES_PER_PARTITION")
        cfg = AbstractCacheConfig(**cfg)
        cfg.CACHE_SIZE = self.llc_config.CACHE_SIZE * \
            NUM_OF_PARTITIONS * \
            NUM_OF_SLICES_PER_PARTITION * \
            config.inst_num.NUM_OF_DIE
        self.llc_cache = ParallelCacheImpl(cfg)
        self.raw_stat_dict = {
            "r_hit": 0,
            "w_hit": 0,
            "r_miss": 0,
            "w_miss": 0
        }
    def __deepcopy__(self, memo):
        # deepcopy逻辑，None表示跳过
        return None
    
    def _process_access(self, action: DataflowAction, access_list: List[DataflowActionMemoryAccess]):
        # def process(self, action: DataflowAction, context: BossaNovaContext, ref: float) -> Generator[bool, None, None]:
        base_addr_list = []
        size_list = []
        rw_list = []
        total_size = 0
        CACHE_LINE_SIZE = self.llc_config.CACHE_LINE_SIZE
        for access in access_list:
            base_addr = int(access.base_addr)
            # TODO: need refactor
            if base_addr < 5*2**40:
                continue
            base_addr_list.append(base_addr)
            line_addr_start = base_addr//CACHE_LINE_SIZE
            line_addr_end = int(base_addr+access.size-1)//CACHE_LINE_SIZE
            line_size = line_addr_end-line_addr_start+1
            size_list.append(line_size)
            rw_list.append(0 if access.rw == 'r' else 1)
            total_size += line_size

        if base_addr_list:
            r_hit, w_hit, r_miss, w_miss = self.llc_cache.process(
                base_addr_list, size_list, rw_list, total_size)
            self.raw_stat_dict = {
                "r_hit": self.raw_stat_dict['r_hit'] + r_hit,
                "w_hit": self.raw_stat_dict['w_hit'] + w_hit,
                "r_miss": self.raw_stat_dict['r_miss'] + r_miss,
                "w_miss": self.raw_stat_dict['w_miss'] + w_miss
            }

    def get_cache_stat_dict(self) -> Dict[str, CacheStat]:
        stat_dict = {}
        r_hit = self.raw_stat_dict["r_hit"]
        w_hit = self.raw_stat_dict["w_hit"]
        r_miss = self.raw_stat_dict["r_miss"]
        w_miss = self.raw_stat_dict["w_miss"]
        stat = CacheStat(
            w_hit+w_miss,
            w_hit,
            r_hit+r_miss,
            r_hit,
        )
        stat_dict["LLC"] = stat
        return stat_dict

    def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):
        stat = self.llc_cache.stat_dict()

        return {
            "LLC": stat
        }


class ParallelCacheImpl():
    def __init__(self, cache_config: AbstractCacheConfig):
        self.cache_line_size = cache_config.CACHE_LINE_SIZE
        self.cache_ways = cache_config.CACHE_WAYS
        assert cache_config.CACHE_SIZE % self.cache_line_size == 0 and cache_config.CACHE_SIZE % self.cache_line_size % self.cache_ways == 0

        self.cache_sets = cache_config.CACHE_SIZE//self.cache_ways//self.cache_line_size

        self.cache_cell = np.zeros(
            self.cache_sets*self.cache_ways, dtype=np.int64)
        self.cache_cell[:] = -1
        self.cache_cell_offset = np.zeros(self.cache_sets, dtype=np.int64)
        self.cache_cell_offset[:] = -1
        self.read_hits_per_set = np.zeros(self.cache_sets, dtype=np.int32)
        self.write_hits_per_set = np.zeros(self.cache_sets, dtype=np.int32)
        self.read_misses_per_set = np.zeros(self.cache_sets, dtype=np.int32)
        self.write_misses_per_set = np.zeros(self.cache_sets, dtype=np.int32)
        # allocate gpu memory
        self.cache_cell_gpu = cuda.mem_alloc(self.cache_cell.nbytes)
        self.cache_cell_offset_gpu = cuda.mem_alloc(
            self.cache_cell_offset.nbytes)
        self.read_hits_per_set_gpu = cuda.mem_alloc(
            self.read_hits_per_set.nbytes)
        self.write_hits_per_set_gpu = cuda.mem_alloc(
            self.write_hits_per_set.nbytes)
        self.read_misses_per_set_gpu = cuda.mem_alloc(
            self.read_misses_per_set.nbytes)
        self.write_misses_per_set_gpu = cuda.mem_alloc(
            self.write_misses_per_set.nbytes)
        # h2d
        cuda.memcpy_htod(self.cache_cell_gpu, self.cache_cell)
        cuda.memcpy_htod(self.cache_cell_offset_gpu, self.cache_cell_offset)
        cuda.memcpy_htod(self.read_hits_per_set_gpu, self.read_hits_per_set)
        cuda.memcpy_htod(self.write_hits_per_set_gpu, self.write_hits_per_set)
        cuda.memcpy_htod(self.read_misses_per_set_gpu,
                         self.read_misses_per_set)
        cuda.memcpy_htod(self.write_misses_per_set_gpu,
                         self.write_misses_per_set)

    def process(self, base_addr_list, size_list, rw_list, total_size):
        # step 1: compute line_addr, set_idx, rw
        list_size = len(base_addr_list)
        base_addr_list = np.array(base_addr_list, dtype=np.int64)
        size_list = np.array(size_list, dtype=np.int32)
        rw_list = np.array(rw_list, dtype=np.int32)

        output_set_idx_list = np.zeros(total_size, dtype=np.int32)
        output_line_addr_list = np.zeros(total_size, dtype=np.int64)
        output_rw_list = np.zeros(total_size, dtype=np.int32)

        chunk_size = 1024

        set_num = self.cache_sets

        base_addr_list_gpu = cuda.mem_alloc(base_addr_list.nbytes)
        size_list_gpu = cuda.mem_alloc(size_list.nbytes)
        rw_list_gpu = cuda.mem_alloc(rw_list.nbytes)

        output_set_idx_list_gpu = cuda.mem_alloc(output_set_idx_list.nbytes)
        output_line_addr_list_gpu = cuda.mem_alloc(
            output_line_addr_list.nbytes)
        output_rw_list_gpu = cuda.mem_alloc(output_rw_list.nbytes)

        cuda.memcpy_htod(base_addr_list_gpu, base_addr_list)
        cuda.memcpy_htod(size_list_gpu, size_list)
        cuda.memcpy_htod(rw_list_gpu, rw_list)
        block_size = 256
        grid_size = ((total_size+chunk_size-1)//chunk_size +
                     block_size - 1) // block_size
        global addr_convert_kernel, cache_kernel
        addr_convert_kernel(
            base_addr_list_gpu,
            size_list_gpu,
            rw_list_gpu,
            np.int32(list_size),
            output_set_idx_list_gpu,
            output_line_addr_list_gpu,
            output_rw_list_gpu,
            np.int32(total_size),
            np.int32(chunk_size),
            np.int32(set_num),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        cuda.memcpy_dtoh(output_set_idx_list, output_set_idx_list_gpu)
        cuda.memcpy_dtoh(output_line_addr_list, output_line_addr_list_gpu)
        cuda.memcpy_dtoh(output_rw_list, output_rw_list_gpu)

        base_addr_list_gpu.free()
        size_list_gpu.free()
        rw_list_gpu.free()

        # step2: compute hit rate
        block_size = 256
        grid_size = (self.cache_sets + block_size - 1) // block_size

        access_hit_flag = np.zeros(total_size, dtype=np.int32)

        # 分配GPU内存
        access_hit_flag_gpu = cuda.mem_alloc(access_hit_flag.nbytes)
        cuda.memcpy_htod(access_hit_flag_gpu, access_hit_flag)

        cache_kernel(
            output_set_idx_list_gpu,
            output_line_addr_list_gpu,
            output_rw_list_gpu,
            np.int32(total_size),
            access_hit_flag_gpu,
            self.cache_cell_gpu,
            self.cache_cell_offset_gpu,
            self.read_hits_per_set_gpu,
            self.write_hits_per_set_gpu,
            self.read_misses_per_set_gpu,
            self.write_misses_per_set_gpu,
            np.int32(set_num),
            np.int32(self.cache_ways),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        cuda.memcpy_dtoh(access_hit_flag, access_hit_flag_gpu)

        unique, counts = np.unique(access_hit_flag, return_counts=True)
        stat = dict(zip(unique, counts))
        access_hit_flag_gpu.free()

        output_set_idx_list_gpu.free()
        output_line_addr_list_gpu.free()
        output_rw_list_gpu.free()

        r_hit = stat.get(1, 0)
        w_hit = stat.get(-1, 0)
        r_miss = stat.get(2, 0)
        w_miss = stat.get(-2, 0)

        return int(r_hit), int(w_hit), int(r_miss), int(w_miss)

    def stat_dict(self):
        cuda.memcpy_dtoh(self.read_hits_per_set, self.read_hits_per_set_gpu)
        cuda.memcpy_dtoh(self.write_hits_per_set, self.write_hits_per_set_gpu)
        cuda.memcpy_dtoh(self.read_misses_per_set,
                         self.read_misses_per_set_gpu)
        cuda.memcpy_dtoh(self.write_misses_per_set,
                         self.write_misses_per_set_gpu)
        read_hits = np.sum(self.read_hits_per_set)
        write_hits = np.sum(self.write_hits_per_set)
        read_misses = np.sum(self.read_misses_per_set)
        write_misses = np.sum(self.write_misses_per_set)
        return {
            "read_hits": int(read_hits),
            "read_misses": int(read_misses),
            "write_hits": int(write_hits),
            "write_misses": int(write_misses),
            "read_hit_rate": float(read_hits/(read_hits+read_misses)) if read_hits+read_misses else 0,
            "write_hit_rate": float(write_hits/(write_hits+write_misses)) if write_hits+write_misses else 0,
        }
