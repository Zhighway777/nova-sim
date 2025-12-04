from typing import Dict, Tuple, List
import math

from nova_platform.config import BossaNovaConfig
from nova_platform.benchmark.op_base import Workload, Operand, list_product
from nova_platform.benchmark.batch_gemm import BatchGemmBase, BatchGemmGridShape


class BatchGemmTpu(BatchGemmBase):
    """
    TPU GEMM tiler：基于阵列尺寸的简易分块，考虑 SRAM 容量与 HBM 带宽。
    """

    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        super().__init__(config, workload)
        self.shape_list: List[Tuple[int, BatchGemmGridShape]] = []
        self._best_shape = None
        compute = getattr(config, "compute", None)
        memory = getattr(config, "memory", None)
        bw = getattr(config, "bw", None)
        freq = getattr(config, "freq", None)

        freq_core = getattr(freq, "CORE", 1.0) if freq else 1.0
        self.array_m = getattr(getattr(compute, "tpu", None), "ARRAY_M", 64) if compute else 64
        self.array_n = getattr(getattr(compute, "tpu", None), "ARRAY_N", 64) if compute else 64
        self.k_tile = getattr(getattr(compute, "tpu", None), "K_TILE", 128) if compute else 128
        # array_flops: macs/s -> 2 flops per mac handled in cost
        self.array_flops = self.array_m * self.array_n * freq_core * 1e9
        # SRAM 容量
        self.sram_bytes = getattr(getattr(memory, "l1", None), "SIZE_PER_CORE", 8 * 1024 * 1024) if memory else 8 * 1024 * 1024
        # HBM 带宽
        hbm_bw_cfg = getattr(getattr(getattr(bw, "mc", None), "l3", None), "bw", 64) if bw else 64
        freq_mc = getattr(freq, "MC", freq_core) if freq else freq_core
        num_die = getattr(getattr(config, "inst_num", None), "NUM_OF_DIE", 1)
        self.hbm_bw = hbm_bw_cfg * freq_mc * 1e9 * num_die

    def split(self):
        candidates = []
        idx = 0
        tile_m_opts = [self.array_m, min(self.array_m * 2, self.in_m)]
        tile_n_opts = [self.array_n, min(self.array_n * 2, self.in_n)]
        tile_k_opts = [self.k_tile, min(self.k_tile * 2, self.in_k)]

        for tm in tile_m_opts:
            for tn in tile_n_opts:
                for tk in tile_k_opts:
                    if tm <= 0 or tn <= 0 or tk <= 0:
                        continue
                    if tm > self.in_m or tn > self.in_n or tk > self.in_k:
                        continue
                    if not self._fit_sram(tm, tn, tk):
                        continue
                    grid_m = math.ceil(self.in_m / tm)
                    grid_n = math.ceil(self.in_n / tn)
                    grid_b = math.ceil(self.in_batch / 1)
                    shape = BatchGemmGridShape(
                        grid_dims=[grid_n, grid_m, grid_b, 1],
                        block_dims=[1, 1, 1, 1],
                        thread_dims=[1, 1, 1, 1],
                        subthread_dims=[1, 1, 1, 1],
                        lhs_buf_cnt_l2=1,
                        rhs_buf_cnt_l2=1,
                        res_buf_cnt_l2=1,
                        lhs_buf_cnt_l1=1,
                        rhs_buf_cnt_l1=1,
                        res_buf_cnt_l1=1,
                        calc_ceil_K_l2=tk,
                        calc_ceil_K_l1=tk,
                    )
                    shape.sol_cost = self._calc_sol_cost(shape, tm, tn, tk)
                    candidates.append((idx, shape))
                    idx += 1
        if not candidates:
            shape = BatchGemmGridShape(
                grid_dims=[1, 1, 1, 1],
                block_dims=[1, 1, 1, 1],
                thread_dims=[1, 1, 1, 1],
                subthread_dims=[1, 1, 1, 1],
                lhs_buf_cnt_l2=1,
                rhs_buf_cnt_l2=1,
                res_buf_cnt_l2=1,
                lhs_buf_cnt_l1=1,
                rhs_buf_cnt_l1=1,
                res_buf_cnt_l1=1,
                calc_ceil_K_l2=self.in_k,
                calc_ceil_K_l1=self.in_k,
            )
            shape.sol_cost = self._calc_sol_cost(shape, self.in_m, self.in_n, self.in_k)
            candidates.append((0, shape))
        self.shape_list = sorted(candidates, key=lambda x: x[1].sol_cost)

    def _fit_sram(self, tm: int, tn: int, tk: int) -> bool:
        lhs_bytes = tm * tk * self.bpe["lhs"]
        rhs_bytes = tn * tk * self.bpe["rhs"]
        res_bytes = tm * tn * self.bpe["res"]
        return lhs_bytes + rhs_bytes + res_bytes <= self.sram_bytes

    def _calc_sol_cost(self, shape: BatchGemmGridShape, tm: int, tn: int, tk: int):
        tiles_m = math.ceil(self.in_m / tm)
        tiles_n = math.ceil(self.in_n / tn)
        tiles_k = math.ceil(self.in_k / tk)
        tiles_b = math.ceil(self.in_batch / 1)
        tiles = tiles_m * tiles_n * tiles_k * tiles_b
        macs_per_tile = tm * tn * tk
        compute = (macs_per_tile * 2) / (self.array_flops or 1.0)
        bytes_per_tile = tm * tk * self.bpe["lhs"] + tn * tk * self.bpe["rhs"] + tm * tn * self.bpe["res"]
        mem = bytes_per_tile / (self.hbm_bw or 1.0)
        return max(compute, mem) * tiles

    def _gen_workloads(self, batch_gemm_shape: BatchGemmGridShape, first_sip_only=False):
        workloads: Dict[int, Dict[int, List[Workload]]] = {}
        tm = min(self.array_m * 2, self.in_m)
        tn = min(self.array_n * 2, self.in_n)
        tk = batch_gemm_shape.calc_ceil_K_l2

        lhs_tensor = self.workload.inputs[0]
        rhs_tensor = self.workload.inputs[1]
        res_tensor = self.workload.outputs[0]

        grid_n, grid_m, grid_b, _ = batch_gemm_shape.grid_dims
        sic_idx = 0
        sip_idx = 0
        workloads.setdefault(sic_idx, {}).setdefault(sip_idx, [])
        for bz in range(grid_b):
            for my in range(grid_m):
                for nx in range(grid_n):
                    m_offset = my * tm
                    n_offset = nx * tn
                    b_offset = bz * 1
                    m_valid = min(tm, self.in_m - m_offset)
                    n_valid = min(tn, self.in_n - n_offset)
                    k_processed = 0
                    while k_processed < self.in_k:
                        k_valid = min(tk, self.in_k - k_processed)
                        lhs = Operand(
                            dim=(1, m_valid, k_valid),
                            addr=lhs_tensor.addr,
                            bpe=lhs_tensor.bpe,
                            dim_offset=(b_offset, m_offset, k_processed),
                            dim_stride=lhs_tensor.dim_stride,
                        )
                        rhs = Operand(
                            dim=(1, k_valid, n_valid),
                            addr=rhs_tensor.addr,
                            bpe=rhs_tensor.bpe,
                            dim_offset=(b_offset, k_processed, n_offset),
                            dim_stride=rhs_tensor.dim_stride,
                        )
                        outs = []
                        if k_processed + k_valid >= self.in_k:
                            outs.append(
                                Operand(
                                    dim=(1, m_valid, n_valid),
                                    addr=res_tensor.addr,
                                    bpe=res_tensor.bpe,
                                    dim_offset=(b_offset, m_offset, n_offset),
                                    dim_stride=res_tensor.dim_stride,
                                )
                            )
                        attr = {
                            **self.workload.attr,
                            "b": 1,
                            "m": m_valid,
                            "n": n_valid,
                            "k": k_valid,
                            "tile_m": tm,
                            "tile_n": tn,
                            "tile_k": tk,
                            "bytes_lhs": list_product((1, m_valid, k_valid)) * lhs.bpe,
                            "bytes_rhs": list_product((1, k_valid, n_valid)) * rhs.bpe,
                            "bytes_res": list_product((1, m_valid, n_valid)) * (res_tensor.bpe if outs else 0),
                        }
                        workloads[sic_idx][sip_idx].append(Workload(inputs=[lhs, rhs], outputs=outs, attr=attr, dtype=self.workload.dtype))
                        k_processed += k_valid
        return workloads

    def _sort_shape_candidates(self, shape_list):
        return sorted(shape_list, key=lambda x: x[1].sol_cost)

    def impl(self):
        if not self.shape_list:
            self.split()
        self.shape_list = self._sort_shape_candidates(self.shape_list)
        self._best_shape = self.get_best_shape()
        self._tiled_workloads = self._gen_workloads(self._best_shape)

    def get_best_shape(self) -> BatchGemmGridShape:
        return self.shape_list[0][1]


def tile_tpu_gemm_workload(config: BossaNovaConfig, chip_workload: Workload) -> Tuple[Dict, BatchGemmGridShape]:
    gemm = BatchGemmTpu(config, chip_workload)
    gemm.split()
    gemm.impl()
    return gemm.get_tiled_workloads(), gemm.get_best_shape()
