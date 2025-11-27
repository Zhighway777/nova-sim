from dataclasses import dataclass, field
from typing import Generator, List, Tuple, Dict
from collections import deque
import itertools
from enum import Enum

from nova_platform.benchmark.op_base import Workload, OpBase, Operand, GridShape, list_product
from nova_platform.benchmark.utils import _iter_access_gen
from nova_platform.config import BossaNovaConfig
from nova_platform.base_model import DType

import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchGemmGridShape(GridShape):
    lhs_buf_cnt_l2: int = 0
    rhs_buf_cnt_l2: int = 0
    res_buf_cnt_l2: int = 0
    block_row_major: bool = True
    bias_buf_cnt_l2: int = 0
    lhs_buf_cnt_l1: int = 0
    rhs_buf_cnt_l1: int = 0
    res_buf_cnt_l1: int = 0
    subthread_row_major: bool = True
    bias_buf_cnt_l1: int = 0
    calc_ceil_K_l2: int = 0
    calc_ceil_K_l1: int = 0

    mem_access_l2_per_sip: int = 0
    mem_access_l1_per_sip: int = 0
    mem_access_total_l3: int = 0
    imax_l3: float = 0.0
    attaible_flops: float = 0.0
    l1_bw_required: float = 0.0  # bytes/cycle
    sol_cost: float = 0.0

    def __post_init__(self):
        self.block_traverse_dim_order = [0, 1, 2, 3]
        self.subthread_traverse_dim_order = [0, 1, 2, 3]
        if not self.block_row_major:
            self.block_traverse_dim_order[0] = 1
            self.block_traverse_dim_order[1] = 0
        if not self.subthread_row_major:
            self.subthread_traverse_dim_order[0] = 1
            self.subthread_traverse_dim_order[1] = 0
        self.tiled_workloads = None

class MemArchType(str, Enum):
    DSM_LOCAL = "dsm.local"
    DSM_SHARED = "dsm.shared"


class QuantType(str, Enum):
    No_Quant = "No_Quant"
    Wf8t_Af8t = "Wf8t_Af8t"  # weight fp8 per tensor quant, activation fp8 per tensor quant
    Wf4g_Af8t = "Wf4g_Af8t"  # weight fp4 per group quant,  activation fp8 per tensor quant
    Wf4g_Af8k = "Wf4g_Af8k"  # weight fp4 per group quant,  activation fp8 per token quant


def bpe_for_quant(dtype: DType, quant_type: QuantType):
    orig_bpe = dtype.get_bpe()
    if quant_type == QuantType.No_Quant:
        return {"lhs": orig_bpe, "rhs": orig_bpe, "res": orig_bpe, "scaling": orig_bpe, "bias": orig_bpe}
    elif quant_type == QuantType.Wf8t_Af8t:
        return {"lhs": 1, "rhs": 1, "res": orig_bpe, "scaling": 2, "bias": 4}
    elif quant_type == QuantType.Wf4g_Af8t:
        return {"lhs": 1, "rhs": 0.5, "res": orig_bpe, "scaling": 2, "bias": 4}
    else:
        raise RuntimeError(f"unsupported quant type {quant_type}")


def compute_dtype_for_quant(dtype: DType, quant_type: QuantType):
    if quant_type == QuantType.No_Quant:
        return {"1d": dtype, "2d": dtype}
    elif quant_type == QuantType.Wf8t_Af8t:
        return {"1d": DType.FP32, "2d": DType.FP8}
    elif quant_type == QuantType.Wf4g_Af8t:
        return {"1d": DType.FP32, "2d": DType.FP4}
    else:
        raise RuntimeError(f"unsupported quant type {quant_type}")


class BatchGemmBase(OpBase):

    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        super().__init__(config, workload)
        self.in_batch = workload.inputs[0].dim[0]
        self.in_m = workload.inputs[0].dim[1]
        self.in_n = workload.inputs[1].dim[2]
        self.in_k = workload.inputs[1].dim[1]
        # print("###", workload)
        self.mem_arch_type = MemArchType.DSM_LOCAL
        self.shape_list = []
        self.quant_type = QuantType(workload.attr.get("quant_type", "No_Quant"))

        self.bpe = bpe_for_quant(self.dtype, self.quant_type)
        self.compute_dtype = compute_dtype_for_quant(self.dtype, self.quant_type)
        self.has_bias = workload.attr.get("has_bias", False)
        if not self.has_bias:
            self.bpe["bias"] = 0
        self.sic_cnt = config.inst_num.NUM_OF_CLUSTER * config.inst_num.NUM_OF_DIE
        self.sip_cnt = config.inst_num.NUM_OF_CORE_PER_CLUSTER
        self.l1_bytes_per_sip = config.memory.l1.SIZE_PER_CORE
        self.l2_bytes_per_sic = config.memory.l2.SIZE_PER_SIC
        self.xpu_FLOPS = (
            config.compute.thread_2d_throughput[self.compute_dtype["2d"]] * 2 * config.freq.CORE * 1e9
        )  # ops/s
        self.l3_bandwidth = config.bw.mc.l3.bw * config.freq.MC * config.inst_num.NUM_OF_DIE * 1e9  # bytes/s
        self.l3_bandwidth_per_sip = self.l3_bandwidth / self.sic_cnt / self.sip_cnt
        if "libra" in str(config.name).lower():
            self.calc_ceil_info = [([64, 64, 1, 1], 8), ([16, 32, 1, 1], 4)]
        else:
            self.calc_ceil_info = [([64, 64, 1, 1], 16), ([8, 8, 1, 1], 16)]
        # self.calc_ceil_info = [([64, 64, 1, 1], 8)]
        self.imax = self.xpu_FLOPS * self.sic_cnt * self.sip_cnt / self.l3_bandwidth
        self.l3_to_l1_latency = 512 / config.freq.CORE / 1e9
        self.l0_to_l3_latency = 512 / config.freq.CORE / 1e9
        self.l1_bandwidth_per_sip = config.bw.l0.local.bw * config.freq.CORE * 1e9

    def gen_buf_cnt_combines(self, grid_dims, block_dims, thread_dims, subthread_dims, l2K, l1K):
        B = self.in_batch
        M = self.in_m
        N = self.in_n
        K = self.in_k

        block_stride_x = block_dims[0] * thread_dims[0] * subthread_dims[0]
        block_stride_y = block_dims[1] * thread_dims[1] * subthread_dims[1]
        block_stride_z = block_dims[2] * thread_dims[2] * subthread_dims[2]
        grid_stride_x = block_stride_x * grid_dims[0]
        grid_stride_y = block_stride_y * grid_dims[1]
        grid_stride_z = block_stride_z * grid_dims[2]
        block_loop_x = (N + grid_stride_x - 1) // grid_stride_x
        block_loop_y = (M + grid_stride_y - 1) // grid_stride_y
        block_loop_z = (B + grid_stride_z - 1) // grid_stride_z
        block_total = block_loop_x * block_loop_y * block_loop_z  # block num on each sic

        # make thread's lhs/rhs cached in L1
        rhs_thread_cnt_l1 = thread_dims[0] * thread_dims[2]
        lhs_thread_cnt_l1 = thread_dims[1] * thread_dims[2]
        if block_loop_y * block_loop_z * thread_dims[1] * thread_dims[2] > 1:
            lhs_thread_cnt_l1 = lhs_thread_cnt_l1 if lhs_thread_cnt_l1 > 2 else 2

        if block_loop_x * block_loop_z * thread_dims[0] * thread_dims[2] > 1:
            rhs_thread_cnt_l1 = rhs_thread_cnt_l1 if rhs_thread_cnt_l1 > 2 else 2

        # make block's lhs/rhs totally cached in L2B and L1
        rhs_max_cnt_l2 = (K + l2K - 1) // l2K
        lhs_max_cnt_l2 = rhs_max_cnt_l2
        rhs_max_cnt_l1 = (K + l1K - 1) // l1K * thread_dims[0] * thread_dims[2]
        lhs_max_cnt_l1 = (K + l1K - 1) // l1K * thread_dims[1] * thread_dims[2]

        if block_loop_y * block_loop_z > 1:
            lhs_max_cnt_l2 = lhs_max_cnt_l2 if lhs_max_cnt_l2 > 2 else 2

        if block_loop_x * block_loop_z > 1:
            rhs_max_cnt_l2 = rhs_max_cnt_l2 if rhs_max_cnt_l2 > 2 else 2

        if block_loop_y * block_loop_z * thread_dims[1] * thread_dims[2] > 1:
            lhs_max_cnt_l1 = lhs_max_cnt_l1 if lhs_max_cnt_l1 > 2 else 2

        if block_loop_x * block_loop_z * thread_dims[0] * thread_dims[2] > 1:
            rhs_max_cnt_l1 = rhs_max_cnt_l1 if rhs_max_cnt_l1 > 2 else 2

        # make input tensor's lhs/rhs totally cached in L2B and L1
        rhs_ultra_max_cnt_l2 = (K + l2K - 1) // l2K * block_loop_x
        lhs_ultra_max_cnt_l2 = (K + l2K - 1) // l2K * block_loop_y
        rhs_ultra_max_cnt_l1 = (K + l1K - 1) // l1K * thread_dims[0] * thread_dims[2] * block_loop_x
        lhs_ultra_max_cnt_l1 = (K + l1K - 1) // l1K * thread_dims[1] * thread_dims[2] * block_loop_y
        if block_loop_y * block_loop_z > 1:
            lhs_ultra_max_cnt_l2 = lhs_ultra_max_cnt_l2 if lhs_ultra_max_cnt_l2 > 2 else 2

        if block_loop_x * block_loop_z > 1:
            rhs_ultra_max_cnt_l2 = rhs_ultra_max_cnt_l2 if rhs_ultra_max_cnt_l2 > 2 else 2

        if block_loop_y * block_loop_z * thread_dims[1] * thread_dims[2] > 1:
            lhs_ultra_max_cnt_l1 = lhs_ultra_max_cnt_l1 if lhs_ultra_max_cnt_l1 > 2 else 2

        if block_loop_x * block_loop_z * thread_dims[0] * thread_dims[2] > 1:
            rhs_ultra_max_cnt_l1 = rhs_ultra_max_cnt_l1 if rhs_ultra_max_cnt_l1 > 2 else 2

        lhs_cnt_l2_vec = []
        rhs_cnt_l2_vec = []
        res_cnt_l2_vec = []
        lhs_cnt_l1_vec = []
        rhs_cnt_l1_vec = []
        res_cnt_l1_vec = []

        res_cnt_l2_vec.append(2 if block_total > 1 else 1)
        res_cnt_l1_vec.append(2 if block_total * thread_dims[0] * thread_dims[1] * thread_dims[2] > 1 else 1)

        block_row_major_vec = []
        if block_loop_y == 1:
            block_row_major_vec.append(True)
        elif block_loop_x == 1:
            block_row_major_vec.append(False)
        else:
            block_row_major_vec.append(True)
            block_row_major_vec.append(False)

        subthread_row_major_vec = []
        if thread_dims[1] == 1:
            subthread_row_major_vec.append(True)
        elif thread_dims[0] == 1:
            subthread_row_major_vec.append(False)
        else:
            subthread_row_major_vec.append(True)
            subthread_row_major_vec.append(False)

        if self.mem_arch_type == MemArchType.DSM_SHARED:
            subthread_row_major_vec.clear()
            subthread_row_major_vec.append(True)
        elif self.mem_arch_type == MemArchType.DSM_LOCAL:
            subthread_row_major_vec.clear()
            subthread_row_major_vec.append(True)

        cb = []
        for block_row_major in block_row_major_vec:
            for subthread_row_major in subthread_row_major_vec:
                lhs_cnt_l2_vec.clear()
                lhs_cnt_l2_vec.append(lhs_max_cnt_l2)
                if lhs_max_cnt_l2 > 2:
                    lhs_cnt_l2_vec.append(2)
                if not block_row_major and block_loop_x > 1:
                    lhs_cnt_l2_vec.append(lhs_ultra_max_cnt_l2)
                lhs_cnt_l2_vec = sorted(list(set(lhs_cnt_l2_vec)))

                rhs_cnt_l2_vec.clear()
                rhs_cnt_l2_vec.append(rhs_max_cnt_l2)
                if rhs_max_cnt_l2 > 2:
                    rhs_cnt_l2_vec.append(2)
                if block_row_major and block_loop_y > 1:
                    rhs_cnt_l2_vec.append(rhs_ultra_max_cnt_l2)
                rhs_cnt_l2_vec = sorted(list(set(rhs_cnt_l2_vec)))

                lhs_cnt_l1_vec.clear()
                lhs_cnt_l1_vec.append(lhs_thread_cnt_l1)
                lhs_cnt_l1_vec.append(lhs_max_cnt_l1)
                if lhs_max_cnt_l1 > 2:
                    lhs_cnt_l1_vec.append(2)
                if not block_row_major:
                    lhs_cnt_l1_vec.append(lhs_ultra_max_cnt_l1)
                lhs_cnt_l1_vec = sorted(list(set(lhs_cnt_l1_vec)))

                rhs_cnt_l1_vec.clear()
                rhs_cnt_l1_vec.append(rhs_thread_cnt_l1)
                rhs_cnt_l1_vec.append(rhs_max_cnt_l1)
                if rhs_max_cnt_l1 > 2:
                    rhs_cnt_l1_vec.append(2)
                if block_row_major:
                    rhs_cnt_l1_vec.append(rhs_ultra_max_cnt_l1)
                rhs_cnt_l1_vec = sorted(list(set(rhs_cnt_l1_vec)))

                if self.mem_arch_type == MemArchType.DSM_SHARED:
                    lhs_cnt_l2_vec = [2]
                    rhs_cnt_l2_vec = [2]
                    res_cnt_l2_vec = [2]
                    res_cnt_l1_vec = [2]
                    lhs_cnt_l1_vec = [2]
                    rhs_cnt_l1_vec = [2]
                elif self.mem_arch_type == MemArchType.DSM_LOCAL:
                    lhs_cnt_l2_vec = [2]
                    rhs_cnt_l2_vec = [2]
                    res_cnt_l2_vec = [0]
                    res_cnt_l1_vec = [0]
                    lhs_cnt_l1_vec = [2]
                    rhs_cnt_l1_vec = [2]

                for lhs_cnt_l2, rhs_cnt_l2, res_cnt_l2, lhs_cnt_l1, rhs_cnt_l1, res_cnt_l1 in itertools.product(
                    lhs_cnt_l2_vec, rhs_cnt_l2_vec, res_cnt_l2_vec, lhs_cnt_l1_vec, rhs_cnt_l1_vec, res_cnt_l1_vec
                ):
                    grid_shape = BatchGemmGridShape(
                        grid_dims=grid_dims,
                        block_dims=block_dims,
                        thread_dims=thread_dims,
                        subthread_dims=subthread_dims,
                        lhs_buf_cnt_l2=lhs_cnt_l2,
                        rhs_buf_cnt_l2=rhs_cnt_l2,
                        res_buf_cnt_l2=res_cnt_l2,
                        lhs_buf_cnt_l1=lhs_cnt_l1,
                        rhs_buf_cnt_l1=rhs_cnt_l1,
                        res_buf_cnt_l1=res_cnt_l1,
                        calc_ceil_K_l2=l2K,
                        calc_ceil_K_l1=l1K,
                    )
                    if not self.check_mem_overflow(grid_shape):
                        cb.append(
                            (
                                lhs_cnt_l2,
                                rhs_cnt_l2,
                                res_cnt_l2,
                                block_row_major,
                                lhs_cnt_l1,
                                rhs_cnt_l1,
                                res_cnt_l1,
                                subthread_row_major,
                            )
                        )
        return cb

    def check_mem_overflow(self, shape: BatchGemmGridShape):

        bpe = self.bpe
        thread_stride_x, thread_stride_y, thread_stride_z = shape.thread_dims_stride()[:3]
        lhs_bytes = bpe["lhs"] * thread_stride_y * shape.calc_ceil_K_l1 * shape.lhs_buf_cnt_l1 * thread_stride_z
        rhs_bytes = bpe["rhs"] * thread_stride_x * shape.calc_ceil_K_l1 * shape.rhs_buf_cnt_l1 * thread_stride_z
        bias_bytes = bpe["bias"] * thread_stride_x * (shape.bias_buf_cnt_l1 if self.has_bias else 0) * thread_stride_z
        res_bytes = 0

        if lhs_bytes + rhs_bytes + bias_bytes + res_bytes > self.l1_bytes_per_sip:
            return True

        block_stride_x, block_stride_y, block_stride_z = shape.block_dims_stride()[:3]
        lhs_bytes = (
            bpe["lhs"]
            * block_stride_z
            * block_stride_y
            * shape.calc_ceil_K_l2
            * shape.lhs_buf_cnt_l2
            * shape.block_dims[0]
        )

        rhs_bytes = (
            bpe["rhs"]
            * block_stride_z
            * block_stride_x
            * shape.calc_ceil_K_l2
            * shape.rhs_buf_cnt_l2
            * shape.block_dims[1]
        )
        bias_bytes = bpe["bias"] * block_stride_x * (shape.bias_buf_cnt_l2 if self.has_bias else 0) * block_stride_z

        res_bytes = (
            bpe["res"] * block_stride_z * block_stride_x * block_stride_y * shape.res_buf_cnt_l2
            if self.mem_arch_type == MemArchType.DSM_SHARED
            else 0
        )
        if lhs_bytes + rhs_bytes + bias_bytes + res_bytes > self.l2_bytes_per_sic:
            return True
        return False

    def calc_mem_access_l1_per_sip(self, shape: BatchGemmGridShape):
        bpe = self.bpe

        thread_stride_x, thread_stride_y, thread_stride_z = shape.thread_dims_stride()[:3]
        lhs_bytes = bpe["lhs"] * thread_stride_y * shape.calc_ceil_K_l1 * shape.lhs_buf_cnt_l1 * thread_stride_z
        rhs_bytes = bpe["rhs"] * thread_stride_x * shape.calc_ceil_K_l1 * shape.rhs_buf_cnt_l1 * thread_stride_z
        bias_bytes = bpe["bias"] * thread_stride_x * (shape.bias_buf_cnt_l1 if self.has_bias else 0) * thread_stride_z
        res_bytes = 0
        return lhs_bytes + rhs_bytes + bias_bytes + res_bytes

    def calc_mem_access_l2_per_sip(self, shape: BatchGemmGridShape):
        bpe = self.bpe

        thread_stride_x, thread_stride_y, thread_stride_z = shape.thread_dims_stride()[:3]
        lhs_bytes = bpe["lhs"] * thread_stride_y * shape.calc_ceil_K_l2 * shape.lhs_buf_cnt_l2 * thread_stride_z
        rhs_bytes = bpe["rhs"] * thread_stride_x * shape.calc_ceil_K_l2 * shape.rhs_buf_cnt_l2 * thread_stride_z
        bias_bytes = bpe["bias"] * thread_stride_x * (shape.bias_buf_cnt_l2 if self.has_bias else 0) * thread_stride_z
        res_bytes = 0
        return lhs_bytes + rhs_bytes + bias_bytes + res_bytes

    def calc_l1_bw_required(self, shape: BatchGemmGridShape):
        bpe = self.bpe
        thread_stride_x, thread_stride_y, thread_stride_z = shape.thread_dims_stride()[:3]
        throughput = self.config.compute.thread_2d_throughput[self.compute_dtype["2d"]]
        l1_bw_required = (
            (bpe["lhs"] * thread_stride_y + bpe["rhs"] * thread_stride_x)
            / (thread_stride_x * thread_stride_y)
            * throughput
        )
        return l1_bw_required

    def calc_mem_access_total_l3(self, shape: BatchGemmGridShape):
        B = self.in_batch
        M = self.in_m
        N = self.in_n
        K = self.in_k
        bpe = self.bpe
        grid_stride_x, grid_stride_y, grid_stride_z = shape.grid_dims_stride()[:3]
        # loop of each sic
        block_loop_x = (N + grid_stride_x - 1) // grid_stride_x
        block_loop_y = (M + grid_stride_y - 1) // grid_stride_y
        block_loop_z = (B + grid_stride_z - 1) // grid_stride_z

        bytes_lhs = bpe["lhs"] * block_loop_x * M * K * shape.grid_dims[0]
        bytes_rhs = bpe["rhs"] * block_loop_y * N * K * shape.grid_dims[1]
        bytes_res = bpe["res"] * M * N

        if shape.block_traverse_dim_order[0] == 0:
            if shape.lhs_buf_cnt_l2 * shape.calc_ceil_K_l2 >= K:
                # block lhs can totally cached in l2 buf
                bytes_lhs = bpe["lhs"] * M * K * shape.grid_dims[0]
            if shape.rhs_buf_cnt_l2 // block_loop_x * shape.calc_ceil_K_l2 >= K:
                # all block rhs in row can totally cached in l2 buf
                bytes_rhs = bpe["rhs"] * N * K * shape.grid_dims[1]
        else:
            if shape.rhs_buf_cnt_l2 * shape.calc_ceil_K_l2 >= K:
                # block rhs can totally cached in l2 buf
                bytes_rhs = bpe["rhs"] * N * K * shape.grid_dims[1]
            if shape.lhs_buf_cnt_l2 // block_loop_y * shape.calc_ceil_K_l2 >= K:
                # all grid lhs in col can totally cached in l2 buf
                bytes_lhs = bpe["lhs"] * M * K * shape.grid_dims[0]

        k_loop_cnt = (K + shape.calc_ceil_K_l2 - 1) // shape.calc_ceil_K_l2
        lhs_rd_cnt = (
            k_loop_cnt
            + (k_loop_cnt - shape.lhs_buf_cnt_l2 if k_loop_cnt > shape.lhs_buf_cnt_l2 else 0) * (block_loop_x - 1)
        ) / k_loop_cnt
        rhs_rd_cnt = (
            k_loop_cnt
            + (k_loop_cnt - shape.rhs_buf_cnt_l2 if k_loop_cnt > shape.rhs_buf_cnt_l2 else 0) * (block_loop_y - 1)
        ) / k_loop_cnt

        bytes_lhs = bpe["lhs"] * M * K * shape.grid_dims[0] * lhs_rd_cnt
        bytes_rhs = bpe["rhs"] * N * K * shape.grid_dims[1] * rhs_rd_cnt
        bytes_res = bpe["res"] * M * N

        bytes_lhs *= shape.block_dims[0]
        bytes_rhs *= shape.block_dims[1]
        bytes_total = bytes_lhs + bytes_rhs + bytes_res
        # bytes_total *= B;
        bytes_total *= grid_stride_z * block_loop_z

        return bytes_total

    def calc_flops_total(self):
        return self.in_batch * self.in_m * self.in_n * self.in_k * 2.0

    def calc_attainable_flops(self, shape: BatchGemmGridShape):
        shape.mem_access_total_l3 = self.calc_mem_access_total_l3(shape)
        xpu_used = shape.block_cnt_in_grid() * shape.thread_cnt_in_block()
        flops_used = self.xpu_FLOPS * xpu_used
        Imax = flops_used / self.l3_bandwidth
        flops_total = self.calc_flops_total()
        I = flops_total / shape.mem_access_total_l3
        shape.imax_l3 = I
        attaible_flops = self.l3_bandwidth * I if I < Imax else flops_used
        shape.attaible_flops = attaible_flops
        shape.mem_access_l1_per_sip = self.calc_mem_access_l1_per_sip(shape)
        shape.mem_access_l2_per_sip = self.calc_mem_access_l2_per_sip(shape)
        shape.l1_bw_required = self.calc_l1_bw_required(shape)
        return attaible_flops

    def split(self):
        self.shape_list = []
        # 这里根据instance(sip或sic数量)生成所有可能的layout(b, m, n, k=1)组合
        def get_layout(instances):
            layout = []
            for B in range(1, instances + 1):
                for M in range(1, instances + 1):
                    if B * M > instances:
                        continue
                    for N in range(1, instances + 1):
                        if B * M * N == instances:
                            layout.append([N, M, B, 1])
            return layout
        # 计算好k的取值候选
        def get_calc_ceil_k(in_k):
            calc_ceil_K_align = 64
            # 向上对其到64的整数倍，64因为硬件适配良好
            k_align = (in_k + calc_ceil_K_align - 1) // calc_ceil_K_align * calc_ceil_K_align
            kVec = list(range(256, 2048 + 1, calc_ceil_K_align))
            if k_align not in kVec:
                kVec.append(k_align)
            kVec = [k for k in kVec if k <= k_align]
            calc_ceil_k = [k for k in kVec if k % calc_ceil_K_align == 0]
            return calc_ceil_k

        def filter_this_shape(grid_dims, block_dims, thread_dims, subthread_dims, maxCeilCnt, l1K):
            grid_x, grid_y, grid_z = grid_dims[0:3]
            block_x, block_y, block_z = block_dims[0:3]
            thread_x, thread_y, thread_z = thread_dims[0:3]
            subthread_x, subthread_y, subthread_z = subthread_dims[0:3]
            # no block-expand when k_tile less than K
            if self.in_k > l1K and maxCeilCnt < thread_dims[0] * thread_dims[1] * thread_dims[2]:
                return True
            if grid_z > self.in_batch:  # sic cnt overflow in batch dim
                return True

            # ceil cnt overflow
            if (
                (self.in_n + grid_x * block_x * subthread_x - 1) // (grid_x * block_x * subthread_x) < thread_x
                or (self.in_m + grid_y * block_y * subthread_y - 1) // (grid_y * block_y * subthread_y) < thread_y
                or (self.in_batch + grid_z * block_z * subthread_z - 1) // (grid_z * block_z * subthread_z) < thread_z
            ):
                return True

            # sic cnt overflow in 1 dim, meanwhile shortage in the other dim
            outputDimStride = [self.in_n, self.in_m, self.in_batch]
            sic_overflow = []
            sic_shortage = []
            for dim in range(3):
                sic_overflow.append(
                    outputDimStride[dim] < subthread_dims[dim] * thread_dims[dim] * block_dims[dim] * grid_dims[dim]
                    and subthread_dims[dim] * thread_dims[dim] * block_dims[dim] * grid_dims[dim] - outputDimStride[dim]
                    >= thread_dims[dim] * subthread_dims[dim] * grid_dims[dim]
                )
                sic_shortage.append(
                    outputDimStride[dim] > subthread_dims[dim] * thread_dims[dim] * block_dims[dim] * grid_dims[dim]
                )

            if any(sic_overflow) and any(sic_shortage):
                return True

            return False

        sicLayout = get_layout(self.sic_cnt)
        sipLayout = get_layout(self.sip_cnt)
        calc_ceil_k = get_calc_ceil_k(self.in_k)
        imax = 0
        shape_list: List[Tuple[int, BatchGemmGridShape]] = []
        shape_idx = 0
        # 枚举出来所有满足上限的NBM组合 作为后续的grid/block的形状
        for ceil_info in self.calc_ceil_info:
            # 设置block的layout组合
            ceilLayout = []
            # max_ceil_cnt 是 block上限， TODO: 为什么这么设置？
            max_ceil_cnt = ceil_info[1]
            for B in range(1, 2):
                for M in range(1, max_ceil_cnt + 1):
                    if B * M > max_ceil_cnt:
                        continue
                    for N in range(1, max_ceil_cnt + 1):
                        if B * M * N <= max_ceil_cnt:
                            ceilLayout.append([N, M, B, 1])

            # combine all possible sic&xpu&ceil layout
            #ceil_layout 是 thread的layout组合 B/M/N/k 方向的线程数量
            for ceil_layout in ceilLayout: #thread dim
                # sip_layout 是block的layout组合 代表在不同方向上的block数量
                for sip_layout in sipLayout: #block dim
                    # sic_layout 是 grid的layout组合 在cluster上铺设块的数量
                    for sic_layout in sicLayout: #grid dim
                        for K in calc_ceil_k:
                            if filter_this_shape(sic_layout, sip_layout, ceil_layout, ceil_info[0], ceil_info[1], K):
                                continue
                            cb_list = self.gen_buf_cnt_combines(sic_layout, sip_layout, ceil_layout, ceil_info[0], K, K)
                            for cb in cb_list:
                                grid_shape = BatchGemmGridShape(
                                    grid_dims=sic_layout,
                                    block_dims=sip_layout,
                                    thread_dims=ceil_layout,
                                    subthread_dims=ceil_info[0],
                                    lhs_buf_cnt_l2=cb[0],
                                    rhs_buf_cnt_l2=cb[1],
                                    res_buf_cnt_l2=cb[2],
                                    block_row_major=cb[3],
                                    lhs_buf_cnt_l1=cb[4],
                                    rhs_buf_cnt_l1=cb[5],
                                    res_buf_cnt_l1=cb[6],
                                    subthread_row_major=cb[7],
                                    calc_ceil_K_l2=K,
                                    calc_ceil_K_l1=K,
                                )
                                self.calc_attainable_flops(grid_shape)
                                shape_list.append((shape_idx, grid_shape))
                                shape_idx += 1
                                imax = max(imax, grid_shape.imax_l3)

        sorted_shape_list = self._sort_shape_candidates(shape_list)
        # pprint(shape_list[0])
        # SYSTEM L3 BW is
        total_flops = self.xpu_FLOPS * self.sic_cnt * self.sip_cnt

        msg = f"SYSTEM L3 BW is {self.l3_bandwidth/ 1e9 :.3f} GB/S, {total_flops / 1e9:.3f} GFLOPS, IMAX is {total_flops/self.l3_bandwidth:.3f} flops/byte \n"
        msg += f"Generated {len(sorted_shape_list)} tiling candidates:\n"

        def tiled_shape(shape, k):
            grid_stride = shape.grid_dims_stride()
            block_stride = shape.block_dims_stride()
            thread_stride = shape.thread_dims_stride()
            return [
                (grid_stride[2], grid_stride[1], grid_stride[0], k),
                (block_stride[2], block_stride[1], block_stride[0], k),
                (thread_stride[2], thread_stride[1], thread_stride[0], k),
            ]

        for idx, shape in sorted_shape_list[:100]:
            tiled = tiled_shape(shape, shape.calc_ceil_K_l2)
            msg += (
                f"Tile Dims: {shape.grid_dims} {shape.block_dims} {shape.thread_dims} {shape.subthread_dims}, "
                f"Tiled Shapes: {tiled[0]} {tiled[1]} {tiled[2]} , "
                f"[{'row' if shape.block_traverse_dim_order[0] == 0 else 'col'}, {'row' if shape.subthread_traverse_dim_order[0] == 0 else 'col'}] "
                f"Cost: {shape.sol_cost * 1e9:.2f} ns, imax: {shape.imax_l3:.3f}, {shape.attaible_flops/1e9:.3f} GFLOPS, L3: {shape.mem_access_total_l3/ 1024 / 1024:.3f} MB, "
                f"L2: {shape.mem_access_l2_per_sip / 1024 / 1024:.3f} MB "
                f"L1 bw required: {shape.l1_bw_required:.2f} bytes/cycle \n"
            )
        self.shape_list = sorted_shape_list
        logger.info(msg)

    def _calc_sol_cost(self, batch_gemm_shape: BatchGemmGridShape):
        raise NotImplementedError()

    def _gen_workloads(self, batch_gemm_shape: BatchGemmGridShape, first_sip_only=False):
        raise NotImplementedError()

    def _sort_shape_candidates(self, shape_list: List[Tuple[int, BatchGemmGridShape]]):
        num = 1000
        prioritized_shape_list_1 = sorted(shape_list, key=lambda x: x[1].l1_bw_required)[:num]
        prioritized_shape_list_2 = sorted(shape_list, key=lambda x: -x[1].attaible_flops)[:num]
        prioritized_shape_list = []
        seen_idxs = set()
        for idx, shape in prioritized_shape_list_1 + prioritized_shape_list_2:
            if idx not in seen_idxs:
                prioritized_shape_list.append((idx, shape))
                seen_idxs.add(idx)
        for idx, shape in prioritized_shape_list:
            shape.sol_cost = self._calc_sol_cost(shape)
        sorted_shape_list = sorted(prioritized_shape_list, key=lambda x: x[1].sol_cost)
        return sorted_shape_list

    def impl(self):
        best_shape = self.get_best_shape()
        # best_shape = BatchGemmGridShape(
        #     grid_dims=[1, 4, 1, 1],
        #     block_dims=[6, 1, 1, 1],
        #     thread_dims=[4, 2, 1, 1],
        #     subthread_dims=[64, 64, 1, 1],
        #     lhs_buf_cnt_l2=2,
        #     rhs_buf_cnt_l2=2,
        #     res_buf_cnt_l2=0,
        #     block_row_major=True,
        #     lhs_buf_cnt_l1=2,
        #     rhs_buf_cnt_l1=2,
        #     res_buf_cnt_l1=0,
        #     subthread_row_major=False,
        #     calc_ceil_K_l1=256,
        #     calc_ceil_K_l2=256,
        # )
        self.tiled_workloads = self._gen_workloads(best_shape)

    def get_best_shape(self) -> BatchGemmGridShape:
        return self.shape_list[0][1]

    def get_tiled_workloads(self):
        return self.tiled_workloads
