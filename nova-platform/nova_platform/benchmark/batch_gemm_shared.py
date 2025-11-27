from dataclasses import dataclass, field
from typing import Generator, List, Tuple, Dict, Union
from collections import deque

from nova_platform.base_model import (
    AddrDomain,
    DataflowActionComputeStat,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
)
from nova_platform.config import BossaNovaConfig
from nova_platform.benchmark.op_base import Workload, OpBase, Operand, GridShape, list_product, DType
from nova_platform.benchmark.batch_gemm import (
    BatchGemmBase,
    MemArchType,
    BatchGemmGridShape,
    QuantType,
    bpe_for_quant,
    compute_dtype_for_quant,
)
from nova_platform.executor.nova_platform_barrier import BarrierManager
from nova_platform.benchmark.utils import _iter_access_gen
from functools import lru_cache


bm = BarrierManager()


def _get_minimum_iter_k(b: int, m: int, n: int, bpe: Dict[str, float], mac_dtype: DType, config: BossaNovaConfig):
    iter_k = 32
    if mac_dtype == DType.FP4:
        iter_k = 64
    k = 0
    l1_bw = config.bw.l0.shared.bw
    l1_latency = config.bw.l0.shared.pre_latency
    shared_l1_cycle = 0
    while shared_l1_cycle < l1_latency:
        k += iter_k
        data_size = b * m * k * bpe["lhs"] + b * n * k * bpe["rhs"]
        shared_l1_cycle = data_size / l1_bw
    return k


def _get_sip_workloads(sic_workload: Workload, batch_gemm_shape: BatchGemmGridShape, config, sip_idx):
    sip_workloads = []
    B_l2 = sic_workload.attr["b"]
    M_l2 = sic_workload.attr["m"]
    N_l2 = sic_workload.attr["n"]
    K_l2 = sic_workload.attr["k"]
    quant_type = sic_workload.attr.get("quant_type", QuantType.No_Quant)
    compute_dtype = compute_dtype_for_quant(sic_workload.dtype, quant_type)
    bpe = bpe_for_quant(sic_workload.dtype, quant_type)
    thread_loop_x, thread_loop_y, thread_loop_z = batch_gemm_shape.block_dims[:3]
    shape_thread = batch_gemm_shape.thread_dims_stride()
    sip_idx_vec = batch_gemm_shape.thread_traverse_idx[sip_idx]

    sip_idx_x, sip_idx_y, sip_idx_z = sip_idx_vec[:3]
    m_offset = shape_thread[1] * (sip_idx_y)
    n_offset = shape_thread[0] * (sip_idx_x)
    b_offset = shape_thread[2] * (sip_idx_z)

    if m_offset >= M_l2 or n_offset >= N_l2 or b_offset >= B_l2:
        return []
    m_valid = shape_thread[1] if M_l2 >= m_offset + shape_thread[1] else M_l2 - m_offset
    n_valid = shape_thread[0] if N_l2 >= n_offset + shape_thread[0] else N_l2 - n_offset
    b_valid = shape_thread[2] if B_l2 >= b_offset + shape_thread[2] else B_l2 - b_offset
    iter_k = _get_minimum_iter_k(b_valid, m_valid, n_valid, bpe, compute_dtype["2d"], config)
    iter_k = min(K_l2, iter_k)
    loop_k = (K_l2 + iter_k - 1) // iter_k
    for i in range(loop_k):
        k_offset = i * iter_k
        if k_offset >= K_l2:
            break
        k_valid = iter_k if K_l2 >= k_offset + iter_k else K_l2 - k_offset
        inputs = [
            Operand(
                dim=(b_valid, m_valid, k_valid),
                addr=0x4000000000,
                bpe=bpe["lhs"],
                dim_offset=(b_offset, m_offset, k_offset),
                dim_stride=(B_l2, M_l2, K_l2),
            ),
            Operand(
                dim=(b_valid, k_valid, n_valid),
                addr=0x4100000000,
                bpe=bpe["rhs"],
                dim_offset=(b_offset, k_offset, n_offset),
                dim_stride=(B_l2, K_l2, N_l2),
            ),
        ]
        if quant_type == QuantType.Wf4g_Af8t and sic_workload.inputs[2]:
            quant_group_size = sic_workload.attr.get("quant_group_size", 32)
            scaling_tensor = sic_workload.inputs[2]
            quant_group_num = scaling_tensor.dim[1]
            group_num_offset = (k_offset + quant_group_size - 1) // quant_group_size
            valid_group_num = (k_valid + quant_group_size - 1) // quant_group_size
            valid_group_num = (
                valid_group_num
                if quant_group_num >= group_num_offset + valid_group_num
                else quant_group_num - group_num_offset
            )
            inputs.append(
                Operand(
                    dim=(b_valid, valid_group_num, n_valid),
                    addr=0x4200000000,
                    bpe=bpe["scaling"],
                    dim_offset=(b_offset, group_num_offset, n_offset),
                    dim_stride=(B_l2, quant_group_num, N_l2),
                )
            )
        outputs = []
        if i == loop_k - 1 and len(sic_workload.outputs) > 0:
            outputs = [
                Operand(
                    dim=(b_valid, m_valid, n_valid),
                    addr=0x4300000000,
                    bpe=bpe["res"],
                    dim_offset=(b_offset, m_offset, n_offset),
                    dim_stride=(B_l2, M_l2, N_l2),
                ),
            ]
        attr = {**sic_workload.attr, "b": b_valid, "m": m_valid, "n": n_valid, "k": k_valid}
        sip_workloads.append(Workload(inputs, outputs, dtype=sic_workload.dtype, attr=attr))
    return sip_workloads


class BatchGemmShared(BatchGemmBase):

    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        super().__init__(config, workload)
        self.mem_arch_type = MemArchType.DSM_SHARED
        self.calc_ceil_K_align = 64
        self.l1_bandwidth_per_sip = config.bw.l0.shared.bw * config.freq.CORE * 1e9

    def _gen_workloads(self, batch_gemm_shape: BatchGemmGridShape, first_sip_only=False):
        sic_workloads = {}
        batch_gemm_shape.gen_traverse_idx([self.in_n, self.in_m, self.in_batch, 1])

        xpu_layout = batch_gemm_shape.block_dims
        shape_grid = batch_gemm_shape.grid_dims_stride()
        shape_block = batch_gemm_shape.block_dims_stride()
        shape_thread = batch_gemm_shape.thread_dims_stride()

        B_l3 = self.in_batch
        M_l3 = self.in_m
        N_l3 = self.in_n
        K_l3 = self.in_k
        B_l2 = shape_block[2]
        M_l2 = shape_block[1]
        N_l2 = shape_block[0]
        K_l2 = batch_gemm_shape.calc_ceil_K_l2

        shape_lhs_l2 = [K_l2, M_l2, B_l2, 1]  # shape after splitting
        shape_rhs_l2 = [N_l2, K_l2, B_l2, 1]

        lhs_cnt_l2 = batch_gemm_shape.lhs_buf_cnt_l2
        rhs_cnt_l2 = batch_gemm_shape.rhs_buf_cnt_l2
        lhs_tensor_l3 = self.workload.inputs[0]
        rhs_tensor_l3 = self.workload.inputs[1]
        res_tensor_l3 = self.workload.outputs[0]

        block_loop_x = (N_l3 + shape_grid[0] - 1) // shape_grid[0]
        block_loop_y = (M_l3 + shape_grid[1] - 1) // shape_grid[1]
        block_loop_z = (B_l3 + shape_grid[2] - 1) // shape_grid[2]
        block_total = block_loop_x * block_loop_y * block_loop_z
        assert block_total == len(batch_gemm_shape.block_traverse_idx[0])
        thread_loop_x, thread_loop_y, thread_loop_z = batch_gemm_shape.block_dims[:3]
        k_loop_l2 = (K_l3 + K_l2 - 1) // K_l2

        sic_total = batch_gemm_shape.block_cnt_in_grid()
        sip_total = batch_gemm_shape.thread_cnt_in_block()

        for sic_idx in range(sic_total):
            # alloc l2 mem
            pass

        for sic_idx in range(sic_total):
            if first_sip_only and sic_idx != 0:
                break
            sic_workloads.setdefault(sic_idx, [])
            for block_idx in range(block_total):
                block_idx_vec = batch_gemm_shape.block_traverse_idx[sic_idx][block_idx]
                if block_idx_vec[0] < 0:
                    continue
                block_idx_x, block_idx_y, block_idx_z = block_idx_vec[:3]
                for k_l2_idx in range(k_loop_l2):
                    k_l2_idx_new = k_l2_idx
                    if block_idx & 1:
                        k_l2_idx_new = k_loop_l2 - 1 - k_l2_idx  # snake-like

                    k_l2_offset = K_l2 * k_l2_idx_new
                    k_l2_size = (K_l3 - k_l2_offset) if (k_l2_idx_new == k_loop_l2 - 1) else K_l2
                    # we need rhs slice with auto padding 0
                    k_l2_size_align = (
                        (k_l2_size + self.calc_ceil_K_align - 1) / self.calc_ceil_K_align * self.calc_ceil_K_align
                    )
                    b_offset = B_l2 * block_idx_z
                    m_offset = M_l2 * block_idx_y
                    n_offset = N_l2 * block_idx_x
                    k_offset = k_l2_offset
                    if b_offset >= B_l3 or m_offset >= M_l3 or n_offset >= N_l3 or k_offset >= K_l3:
                        continue
                    b_valid = B_l2 if B_l3 >= b_offset + B_l2 else B_l3 - b_offset
                    m_valid = M_l2 if M_l3 >= m_offset + M_l2 else M_l3 - m_offset
                    n_valid = N_l2 if N_l3 >= n_offset + N_l2 else N_l3 - n_offset
                    k_valid = K_l2 if K_l3 >= k_offset + K_l2 else (K_l3 - k_offset)

                    inputs = [
                        Operand(
                            dim=(b_valid, m_valid, k_valid),
                            addr=lhs_tensor_l3.addr,
                            bpe=lhs_tensor_l3.bpe,
                            dim_offset=(b_offset, m_offset, k_offset),
                            dim_stride=lhs_tensor_l3.dim_stride,
                        ),
                        Operand(
                            dim=(b_valid, k_valid, n_valid),
                            addr=rhs_tensor_l3.addr,
                            bpe=rhs_tensor_l3.bpe,
                            dim_offset=(b_offset, k_offset, n_offset),
                            dim_stride=rhs_tensor_l3.dim_stride,
                        ),
                    ]
                    if self.quant_type == QuantType.Wf4g_Af8t:
                        quant_group_size = self.workload.attr.get("quant_group_size", 32)
                        scaling_tensor = self.workload.inputs[2]
                        quant_group_num = scaling_tensor.dim[1]
                        group_num_offset = (k_offset + quant_group_size - 1) // quant_group_size
                        valid_group_num = (k_valid + quant_group_size - 1) // quant_group_size
                        valid_group_num = (
                            valid_group_num
                            if quant_group_num >= group_num_offset + valid_group_num
                            else quant_group_num - group_num_offset
                        )
                        inputs.append(
                            Operand(
                                dim=(b_valid, valid_group_num, n_valid),
                                addr=scaling_tensor.addr,
                                bpe=scaling_tensor.bpe,
                                dim_offset=(b_offset, group_num_offset, n_offset),
                                dim_stride=scaling_tensor.dim_stride,
                            )
                        )
                    if k_l2_idx == k_loop_l2 - 1:
                        # res store
                        outputs = [
                            Operand(
                                dim=(b_valid, m_valid, n_valid),
                                addr=res_tensor_l3.addr,
                                bpe=res_tensor_l3.bpe,
                                dim_offset=(b_offset, m_offset, n_offset),
                                dim_stride=res_tensor_l3.dim_stride,
                            )
                        ]
                    else:
                        outputs = []
                    attr = {**self.workload.attr, "b": b_valid, "m": m_valid, "n": n_valid, "k": k_valid}
                    sic_workloads[sic_idx].append(Workload(inputs, outputs, dtype=self.dtype, attr=attr))
        self.handle_pingpong_buf(batch_gemm_shape, sic_workloads)
        return sic_workloads

    def handle_pingpong_buf(self, batch_gemm_shape: BatchGemmGridShape, workloads: Dict[int, Workload]):
        for sic_id in workloads:
            lhs_buffer = deque(maxlen=batch_gemm_shape.lhs_buf_cnt_l2)
            rhs_buffer = deque(maxlen=batch_gemm_shape.rhs_buf_cnt_l2)
            scaling_buffer = deque(maxlen=batch_gemm_shape.rhs_buf_cnt_l2)
            for wl in workloads[sic_id]:
                lhs_operand = wl.inputs[0]
                rhs_operand = wl.inputs[1]
                if lhs_operand in lhs_buffer:
                    wl.inputs[0] = None
                else:
                    lhs_buffer.append(lhs_operand)

                if rhs_operand in rhs_buffer:
                    wl.inputs[1] = None
                else:
                    rhs_buffer.append(rhs_operand)
                if len(wl.inputs) == 3:
                    scaling_operand = wl.inputs[2]
                    if scaling_operand in scaling_buffer:
                        wl.inputs[2] = None
                    else:
                        scaling_buffer.append(scaling_operand)

    def _calc_sol_cost(self, batch_gemm_shape: BatchGemmGridShape) -> float:
        def _cal_input_latency(wl):
            size = 0
            for i in wl.inputs:
                size += list_product(tuple(i.dim)) * i.bpe if i else 0
            size = size / self.sip_cnt
            if size == 0:
                return 0
            return size / self.l3_bandwidth_per_sip + self.l3_to_l1_latency

        def _cal_output_latency(wl):
            size = 0
            for i in wl.outputs:
                size += list_product(tuple(i.dim)) * i.bpe if i else 0

            size = size / self.sip_cnt
            if size == 0:
                return 0
            return size / self.l3_bandwidth_per_sip

        def _cal_compute_latency(sip_workloads: List[Workload]):
            latency = 0
            for workload in sip_workloads:
                lhs = workload.inputs[0]
                rhs = workload.inputs[1]
                ops = 2 * workload.attr["b"] * workload.attr["m"] * workload.attr["n"] * workload.attr["k"]
                l1_access = lhs.dim[0] * lhs.dim[1] * lhs.dim[2] * lhs.bpe
                l1_access += rhs.dim[0] * rhs.dim[1] * rhs.dim[2] * rhs.bpe
                latency += max(ops / self.xpu_FLOPS, l1_access / self.l1_bandwidth_per_sip)
            return latency + self.config.bw.l0.shared.pre_latency / self.config.freq.CORE / 1e9

        sic_workloads: Dict[int, List[Workload]] = self._gen_workloads(batch_gemm_shape)
        if len(sic_workloads) == 0:
            return 0
        total_latency = 0
        cdte_ref = 0
        compute_ref = 0
        ping_buffer_release_ref = 0
        out_ref = 0
        for workload in sic_workloads[0]:
            input_latency = _cal_input_latency(workload)
            output_latency = _cal_output_latency(workload)
            compute_latency = _cal_compute_latency(_get_sip_workloads(workload, batch_gemm_shape, self.config, 0))
            compute_ref = cdte_ref + input_latency
            cdte_ref = max(cdte_ref + input_latency - self.l3_to_l1_latency, ping_buffer_release_ref)
            ping_buffer_release_ref = compute_ref + compute_latency
            out_ref = compute_ref + compute_latency
            total_latency = out_ref + output_latency
        return total_latency


@lru_cache()
def tile_shared_gemm_workload(config: BossaNovaConfig, chip_workload: Workload) -> Tuple[Workload, BatchGemmGridShape]:
    batch_gemm = BatchGemmShared(config, chip_workload)
    batch_gemm.split()
    batch_gemm.impl()
    best_shape = batch_gemm.get_best_shape()
    return batch_gemm.get_tiled_workloads(), best_shape


def dsm_shared_gemm_kernel(
    sic_workloads: Dict[int, List[Workload]],
    batch_gemm_shape: BatchGemmGridShape,
    config: BossaNovaConfig,
    kernel_id: int,
    sic_id: int,
    sip_id: int,
    relative_ts: int,
    abs_ref: int,
):

    def _iter_tensor_addr(
        tensors: Union[List[Operand], Operand], rw: str
    ) -> Generator[DataflowActionMemoryAccess, None, None]:
        tensors = tensors if isinstance(tensors, (tuple, list)) else [tensors]
        for tensor in tensors:
            mem_access = tensor.get_contiguous_mem_accesses()
            for addr, size in mem_access:
                yield DataflowActionMemoryAccess(addr, size, rw)

    def _tensor_transport_memory_stat(
        tensors: Union[List[Operand], Operand], master, src, dst, rw, relative_ts, stat_name
    ):
        if not tensors:
            return 0, 0
        tensors = tensors if isinstance(tensors, (tuple, list)) else [tensors]
        tensor_gen = _iter_access_gen(list(_iter_tensor_addr(tensors, rw)))
        next(tensor_gen)
        data_size = 0
        for tensor in tensors:
            data_size += list_product(tuple(tensor.dim)) * tensor.bpe
        tensor_read = DataflowActionMemoryStat(
            total_count=int(data_size),
            master=master,
            src=src,
            dst=dst,
            rw=rw,
            relative_ts=relative_ts,
            memory_access_list=tensor_gen.send(data_size),
            name=stat_name,
        )
        yield tensor_read
        return tensor_read.latency, tensor_read.leading_latency

    def _cdte_l3_to_shared(tensor, cdte_name, relative_ts):
        return _tensor_transport_memory_stat(
            tensors=tensor,
            master=DataflowActionType.CDTE,
            src=AddrDomain.SHARED,
            dst=AddrDomain.L3,
            rw="r",
            relative_ts=relative_ts,
            stat_name=cdte_name,
        )

    def _cdte_shared_to_l3(tensor, cdte_name, relative_ts):
        return _tensor_transport_memory_stat(
            tensors=tensor,
            master=DataflowActionType.CDTE,
            src=AddrDomain.SHARED,
            dst=AddrDomain.L3,
            rw="w",
            relative_ts=relative_ts,
            stat_name=cdte_name,
        )

    def _ld_shared_to_l0(tensors: Union[List[Operand], Operand], st_name: str, relative_ts: int):
        return _tensor_transport_memory_stat(
            tensors=tensors,
            master=DataflowActionType.XPU,
            src=AddrDomain.L0,
            dst=AddrDomain.SHARED,
            rw="r",
            relative_ts=relative_ts,
            stat_name=st_name,
        )

    def _st_l0_to_shared(tensor, st_name, relative_ts):
        return _tensor_transport_memory_stat(
            tensors=tensor,
            master=DataflowActionType.XPU,
            src=AddrDomain.L0,
            dst=AddrDomain.SHARED,
            rw="w",
            relative_ts=relative_ts,
            stat_name=st_name,
        )

    def _split_operands(input: Operand, total_sip: int, sip_id: int):
        if not input:
            return None
        for idx, dim in enumerate(input.dim):
            if dim < total_sip:
                continue
            split_size = (dim + total_sip - 1) // total_sip
            offset = split_size * sip_id
            if offset >= dim:
                return None
            new_dim = split_size if dim >= split_size + offset else dim - offset
            new_tensor_dim = tuple([new_dim if idx == i else d for i, d in enumerate(input.dim)])
            new_tensor_offset = tuple([d + offset if idx == i else d for i, d in enumerate(input.dim_offset)])
            return Operand(new_tensor_dim, input.addr, input.bpe, new_tensor_offset, input.dim_stride)
        return input

    def _gemm_l2_kernel(sip_workloads: List[Workload], relative_ts, idx):
        total_latency = relative_ts
        stat_ref = relative_ts
        subthread_num = batch_gemm_shape.subthread_cnt_in_thread()
        for workload in sip_workloads:
            input_tensors = []
            input_names = []
            if workload.inputs[0]:
                input_tensors.append(workload.inputs[0])
                input_names.append("lhs")
            if workload.inputs[1]:
                input_tensors.append(workload.inputs[1])
                input_names.append("rhs")

            quant_type = workload.attr.get("quant_type", QuantType.No_Quant)
            mac_dtype = workload.dtype
            if quant_type == QuantType.Wf4g_Af8t:
                mac_dtype = DType.FP4
                if len(workload.inputs) == 3:
                    input_tensors.append(workload.inputs[2])
                    input_names.append("scaling")
            elif quant_type == QuantType.Wf8t_Af8t:
                mac_dtype = DType.FP8
            input_latency, input_leading = yield from _ld_shared_to_l0(
                input_tensors, f"{'+'.join(input_names)}:shared->L0", stat_ref
            )

            compute_ref = stat_ref + input_leading
            scalar_ops = 4 * subthread_num
            compute_stat1 = DataflowActionComputeStat(
                name=f"{idx} scalar",
                compute_scalar_cycle=scalar_ops,
                relative_ts=compute_ref,
            )
            yield compute_stat1
            in_b = workload.attr["b"]
            in_m = workload.attr["m"]
            in_n = workload.attr["n"]
            in_k = workload.attr["k"]
            compute_mac = in_b * in_m * in_n * in_k
            compute_stat2 = DataflowActionComputeStat(
                name=f"{idx} mac",
                compute_2d_ops={mac_dtype: compute_mac * 2},
                relative_ts=compute_ref,
            )
            yield compute_stat2
            compute_latency = compute_stat2.latency
            VMM = 10
            compute_vmm = DataflowActionComputeStat(
                name=f"{idx} vmm",
                compute_nop_cycle=VMM,
                relative_ts=compute_ref + compute_latency,
            )
            yield compute_vmm
            compute_latency += compute_vmm.latency

            res_latency = 0
            res_ref = compute_ref
            if len(workload.outputs) > 0:
                if quant_type == QuantType.Wf4g_Af8t or quant_type == QuantType.Wf8t_Af8t:
                    compute_convert = DataflowActionComputeStat(
                        name=f"{idx} convert",
                        compute_1d_ops={DType.FP32: in_m * in_n},
                        relative_ts=compute_ref,
                    )
                    yield compute_convert
                    compute_latency = max(compute_convert.latency, compute_latency)
                res_tensor = workload.outputs[0]
                res_latency, res_leading = yield from _st_l0_to_shared(res_tensor, f"{idx}:res:l0->shared", res_ref)
            total_latency = max(
                compute_ref + compute_latency,
                stat_ref + input_latency,
                res_ref + res_latency,
            )
            stat_ref = max(stat_ref, total_latency - input_leading, compute_ref)

        return total_latency

    total_latency = relative_ts
    cdte_ref = relative_ts
    compute_release_ref = relative_ts
    ping_buffer_release_ref = relative_ts
    if sic_id not in sic_workloads:
        return 0

    sip_total = config.inst_num.NUM_OF_CORE_PER_CLUSTER
    sip_compute_total = batch_gemm_shape.thread_cnt_in_block()
    stat_ref = 0
    for idx, workload in enumerate(sic_workloads[sic_id]):
        lhs_tensor = _split_operands(workload.inputs[0], sip_total, sip_id)
        # lhs_tensor = workload.inputs[0]
        lhs_latency, lhs_leading = yield from _cdte_l3_to_shared(lhs_tensor, f"{idx}:lhs:l3->shared", cdte_ref)
        rhs_tensor = _split_operands(workload.inputs[1], sip_total, sip_id)
        # rhs_tensor = workload.inputs[1]
        rhs_latency, rhs_leading = yield from _cdte_l3_to_shared(rhs_tensor, f"{idx}:rhs:l3->shared", cdte_ref)
        cdte_latency = max(total_latency, lhs_latency + cdte_ref, rhs_latency + cdte_ref)
        quant_type = workload.attr.get("quant_type", QuantType.No_Quant)
        if quant_type == QuantType.Wf4g_Af8t:
            scaling_tensor = _split_operands(workload.inputs[2], sip_total, sip_id)
            scaling_latency, scaling_leading = yield from _cdte_l3_to_shared(
                scaling_tensor, f"{idx}:scaling:l3->shared", cdte_ref
            )
            cdte_latency = max(cdte_latency, scaling_latency + cdte_ref)
        cdte_latency = yield from bm.get_barrier(f"lhs_rhs_{kernel_id}_{sic_id}_{idx}", sip_total).wait(
            abs_ref + cdte_latency
        )
        cdte_latency -= abs_ref
        if sip_id < sip_compute_total:
            sip_workloads = _get_sip_workloads(workload, batch_gemm_shape, config, sip_id)
            gemm_latency = yield from _gemm_l2_kernel(sip_workloads, cdte_latency, idx)
        else:
            gemm_latency = cdte_latency
        gemm_latency = yield from bm.get_barrier(f"gemm_{kernel_id}_{sic_id}_{idx}", sip_total).wait(
            abs_ref + gemm_latency
        )
        gemm_latency -= abs_ref
        cdte_ref = max(cdte_latency - max(lhs_leading, rhs_leading), ping_buffer_release_ref)
        ping_buffer_release_ref = gemm_latency
        total_latency = max(total_latency, gemm_latency)
        if len(workload.outputs) > 0:
            res_tensor = _split_operands(workload.outputs[0], sip_total, sip_id)
            res_latency, res_leading = yield from _cdte_shared_to_l3(res_tensor, f"{idx}:res:shared->l3", gemm_latency)
            total_latency = max(total_latency, res_latency)

    return total_latency
