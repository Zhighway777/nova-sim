from dataclasses import dataclass, field
from typing import Generator, List, Tuple, Dict
from collections import deque

from nova_platform.base_model import (
    AddrDomain,
    DataflowActionComputeStat,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
)
from nova_platform.config import BossaNovaConfig, DType
from nova_platform.benchmark.op_base import Workload, OpBase, Operand, GridShape, list_product
from nova_platform.benchmark.batch_gemm import (
    XpuBatchGemmBase,
    MemArchType,
    BatchGemmGridShape,
    QuantType,
    bpe_for_quant,
)


from nova_platform.benchmark.utils import _iter_access_gen
from functools import lru_cache


class BatchGemmLocal(XpuBatchGemmBase):
    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        super().__init__(config, workload)
        self.mem_arch_type = MemArchType.DSM_LOCAL
    # 重写计算解的代价函数
    def _calc_sol_cost(self, batch_gemm_shape: BatchGemmGridShape):
        # 根据workload的inputs size和bandwidth计算输入延迟
        def cal_input_latency(wl: Workload):
            size = 0
            for i in wl.inputs:
                size += list_product(tuple(i.dim)) * i.bpe if i else 0
            if size == 0:
                return 0
            # TODO: 这里为什么要加上l3_to_l1_latency？
            return size / self.l3_bandwidth_per_sip + self.l3_to_l1_latency
        # 根据workload的outputs size和bandwidth计算输出延迟
        def cal_output_latency(wl):
            size = 0
            for i in wl.outputs:
                size += list_product(tuple(i.dim)) * i.bpe if i else 0
            if size == 0:
                return 0
            return size / self.l3_bandwidth_per_sip
        # 根据workload的计算量和计算能力计算计算延迟
        def cal_compute_latency(workload: Workload):
            ops = 2 * workload.attr["b"] * workload.attr["m"] * workload.attr["n"] * workload.attr["k"]
            l1_access = workload.attr["b"] * workload.attr["m"] * workload.attr["k"] * workload.dtype.get_bpe()
            l1_access += workload.attr["b"] * workload.attr["n"] * workload.attr["k"] * workload.dtype.get_bpe()
            # TODO: 为什么取最大值？
            return max(ops / self.xpu_FLOPS, l1_access / self.l1_bandwidth_per_sip)

        workloads = self._gen_workloads(batch_gemm_shape, True)
        latency_list = []
        for sic_id in workloads:
            for sip_id in workloads[sic_id]:
                latency = 0
                cdte_ref = 0
                compute_ref = 0
                ping_buffer_release_ref = 0
                out_ref = 0
                for idx, wl in enumerate(workloads[sic_id][sip_id]):
                    input_latency = cal_input_latency(wl)
                    output_latency = cal_output_latency(wl)
                    compute_latency = cal_compute_latency(wl)
                    compute_ref = cdte_ref + input_latency
                    cdte_ref = max(cdte_ref + input_latency - self.l3_to_l1_latency, ping_buffer_release_ref)
                    ping_buffer_release_ref = compute_ref + compute_latency
                    out_ref = compute_ref + compute_latency
                    latency = out_ref + output_latency

                latency_list.append(latency)
        return max(latency_list)

    def handle_pingpong_buf(self, batch_gemm_shape: BatchGemmGridShape, workloads: Dict[int, Dict[int, Workload]]):
        for sic_id in workloads:
            for sip_id in workloads[sic_id]:
                lhs_buffer = deque(maxlen=batch_gemm_shape.lhs_buf_cnt_l2)
                rhs_buffer = deque(maxlen=batch_gemm_shape.rhs_buf_cnt_l2)
                scaling_buffer = deque(maxlen=batch_gemm_shape.rhs_buf_cnt_l2)
                for wl in workloads[sic_id][sip_id]:
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

                    if self.quant_type == QuantType.Wf4g_Af8t:
                        scaling_operand = wl.inputs[2]
                        if scaling_operand in scaling_buffer:
                            wl.inputs[2] = None
                        else:
                            scaling_buffer.append(scaling_operand)

    def _gen_workloads(self, batch_gemm_shape: BatchGemmGridShape, first_sip_only=False):
        sip_workload = {}
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

        shape_lhs_l2 = [K_l2, shape_thread[1], shape_thread[2], 1]  # shape after splitting
        shape_rhs_l2 = [shape_block[0], K_l2, shape_block[2], 1]
        lhs_cnt_l2 = batch_gemm_shape.lhs_buf_cnt_l2
        rhs_cnt_l2 = batch_gemm_shape.rhs_buf_cnt_l2
        lhs_tensor_l3 = self.workload.inputs[0]
        rhs_tensor_l3 = self.workload.inputs[1]
        if self.quant_type == QuantType.Wf4g_Af8t:
            scaling_tensor_l3 = self.workload.inputs[2]
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
        for block_idx in range(block_total):
            for sic_idx in range(sic_total):
                if first_sip_only and sic_idx != 0:
                    break
                block_idx_vec = batch_gemm_shape.block_traverse_idx[sic_idx][block_idx]
                if block_idx_vec[0] < 0:
                    continue

                block_idx_x, block_idx_y, block_idx_z = block_idx_vec[:3]
                sip_workload.setdefault(sic_idx, {})
                for sip_idx in range(sip_total):
                    for k_l2_idx in range(k_loop_l2):
                        sip_eng_idx = sip_idx + sic_idx * self.sip_cnt
                        sip_idx_vec = batch_gemm_shape.thread_traverse_idx[sip_idx]
                        # print("!!!sip_idx_vec", sip_idx_vec)
                        sip_idx_x, sip_idx_y, sip_idx_z = sip_idx_vec[:3]
                        m_offset = shape_thread[1] * (sip_idx_y + thread_loop_y * block_idx_y)
                        # offset from l3 tensor
                        n_offset = shape_thread[0] * (sip_idx_x + thread_loop_x * block_idx_x)
                        b_offset = shape_thread[2] * (sip_idx_z + thread_loop_z * block_idx_z)
                        k_l2_idx_new = k_l2_idx
                        if block_idx & 1:
                            k_l2_idx_new = k_loop_l2 - 1 - k_l2_idx
                            # snake-like
                        k_offset = K_l2 * k_l2_idx_new
                        # print(
                        #     f"!sip_idx:{sip_idx}, m_offset:{m_offset}, n_offset:{n_offset}, b_offset:{b_offset}, k_offset:{k_offset}"
                        # )
                        # print(f"!sip_idx:{sip_idx}, M_l3:{M_l3}, N_l3:{N_l3}, B_l3:{B_l3}, K_l3:{K_l3}")
                        if m_offset >= M_l3 or n_offset >= N_l3 or k_offset >= K_l3 or b_offset >= B_l3:
                            continue

                        m_valid = shape_thread[1] if M_l3 >= m_offset + shape_thread[1] else M_l3 - m_offset
                        n_valid = shape_thread[0] if N_l3 >= n_offset + shape_thread[0] else N_l3 - n_offset
                        b_valid = shape_thread[2] if B_l3 >= b_offset + shape_thread[2] else B_l3 - b_offset
                        k_valid = K_l2 if K_l3 >= k_offset + K_l2 else (K_l3 - k_offset)

                        # lhs_offset = [k_offset, m_offset, b_offset, 0]
                        # rhs_offset = [n_offset, k_offset, b_offset, 0]
                        # res_offset = [n_offset, m_offset, b_offset, 0]
                        lhs_offset = [b_offset, m_offset, k_offset]
                        rhs_offset = [b_offset, k_offset, n_offset]
                        res_offset = [b_offset, m_offset, n_offset]
                        bias_offset = [n_offset, 0, 0, 0]
                        sip_workload[sic_idx].setdefault(sip_idx, [])
                        inputs = [
                            Operand(
                                dim=(b_valid, m_valid, k_valid),
                                addr=lhs_tensor_l3.addr,
                                bpe=lhs_tensor_l3.bpe,
                                dim_offset=lhs_offset,
                                dim_stride=lhs_tensor_l3.dim_stride,
                            ),
                            Operand(
                                dim=(b_valid, k_valid, n_valid),
                                addr=rhs_tensor_l3.addr,
                                bpe=rhs_tensor_l3.bpe,
                                dim_offset=rhs_offset,
                                dim_stride=rhs_tensor_l3.dim_stride,
                            ),
                        ]
                        if self.quant_type == QuantType.Wf4g_Af8t:
                            quant_group_size = self.workload.attr.get("quant_group_size", 32)
                            quant_group_num = scaling_tensor_l3.dim[1]
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
                                    addr=scaling_tensor_l3.addr,
                                    bpe=scaling_tensor_l3.bpe,
                                    dim_offset=(b_offset, group_num_offset, n_offset),
                                    dim_stride=scaling_tensor_l3.dim_stride,
                                )
                            )
                        if k_l2_idx == k_loop_l2 - 1:
                            outputs = [
                                Operand(
                                    dim=(b_valid, m_valid, n_valid),
                                    addr=res_tensor_l3.addr,
                                    bpe=res_tensor_l3.bpe,
                                    dim_offset=res_offset,
                                    dim_stride=res_tensor_l3.dim_stride,
                                )
                            ]
                        else:
                            outputs = []
                        attr = {**self.workload.attr, "b": b_valid, "m": m_valid, "n": n_valid, "k": k_valid}
                        sip_workload[sic_idx][sip_idx].append(Workload(inputs, outputs, attr=attr))

        self.handle_pingpong_buf(batch_gemm_shape, sip_workload)
        return sip_workload


@lru_cache()
def tile_local_gemm_workload(config: BossaNovaConfig, chip_workload: Workload) -> Tuple[Workload, BatchGemmGridShape]:
    batch_gemm = BatchGemmLocal(config, chip_workload)
    batch_gemm.split()
    batch_gemm.impl()
    best_shape = batch_gemm.get_best_shape()
    return batch_gemm.get_tiled_workloads(), best_shape


def dsm_local_gemm_kernel(
    sip_workloads: List[Workload],
    config: BossaNovaConfig,
    subthread_num: int,
    relative_ts: int,
):

    lhs_buffer = deque(maxlen=2)
    rhs_buffer = deque(maxlen=2)
    scaling_buffer = deque(maxlen=2)

    def _iter_tensor_addr(tensor: Operand, rw) -> Generator[DataflowActionMemoryAccess, None, None]:
        mem_access = tensor.get_contiguous_mem_accesses()
        for addr, size in mem_access:
            yield DataflowActionMemoryAccess(addr, size, rw)

    def _get_minimum_iter_k(b, m, n, mac_dtype):
        iter_k = 32
        if mac_dtype == DType.FP4:
            iter_k = 64
        throughput = config.compute.thread_2d_throughput[dtype]
        l1_latency = config.bw.l0.local.pre_latency
        k = 0
        mac_cycle = 0
        while mac_cycle < l1_latency:
            k += iter_k
            mac_cycle = b * m * n * k / throughput
        return k

    def _get_operands(sip_workload: Workload, stat_ref: int, idx: int):

        def cdte_operand(tensor, cdte_name, ref):
            tensor_gen = _iter_access_gen(list(_iter_tensor_addr(tensor, "r")))
            next(tensor_gen)
            data_size = list_product(tuple(tensor.dim)) * tensor.bpe
            tensor_read = DataflowActionMemoryStat(
                total_count=int(data_size),
                master=DataflowActionType.CDTE,
                src=AddrDomain.LOCAL,
                dst=AddrDomain.L3,
                rw="r",
                relative_ts=ref,
                memory_access_list=tensor_gen.send(data_size),
                name=cdte_name,
            )
            yield tensor_read
            return tensor_read.latency, tensor_read.leading_latency

        lhs_tensor = sip_workload.inputs[0]
        rhs_tensor = sip_workload.inputs[1]
        quant_type = sip_workload.attr.get("quant_type", QuantType.No_Quant)
        scaling_tensor = None
        if quant_type == QuantType.Wf4g_Af8t:
            if len(sip_workload.inputs) >= 3 and sip_workload.inputs[2]:
                scaling_tensor = sip_workload.inputs[2]

        lhs_ref = 0
        rhs_ref = 0
        lhs_leading_latency = 0
        rhs_leading_latency = 0
        if lhs_tensor and lhs_tensor not in lhs_buffer:
            lhs_ref, lhs_leading_latency = yield from cdte_operand(lhs_tensor, f"{idx} lhs:l3->local", stat_ref)
            lhs_buffer.append(lhs_tensor)

        if rhs_tensor and rhs_tensor not in rhs_buffer:
            rhs_ref, rhs_leading_latency = yield from cdte_operand(rhs_tensor, f"{idx} rhs:l3->local", stat_ref)
            rhs_buffer.append(rhs_tensor)

        latency = max(lhs_ref, rhs_ref)
        leading = max(lhs_leading_latency, rhs_leading_latency)

        if scaling_tensor and rhs_tensor not in scaling_buffer:
            scaling_ref, scaling_leading_latency = yield from cdte_operand(
                scaling_tensor, f"{idx} scaling:l3->local", stat_ref
            )
            scaling_buffer.append(rhs_tensor)
            latency = max(latency, scaling_ref)
            leading = max(leading, scaling_leading_latency)
        return latency, leading

    total_latency = relative_ts
    cdte_ref = relative_ts
    out_release_ref = relative_ts
    compute_release_ref = relative_ts
    ping_buffer_release_ref = relative_ts

    for idx, sip_workload in enumerate(sip_workloads):
        quant_type = sip_workload.attr.get("quant_type", QuantType.No_Quant)
        mac_dtype = sip_workload.dtype
        dtype = sip_workload.dtype
        bpe = bpe_for_quant(sip_workload.dtype, quant_type)
        if quant_type == QuantType.Wf4g_Af8t:
            mac_dtype = DType.FP4
            if len(sip_workload.inputs) > 3 and sip_workload.inputs[2]:
                scaling_tensor = sip_workload.inputs[2]
        elif quant_type == QuantType.Wf8t_Af8t:
            mac_dtype = DType.FP8

        in_b = sip_workload.attr["b"]
        in_m = sip_workload.attr["m"]
        in_n = sip_workload.attr["n"]
        in_k = sip_workload.attr["k"]

        local_input_size = in_b * in_m * in_k * bpe["lhs"] * in_b * in_n * in_k * bpe["rhs"]
        input_name = "lhs+rhs"
        quant_group_num = 0
        if quant_type == QuantType.Wf4g_Af8t:
            quant_group_size = sip_workload.attr.get("quant_group_size", 32)
            quant_group_num = (in_k + quant_group_size - 1) // quant_group_size
            local_input_size += in_b * quant_group_num * in_n * bpe["scaling"]
            input_name = "lhs+rhs+scaling"
        local_input_gen = _iter_access_gen([DataflowActionMemoryAccess(0x4000000000, local_input_size, "r")])
        next(local_input_gen)

        oprands_latency, oprands_leading = yield from _get_operands(sip_workload, cdte_ref, idx)
        iter_ref = max(cdte_ref + oprands_latency, compute_release_ref)
        cdte_ref = max(cdte_ref + oprands_latency - oprands_leading, ping_buffer_release_ref)
        out_ref = iter_ref

        iter_k = _get_minimum_iter_k(in_b, in_m, in_n, dtype)
        iter_k = min(in_k, iter_k)
        for i in range(0, in_k, iter_k):
            k = min(iter_k, in_k - i)
            l0_input_size = int(in_m * k * bpe["lhs"] + in_n * k * bpe["rhs"] + quant_group_num * k * bpe["scaling"])
            l0_input_read = DataflowActionMemoryStat(
                total_count=l0_input_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.LOCAL,
                rw="r",
                relative_ts=iter_ref,
                memory_access_list=local_input_gen.send(l0_input_size),
                name=f"{idx} {input_name}:local->l0",
            )
            yield l0_input_read
            compute_ref = iter_ref + l0_input_read.leading_latency
            ping_buffer_release_ref = iter_ref + l0_input_read.latency
            scalar_ops = 4 * subthread_num
            compute_stat1 = DataflowActionComputeStat(
                name=f"{idx} scalar",
                compute_scalar_cycle=scalar_ops,
                relative_ts=compute_ref,
            )
            yield compute_stat1
            scalar_cost = compute_stat1.latency

            compute_mac = in_m * in_n * k
            compute_stat2 = DataflowActionComputeStat(
                name=f"{idx} mac",
                compute_2d_ops={mac_dtype: compute_mac * 2},
                relative_ts=compute_ref,
            )
            yield compute_stat2
            VMM = 10
            compute_vmm = DataflowActionComputeStat(
                name=f"{idx} vmm",
                compute_nop_cycle=VMM,
                relative_ts=compute_ref + compute_stat2.latency,
            )
            yield compute_vmm
            out_ref = compute_ref
            compute_latency = compute_ref + compute_stat2.latency + compute_vmm.latency
            iter_ref = (
                max(
                    compute_latency,
                    iter_ref + l0_input_read.latency,
                )
                - l0_input_read.leading_latency
            )
            compute_release_ref = iter_ref
        out_ref = max(out_ref, out_release_ref)
        if len(sip_workload.outputs):
            out_gen = _iter_access_gen(list(_iter_tensor_addr(sip_workload.outputs[0], "w")))
            next(out_gen)
            if quant_type == QuantType.Wf4g_Af8t or quant_type == QuantType.Wf8t_Af8t:
                compute_convert = DataflowActionComputeStat(
                    name=f"{idx} convert",
                    compute_1d_ops={DType.FP32: in_m * in_n},
                    relative_ts=compute_ref,
                )
                yield compute_convert
                compute_latency = max(compute_convert.latency, compute_latency)

            out_size = in_m * in_n * bpe["res"]
            out_wirte = DataflowActionMemoryStat(
                total_count=out_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=out_ref,
                memory_access_list=out_gen.send(out_size),
                name=f"{idx} out",
            )
            yield out_wirte
            out_release_ref = out_ref + out_wirte.latency - out_wirte.leading_latency
            total_latency = out_ref + out_wirte.latency
    return total_latency
