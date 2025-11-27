from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

from nova_platform.base_model import (
    AddrDomain,
    DType,
    DataflowActionComputeStat,
    DataflowActionMemoryStat,
    DataflowActionType,
    DataflowActionMemoryAccess,
)
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.dataflow.action.diag_action import DiagDataflowAction, DiagTensor

import logging

logger = logging.getLogger(__file__)

BYPTES_PER_OA = 512

SIP_1D_EFFICIENCY = 0.7
SIP_2D_EFFICIENCY = 0.7


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    head_dim: int = 128
    head_num: int = 1
    row_block: int = 1
    col_block: int = 1
    kv_seq: int = 1024
    with_dropout: bool = True


def sdpa_kernel(row_block, col_block, head_num, kv_seq, dtype: DType):
    return InstructionInfo(dtype=dtype, row_block=row_block, col_block=col_block, head_num=head_num, kv_seq=kv_seq)


def cal_dropout_cycles_per_element(rng_dtype, round, throughput_1D_fp32):
    rng_num_per_iter = {"u32": 4, "u8": 16}
    dropout_1d_cal = {"u32": 2, "u8": 4}
    rng_cycles = 10 + round * 10
    return (
        rng_cycles / (rng_num_per_iter[rng_dtype] * throughput_1D_fp32) +
        dropout_1d_cal[rng_dtype] / throughput_1D_fp32
    )


@dataclass
class CoreCost(BaseCoreStat):
    oa_occupation: int = 0  # Bytes
    main_body_length: int = 0
    main_body_num: int = 0
    unroll_num: int = 8
    subthread_num: int = 8

    data_preparation_cost: int = 0
    compute_cost: int = 0

    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)


@dataclass
class XPUSdpaAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        col_block = self.data[0]
        row_block = self.data[1]
        head_num = self.data[2]
        kv_seq = self.inputs[1].tensor[0].dims[1]
        head_dim = self.inputs[1].tensor[0].dims[0]
        return row_block, col_block, head_num, kv_seq, head_dim

    def _get_l0_occupation(self):  # step 3
        self.core_cost.oa_occupation = 3 * BYPTES_PER_OA  # 2 for inputs, 1 for outputs

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = self.core_cost.unroll_num * \
            self.core_cost.subthread_num

    def _iter_addr(self, tensor: DiagTensor, rw) -> DataflowActionMemoryAccess:
        base_addr = tensor.addr
        data_size = tensor.dims[0] * tensor.dims[1] * \
            tensor.dims[2] * tensor.dims[3] * tensor.bpe
        return DataflowActionMemoryAccess(base_addr, data_size, rw)

    # parallel patten: 2d&sfu-1d, 2d-1d-sfu, 2d&1d-sfu
    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        with_dropout = self.dataflow_config.get("bench_sdpa_with_dropout", 1)
        assert self.config.compute.compute_parallel_capability in [
            0, 1, 2]  # 2d&sfu-1d, 2d-1d-sfu, 2d&1d-sfu
        compute_parallel_capability = self.config.compute.compute_parallel_capability
        row_block, col_block, head_num, kv_seq, head_dim = self.get_valid_shape()

        dtype = self.get_dtype()
        self._basic_stat_info()

        self.core_cost.instruction_info = InstructionInfo(
            dtype=dtype,
            head_dim=head_dim,
            head_num=head_num,
            row_block=row_block,
            col_block=col_block,
            kv_seq=kv_seq,
            with_dropout=with_dropout,
        )
        instruction_info = self.core_cost.instruction_info
        # print("!!!instruction_info", self.core_cost.instruction_info)
        bpe = instruction_info.dtype.get_bpe()

        def shape_nbypes(shape):
            return reduce(lambda a, b: a * b, shape, 1) * bpe

        q_access = self._iter_addr(self.inputs[0].tensor[0], "r")
        q_gen = self._iter_access_gen([q_access])
        next(q_gen)

        k_access = self._iter_addr(self.inputs[1].tensor[0], "r")
        k_gen = self._iter_access_gen([k_access])
        next(k_gen)

        v_access = self._iter_addr(self.inputs[2].tensor[0], "r")
        v_gen = self._iter_access_gen([v_access])
        next(v_gen)

        out_access = self._iter_addr(self.outputs[0].tensor[0], "w")
        out_gen = self._iter_access_gen([out_access])
        next(out_gen)

        stat_ref = 0
        total_latency = 0
        self.core_cost.tensor_macs[self.get_dtype()] = 0
        for head_index in range(instruction_info.head_num):
            for col_index in range(0, instruction_info.kv_seq, instruction_info.col_block):
                col_size = min(instruction_info.col_block,
                               instruction_info.kv_seq - col_index)
                q_tile_shape = [instruction_info.row_block,
                                instruction_info.head_dim]
                k_tile_shape = [instruction_info.head_dim, col_size]
                v_tile_shape = [col_size, instruction_info.head_dim]
                s_tile_shape = [instruction_info.row_block, col_size]
                o_tile_shape = [instruction_info.row_block,
                                instruction_info.head_dim]
                q_tile_size = shape_nbypes(q_tile_shape)
                k_tile_size = shape_nbypes(k_tile_shape)
                v_tile_size = shape_nbypes(v_tile_shape)
                s_tile_size = shape_nbypes(s_tile_shape)
                o_tile_size = shape_nbypes(o_tile_shape)
                o_elements = instruction_info.row_block * instruction_info.head_dim
                k_read = DataflowActionMemoryStat(
                    total_count=k_tile_size,
                    master=DataflowActionType.CDTE,
                    src=AddrDomain.SHARED,
                    dst=AddrDomain.L3,
                    rw="r",
                    relative_ts=stat_ref,
                    memory_access_list=k_gen.send(k_tile_size),
                    name="k_tile",
                )
                yield k_read
                ref_after_k_read = stat_ref + k_read.latency
                self.core_cost.ld_shared_l3 += k_tile_size
                self.core_cost.ld_l0_shared += k_tile_size
                v_read = DataflowActionMemoryStat(
                    total_count=v_tile_size,
                    master=DataflowActionType.CDTE,
                    src=AddrDomain.SHARED,
                    dst=AddrDomain.L3,
                    rw="r",
                    relative_ts=ref_after_k_read,
                    memory_access_list=k_gen.send(v_tile_size),
                    name="v_tile",
                )
                yield v_read
                self.core_cost.ld_shared_l3 += v_tile_size
                self.core_cost.ld_l0_shared += v_tile_size

                q_read = DataflowActionMemoryStat(
                    total_count=q_tile_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="r",
                    relative_ts=ref_after_k_read,
                    memory_access_list=q_gen.send(q_tile_size),
                    name="q_tile",
                )
                yield q_read
                ref_after_q_leading = ref_after_k_read + q_read.leading_latency

                matmul_qk_macs = instruction_info.row_block * \
                    instruction_info.head_dim * col_size
                self.core_cost.tensor_macs[dtype] += matmul_qk_macs
                compute_matmul_qk = DataflowActionComputeStat(
                    compute_2d_ops={
                        instruction_info.dtype: matmul_qk_macs * 2},
                    compute_2d_efficiency=SIP_2D_EFFICIENCY,
                    relative_ts=ref_after_q_leading,
                    name="matmul_qk",
                )
                yield compute_matmul_qk
                self.core_cost.st_l0_shared += s_tile_size

                if compute_parallel_capability == 1:
                    ref_sfu = ref_after_q_leading
                elif compute_parallel_capability == 2:
                    ref_sfu = ref_after_q_leading
                else:
                    ref_sfu = ref_after_q_leading + compute_matmul_qk.latency
                softmax_sfu_ops = instruction_info.row_block * col_size
                self.core_cost.sfu_ops += softmax_sfu_ops
                compute_softmax_sfu = DataflowActionComputeStat(
                    compute_msf_ops=softmax_sfu_ops,
                    compute_sfu_efficiency=SIP_1D_EFFICIENCY,
                    relative_ts=ref_sfu,
                    name="softmax_sfu",
                )
                yield compute_softmax_sfu

                if compute_parallel_capability == 1:
                    ref_1d = ref_after_q_leading
                elif compute_parallel_capability == 2:
                    ref_1d = ref_after_q_leading + compute_matmul_qk.latency
                else:
                    ref_1d = ref_after_q_leading
                softmax_1d_ops = 7 * instruction_info.row_block * col_size
                self.core_cost.vector_ops[dtype] += softmax_1d_ops
                compute_softmax_1d = DataflowActionComputeStat(
                    compute_1d_ops={DType.FP32: softmax_1d_ops},
                    compute_1d_efficiency=SIP_1D_EFFICIENCY,
                    relative_ts=ref_1d,
                    name="softmax_1d",
                )
                yield compute_softmax_1d
                dropout_compute_latency = 0
                if with_dropout:
                    throughput_1D_fp32 = self.config.compute.thread_1d_throughput[DType.FP32]
                    dropout_cycles_per_element = cal_dropout_cycles_per_element(
                        "u8", 7, throughput_1D_fp32)
                    dropout_1d_ops = int(
                        instruction_info.row_block * col_size *
                        dropout_cycles_per_element * throughput_1D_fp32
                    )
                    ref_dropout = ref_1d + compute_softmax_1d.latency
                    self.core_cost.vector_ops[dtype] += dropout_1d_ops
                    compute_dropout_1d = DataflowActionComputeStat(
                        compute_1d_ops={DType.FP32: dropout_1d_ops},
                        compute_1d_efficiency=SIP_1D_EFFICIENCY,
                        relative_ts=ref_dropout,
                        name="dropout_1d",
                    )
                    yield compute_dropout_1d
                    dropout_compute_latency = compute_dropout_1d.latency

                if compute_parallel_capability == 1:
                    ref_pv = ref_after_q_leading + compute_matmul_qk.latency
                elif compute_parallel_capability == 2:
                    ref_pv = (
                        ref_after_q_leading
                        + compute_matmul_qk.latency
                        + compute_softmax_1d.latency
                        + dropout_compute_latency
                    )
                else:
                    ref_pv = ref_after_q_leading + compute_matmul_qk.latency + compute_softmax_sfu.latency
                matmul_pv_macs = instruction_info.row_block * \
                    instruction_info.head_dim * col_size
                self.core_cost.tensor_macs[dtype] += matmul_pv_macs
                compute_matmul_pv = DataflowActionComputeStat(
                    compute_2d_ops={
                        instruction_info.dtype: matmul_pv_macs * 2},
                    compute_2d_efficiency=SIP_2D_EFFICIENCY,
                    relative_ts=ref_pv,
                    name="matmul_pv",
                )
                yield compute_matmul_pv

                # epilog_mul_exp
                epilog_1d_ops = o_elements
                if col_index != 0:
                    epilog_1d_ops += o_elements * 2

                # epilog_div_exp
                epilog_1d_ops += o_elements

                if compute_parallel_capability == 2:
                    ref_epilog = (
                        ref_after_q_leading
                        + compute_matmul_qk.latency
                        + compute_softmax_1d.latency
                        + dropout_compute_latency
                        + compute_matmul_pv.latency
                    )
                else:
                    ref_epilog = ref_after_q_leading + \
                        compute_softmax_1d.latency + dropout_compute_latency
                self.core_cost.vector_ops[dtype] += epilog_1d_ops
                compute_epilog_1d = DataflowActionComputeStat(
                    compute_1d_ops={instruction_info.dtype: epilog_1d_ops},
                    compute_1d_efficiency=SIP_1D_EFFICIENCY,
                    relative_ts=ref_epilog,
                    name="epilog_1d",
                )
                yield compute_epilog_1d

                o_write = DataflowActionMemoryStat(
                    total_count=o_tile_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="w",
                    relative_ts=ref_after_q_leading,
                    memory_access_list=out_gen.send(o_tile_size),
                    name="o_tile",
                )
                yield o_write
                self.core_cost.st_shared_l3 += o_tile_size
                read_kv_latency = stat_ref + k_read.latency + v_read.latency
                read_q_latency = ref_after_q_leading + q_read.latency
                write_o_latency = ref_after_q_leading + o_write.latency
                compute_latency = 0
                if compute_parallel_capability == 0:  # 2d&sfu-1d,
                    compute_latency = max(
                        ref_pv + compute_matmul_pv.latency,
                        ref_epilog + compute_epilog_1d.latency,
                    )
                elif compute_parallel_capability == 1:  # 2d-1d-sfu,
                    compute_latency = max(
                        ref_pv + compute_matmul_pv.latency,
                        ref_epilog + compute_epilog_1d.latency,
                        ref_sfu + compute_softmax_sfu.latency,
                    )
                else:  # 2d&1d-sfu
                    compute_latency = max(
                        ref_sfu + compute_softmax_sfu.latency,
                        ref_epilog + compute_epilog_1d.latency,
                    )
                total_latency = max(
                    total_latency,
                    read_kv_latency,
                    compute_latency,
                    read_q_latency,
                    write_o_latency,
                )
                stat_ref = max(total_latency - k_read.latency,
                               read_kv_latency - k_read.leading_latency)
        self.core_cost.latency = total_latency
        return

    def _basic_stat_info(self):
        dtype = self.core_cost.instruction_info.dtype
        self.core_cost.vector_ops[dtype] = 0
        self.core_cost.tensor_macs[dtype] = 0
