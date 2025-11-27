from dataclasses import dataclass, field
from typing import Generator, List, Tuple
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.config import TOPO
from nova_platform.executor.nova_platform_barrier import BarrierManager
from nova_platform.executor.nova_platform_event import EventManager
from nova_platform.benchmark.op_base import Operand, list_product, Workload
from nova_platform.benchmark.batch_gemm_local import tile_local_gemm_workload, dsm_local_gemm_kernel
from nova_platform.benchmark.batch_gemm_shared import tile_shared_gemm_workload, dsm_shared_gemm_kernel
from nova_platform.benchmark.eccl import SimpleProtoPrimitives
from nova_platform.config import BossaNovaConfig

from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    DataflowActionComputeStat,
    AddrDomain,
)

from functools import lru_cache


def action_tensor_to_operand(tensor):
    return Operand(
        dim=tuple(tensor.dims),
        addr=tensor.addr,
        bpe=tensor.bpe,
        dim_offset=tuple(tensor.offsets),
        dim_stride=tuple(tensor.stride_dims),
    )


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16


@dataclass
class CoreCost(BaseCoreStat):
    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)
    sip_workloads: list = field(default_factory=list)


def get_peer_ranks(channel, rank, rank_cnt):
    pre = rank - 1 if rank != 0 else rank_cnt - 1
    next = rank + 1 if rank < rank_cnt - 1 else 0
    return pre, next


@lru_cache()
def split_workload(workload: Workload, nranks: int, rank: int) -> List[Tuple[Workload]]:
    b = workload.inputs[0].dim[0]
    m = workload.inputs[0].dim[1]
    k = workload.inputs[0].dim[2]
    workloads = []
    tmp_buffer = Operand(
        dim=workload.inputs[0].dim,
        addr=0x53000000000,
        dim_offset=(0, 0, 0, 0),
        bpe=workload.inputs[0].bpe,
        dim_stride=workload.inputs[0].dim,
    )
    output = workload.outputs[0]

    def get_received_peer_ranks(rank, rank_cnt):
        peer_rank = list(range(0, rank_cnt))
        peer_rank = list(reversed(peer_rank[0 : rank + 1])) + list(reversed(peer_rank[rank + 1 :]))
        return peer_rank

    peers = get_received_peer_ranks(rank, nranks)
    for step, peer in enumerate(peers):
        step_output = Operand(
            dim=output.dim,
            addr=output.addr,
            bpe=output.bpe,
            dim_offset=(0, m * peer, 0, 0),
            dim_stride=output.dim_stride,
        )

        if step % 2 == 0:
            ag_work = Workload((workload.inputs[0],), (tmp_buffer,), dtype=workload.dtype)
            gemm_work = Workload((workload.inputs[0], workload.inputs[1]), (step_output,), dtype=workload.dtype)
        else:
            ag_work = Workload((tmp_buffer,), (workload.inputs[0],))
            gemm_work = Workload((tmp_buffer, workload.inputs[1]), (step_output,), dtype=workload.dtype)
        workloads.append((ag_work, gemm_work))
    return workloads


bm = BarrierManager()
@dataclass
class XPUAllGatherGemmAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def _basic_stat_info(self):
        pass

    def get_kernel_idx(self):
        return self.data[0], self.data[1]

    def get_workload(self) -> Workload:
        workload = Workload(
            inputs=(
                action_tensor_to_operand(self.inputs[0].tensor[0]),
                action_tensor_to_operand(self.inputs[1].tensor[0]),
            ),
            outputs=(action_tensor_to_operand(self.outputs[0].tensor[0]),),
            dtype=self.get_dtype(),
        )
        return workload

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:

        total_latency = 0
        sic_id, sip_id = self.get_kernel_idx()
        rank_num = self.topo.value[1]
        rank = self.config.gcu_id
        total_sip_num = (
            self.config.inst_num.NUM_OF_CLUSTER
            * self.config.inst_num.NUM_OF_CORE_PER_CLUSTER
            * self.config.inst_num.NUM_OF_DIE
        )

        chip_workload = self.get_workload()
        tiled_workloads = _tile_workload(chip_workload, rank_num * 2)
        tile_num = len(tiled_workloads)
        slice_per_rank = (tile_num + rank_num - 1) // rank_num

        def _gemm(tile_idx, relative_ts):
            gemm_workloads, best_shape = tile_shared_gemm_workload(self.config, tiled_workloads[tile_idx])
            latency = yield from dsm_shared_gemm_kernel(
                gemm_workloads,
                best_shape,
                self.config,
                f"{self.case_id}_{rank}_{tile_idx}",
                sic_id,
                sip_id,
                relative_ts,
                self.ref,
            )
            return latency

        def _allgather_master_mode(send_tile_idx, recv_tile_idx, trunk_idx, relative_ts):
            channel_cnt = 2
            if sic_id not in [0, 1] or sip_id != 0:
                return relative_ts

            def _get_channels(sic_id, sic_num):
                num_channels = channel_cnt // sic_num
                start_channel = sic_id * num_channels
                end_channel = start_channel + num_channels
                return list(range(start_channel, end_channel))

            workload = tiled_workloads[send_tile_idx]
            in_tensor = workload.inputs[0]
            element_cnt = list_product(in_tensor.dim) * in_tensor.bpe
            element_cnt_per_channel = element_cnt // channel_cnt
            channel_list = _get_channels(sic_id, 2)
            stat_ref = relative_ts
            total_latency = 0
            for channel_id in channel_list:
                channel_offset = channel_id * element_cnt_per_channel
                pre_rank, next_rank = get_peer_ranks(channel_id, rank, rank_num)
                prims = SimpleProtoPrimitives(
                    case_id=f"{self.case_id}",
                    channel_id=channel_id,
                    recvPeers=[pre_rank],
                    sendPeers=[next_rank],
                    sendBuff=in_tensor.addr,
                    recvBuff=in_tensor.addr,
                    rank=rank,
                    ref=self.ref,
                    esl_bw_factor=16,
                )
                send_latency = yield from prims.directSend(channel_offset, element_cnt_per_channel, stat_ref, trunk_idx)
                recv_latency = yield from prims.directRecv(
                    channel_offset, element_cnt_per_channel, send_latency, trunk_idx
                )
                total_latency = max(total_latency, recv_latency)
            return total_latency

        total_latency = 0
        gemm_ref = 0
        ag_ref = 0
        gemm_latency = 0
        allgather_latency = 0
        trunk_ite_idx = 0
        trunk_ready_ref = {}
        for i in range(rank_num):
            for j in range(slice_per_rank):
                send_tile_idx = ((rank - i) % rank_num) * slice_per_rank + j
                recv_tile_idx = ((rank - i - 1) % rank_num) * slice_per_rank + j
                gemm_ref = max(gemm_ref, trunk_ready_ref.get(send_tile_idx, 0))
                gemm_latency = yield from _gemm(send_tile_idx, gemm_ref)
                if i < rank_num - 1:

                    allgather_latency = yield from _allgather_master_mode(
                        send_tile_idx, recv_tile_idx, trunk_ite_idx, ag_ref
                    )
                else:
                    allgather_latency = ag_ref
                allgather_latency = yield from bm.get_barrier(
                    f"{self.case_id}_{rank}_{trunk_ite_idx}", total_sip_num
                ).wait(self.ref + allgather_latency)
                allgather_latency -= self.ref
                trunk_ready_ref[recv_tile_idx] = allgather_latency
                trunk_ite_idx += 1
                gemm_ref = gemm_latency

                ag_ref = allgather_latency
        total_latency = max(gemm_latency, allgather_latency)
        self.core_cost.latency = total_latency


def _tile_workload(chip_workload: Workload, tile_num):
    lhs = chip_workload.inputs[0]
    rhs = chip_workload.inputs[1]
    res = chip_workload.outputs[0]
    B = lhs.dim[0]
    M = lhs.dim[1]
    K = lhs.dim[2]
    N = rhs.dim[2]
    m_per_tile = (M + tile_num - 1) // tile_num
    tiled_workloads = []

    for idx in range(tile_num):
        offset_m = m_per_tile * idx
        if offset_m >= M:
            break
        valid_m = m_per_tile if M >= m_per_tile + offset_m else M - offset_m
        tiled_lhs = Operand(
            dim=(B, valid_m, K),
            addr=lhs.addr,
            bpe=lhs.bpe,
            dim_offset=(0, offset_m, 0),
            dim_stride=lhs.dim_stride,
        )
        tiled_res = Operand(
            dim=(B, valid_m, N),
            addr=res.addr,
            bpe=res.bpe,
            dim_offset=(0, offset_m, 0),
            dim_stride=res.dim_stride,
        )
        attr = {"b": B, "m": valid_m, "n": N, "k": K}
        tiled_workloads.append(
            Workload(inputs=[tiled_lhs, rhs], outputs=[tiled_res], dtype=chip_workload.dtype, attr=attr)
        )
    return tiled_workloads


def _get_comm_tile_num(M, N, K, bpe, ranks, config):
    min_m = 128
    max_tile_num = (M + min_m - 1) // min_m
    min_tile_num = ranks
    tile_num = max_tile_num
    l3_bw = 128
    while tile_num >= min_tile_num:
        tiled_m = (M + tile_num - 1) // tile_num
        mac_ops = 2 * tiled_m * N * K
        mac_cycles = mac_ops / 4096
        l3_access_bytes = bpe[0] * tiled_m * K + bpe[1] * K * N + bpe[2] * tiled_m * N
        l3_bw_required = l3_access_bytes
        if l3_bw_required < l3_bw:
            break
        tile_num = tile_num / 2
