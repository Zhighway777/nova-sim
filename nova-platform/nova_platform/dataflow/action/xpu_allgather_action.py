from dataclasses import dataclass, field
from typing import Generator, List
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.config import TOPO
from nova_platform.benchmark.op_base import Operand, list_product
from nova_platform.benchmark.eccl import SimpleProtoPrimitives

from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
)

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


@dataclass
class XPUAllGatherAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_kernel_idx(self):
        return self.data[0], self.data[1]

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        if self.topo in [TOPO.FULLMESH8]:
            yield from self.fullmesh_allgather()
        elif self.topo in [TOPO.SUPERNODE4, TOPO.SUPERNODE8, TOPO.SUPERNODE16, TOPO.SUPERNODE32]:
            yield from self.ring_allgather()
        else:
            raise NotImplementedError

    def ring_allgather(self) -> Generator[DataflowActionMemoryStat, None, None]:
        channel_cnt = 8
        rank_num = self.topo.value[1]
        def get_peer_ranks(channel, rank, rank_cnt):
            pre = rank - 1 if rank != 0 else rank_cnt - 1
            next = rank + 1 if rank < rank_cnt - 1 else 0
            return pre, next

        def get_channels(sic_id, sic_num):
            num_channels = channel_cnt // sic_num
            start_channel = sic_id * num_channels
            end_channel = start_channel + num_channels
            return list(range(start_channel, end_channel))

        sic_id, sip_id = self.get_kernel_idx()
        if sic_id not in [0, 1] or sip_id != 0:
            return
        rank = self.config.gcu_id
        in_tensor = action_tensor_to_operand(self.inputs[0].tensor[0])
        out_tensor = action_tensor_to_operand(self.outputs[0].tensor[0])
        dtype = self.get_dtype()
        bpe = dtype.get_bpe()

        element_cnt = list_product(in_tensor.dim) * bpe
        element_cnt_per_channel = element_cnt // channel_cnt
        stat_ref = 0
        total_latency = 0
        channel_list = get_channels(sic_id, 2)
        for channel_id in channel_list:
            # print(
            #     f"!!!gcu:{self.config.gcu_id}, sic: {sic_id}, channel:{channel_id}",
            # )
            channel_offset = channel_id * element_cnt_per_channel
            dst_offset = element_cnt * rank + channel_offset
            pre_rank, next_rank = get_peer_ranks(channel_id, rank, rank_num)
            prims = SimpleProtoPrimitives(
                case_id=self.case_id,
                channel_id=channel_id,
                recvPeers=[pre_rank],
                sendPeers=[next_rank],
                sendBuff=in_tensor.addr,
                recvBuff=out_tensor.addr,
                rank=rank,
                ref=self.ref,
                esl_bw_factor=16,
            )
            # step 0
            if in_tensor.addr + channel_offset == out_tensor.addr + dst_offset:
                latency = yield from prims.directSend(channel_offset, element_cnt_per_channel, stat_ref, 0)
            else:
                latency = yield from prims.directCopySend(
                    channel_offset, dst_offset, element_cnt_per_channel, stat_ref, 0
                )

            # step 1 ~ k-2
            for idx in range(1, rank_num - 1):
                latency = yield from prims.directRecvCopySend(0, element_cnt_per_channel, latency, idx - 1, idx)

            # final
            latency = yield from prims.directRecv(0, element_cnt_per_channel, latency, rank_num - 2)
            total_latency = max(total_latency, latency)
            # print(
            #     f"!!!gcu:{self.config.gcu_id}, sic: {sic_id}, channel:{channel_id} done",
            # )
        self.core_cost.latency = total_latency
        print(
            f"!!!gcu:{self.config.gcu_id}, sic: {sic_id} done",
        )

    def fullmesh_allgather(self) -> Generator[DataflowActionMemoryStat, None, None]:

        def get_peer_ranks(rank_num, rank):
            peer_rank = list(range(0, rank_num))
            peer_rank = peer_rank[rank + 1 :] + peer_rank[0:rank]
            return peer_rank

        sic_id, sip_id = self.get_kernel_idx()
        if sic_id not in [0, 1] or sip_id != 0:
            return
        # sic 0 for mesh 0, sic 1 for mesh 1
        mesh_num = 2
        esl_bw_factor = 1 / mesh_num
        mesh_id = sic_id
        in_tensor = action_tensor_to_operand(self.inputs[0].tensor[0])
        out_tensor = action_tensor_to_operand(self.outputs[0].tensor[0])
        rank_cnt = 8
        element_cnt = list_product(in_tensor.dim)
        element_cnt_per_mesh = element_cnt // mesh_num
        element_cnt_per_rank = element_cnt_per_mesh // rank_cnt
        rank = self.config.gcu_id
        peer_ranks = get_peer_ranks(rank_cnt, rank)
        dtype = self.get_dtype()
        bpe = dtype.get_bpe()
        addr_l3_base = 0x51000000000 + (mesh_id + 1) * 0x00010000000
        flag_l3_base = 0x52000000000 + (mesh_id + 1) * 0x00010000000
        data_l3_base = in_tensor.addr + mesh_id * element_cnt_per_mesh * bpe
        total_latency = 0
        stat_ref = 0

        prims = SimpleProtoPrimitives(
            case_id=self.case_id,
            channel_id=mesh_id,
            recvPeers=peer_ranks,
            sendPeers=peer_ranks,
            sendBuff=in_tensor.addr,
            recvBuff=out_tensor.addr,
            rank=rank,
            ref=self.ref,
        )
        yield from prims.meshAllGather(mesh_id * element_cnt_per_mesh * bpe, element_cnt_per_mesh * bpe, stat_ref)

    def _basic_stat_info(self):
        pass
