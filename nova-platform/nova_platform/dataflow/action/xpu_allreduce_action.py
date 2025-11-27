from dataclasses import dataclass, field
from typing import Generator, List


from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
)
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.config import TOPO
from nova_platform.executor.nova_platform_barrier import BarrierManager

from nova_platform.benchmark.op_base import Operand, list_product

import logging

from nova_platform.executor.nova_platform_event import EventManager

logger = logging.getLogger(__name__)


@dataclass
class InstructionInfo:
    dtype: DType = DType.FP16
    grid_dim: List[int] = field(default_factory=list)
    block_dim: List[int] = field(default_factory=list)
    thread_dim: List[int] = field(default_factory=list)
    subthread_dim: List[int] = field(default_factory=list)


@dataclass
class CoreCost(BaseCoreStat):
    instruction_info: InstructionInfo = field(default_factory=InstructionInfo)
    sip_workloads: list = field(default_factory=list)


def action_tensor_to_operand(tensor):
    return Operand(
        dim=tuple(tensor.dims),
        addr=tensor.addr,
        bpe=tensor.bpe,
        dim_offset=tuple(tensor.offsets),
        dim_stride=tuple(tensor.stride_dims),
    )


def get_peer_ranks(rank_num, rank):
    peer_rank = list(range(0, rank_num))
    peer_rank = peer_rank[rank + 1:] + peer_rank[0:rank]
    return peer_rank


em = EventManager()


@dataclass
class XPUAllReduceAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_kernel_idx(self):
        return self.data[0], self.data[1]

    def _iter_tensor_addr(self, tensor: Operand, rw) -> Generator[DataflowActionMemoryAccess, None, None]:
        mem_access = tensor.get_contiguous_mem_accesses()
        for addr, size in mem_access:
            yield DataflowActionMemoryAccess(addr, size, rw)

    def _iter_addr(self, tensor: Operand, rw) -> DataflowActionMemoryAccess:
        base_addr = tensor.get_phy_addr_by_offset(tensor.dim_offset)
        data_size = list_product(tuple(tensor.dim)) * tensor.bpe
        return DataflowActionMemoryAccess(base_addr, data_size, rw)

    def get_sync_barriers(self, mesh_id, rank_cnt):
        bm = BarrierManager()
        barrier_id = f"{self.case_id}_{mesh_id}"
        return bm.get_barrier(barrier_id + "_rs", rank_cnt)

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        if self.topo in [TOPO.FULLMESH8]:
            yield from self.fullmesh_allreduce()
        elif self.topo in [TOPO.SUPERNODE32]:
            yield from self.get_memory_stat_supernode()
        elif self.topo in [TOPO.SUPERNODE16]:
            yield from self.test()
        else:
            raise NotImplementedError

    def test(self):
        data_size = 10*2**20
        addr_l3_base = 0x51000000000
        src_gen = self._iter_access_gen(
            [DataflowActionMemoryAccess(
                addr_l3_base,
                data_size, "r")]
        )
        next(src_gen)
        tar_gen = self._iter_access_gen(
            [DataflowActionMemoryAccess(
                addr_l3_base,
                data_size, "w")]
        )
        next(tar_gen)
        if self.config.gcu_id == 0 and self.get_die_id() == 0 and self.get_cluster_id() == 0 and self.get_local_engine_id() == 0:
            mem_stat = DataflowActionMemoryStat(
                total_count=data_size,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=1,
                relative_ts=0,
                memory_access_list=src_gen.send(data_size),
                remote_target_mem_access_list=tar_gen.send(
                    data_size),
                bw_factor=1,
                name=f"RS l3->esl->gcu{1}_l3",
            )
            yield mem_stat
            em.get_event("test").set(self.ref+mem_stat.latency)
            total_latency = mem_stat.latency
        elif self.config.gcu_id == 1 and self.get_die_id() == 0 and self.get_cluster_id() == 0 and self.get_local_engine_id() == 0:
            max_time = yield from em.get_event("test").wait(self.ref)
            relative_ts = max_time-self.ref
            mem_stat = DataflowActionMemoryStat(
                total_count=data_size,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=0,
                relative_ts=relative_ts,
                memory_access_list=src_gen.send(data_size),
                remote_target_mem_access_list=tar_gen.send(
                    data_size),
                bw_factor=1,
                name=f"RS l3->esl->gcu{0}_l3",
            )
            yield mem_stat
            total_latency = max_time + mem_stat.latency
        else:
            total_latency = 1e-9

        self.core_cost.latency = total_latency

    def get_memory_stat_supernode(self) -> Generator[DataflowActionMemoryStat, None, None]:
        data_size = 10*2**20
        addr_l3_base = 0x51000000000

        src_gen = self._iter_access_gen(
            [DataflowActionMemoryAccess(
                addr_l3_base,
                data_size, "r")]
        )
        next(src_gen)
        tar_gen = self._iter_access_gen(
            [DataflowActionMemoryAccess(
                addr_l3_base,
                data_size, "w")]
        )
        next(tar_gen)
        total_latency = 1e-9
        if self.config.gcu_id in [0, 1]:
            mem_stat = DataflowActionMemoryStat(
                total_count=data_size,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=2,
                relative_ts=0,
                memory_access_list=src_gen.send(data_size),
                remote_target_mem_access_list=tar_gen.send(
                    data_size),
                bw_factor=1,
                name=f"RS l3->esl->gcu{2}_l3",
            )
            yield mem_stat
            total_latency = mem_stat.latency

        self.core_cost.latency = total_latency
        return

    def ring_allreduce(self) -> Generator[DataflowActionMemoryStat, None, None]:
        sic_id, sip_id = self.get_kernel_idx()
        if sic_id not in [0, 1] or sip_id != 0:
            return
        gcu_cnt = 16
        ring_num = 8
        slice_num = 1
        ring_group = sic_id
        dtype = self.get_dtype()
        bpe = dtype.get_bpe()
        in_tensor = action_tensor_to_operand(self.inputs[0].tensor[0])
        element_cnt = list_product(in_tensor.dim)
        element_per_slice = element_cnt / gcu_cnt / ring_num / ring_group / slice_num
        size_per_slice = element_per_slice * bpe
        l3_base_addr = 0x51000000000
        BUFFER_SIZE = 0x400000
        esl_bw_factor = 1
        rank = self.config.gcu_id

        def get_sync_barriers(self, ring_id, pre_rank, post_rank):
            bm = BarrierManager()
            barrier_id = f"{self.case_id}_ring{ring_id}_rank{pre_rank}_rank{post_rank}"
            return bm.get_barrier(barrier_id, 2)

        def get_peer_ranks(self, rank, rank_num, ring_id, ring_num):
            pre_rank = (rank - 1 + rank_num) % rank_num
            post_rank = (rank + 1) % rank_num
            return pre_rank, post_rank

        stat_ref = 0
        total_latency = 0
        for ring_id in range(1, ring_num + 1):
            for slice_id in range(1, slice_num + 1):
                temp_buffer_addr = l3_base_addr + ring_id * BUFFER_SIZE
                inout_buffer_addr = l3_base_addr + ring_num * ring_id * BUFFER_SIZE

                temp_buffer_gen = self._iter_access_gen(
                    [DataflowActionMemoryAccess(temp_buffer_addr, size_per_slice, "w")]
                )

                pre_rank, post_rank = get_peer_ranks(self, rank, gcu_cnt, ring_id, ring_num)
                receive_barrier = get_sync_barriers(self, ring_id, pre_rank, rank)
                send_barrier = get_sync_barriers(self, ring_id, rank, post_rank)

                # reduce scatter
                rs_ref = stat_ref
                for step in range(1, gcu_cnt):
                    rs_send = DataflowActionMemoryStat(
                        total_count=size_per_slice,
                        master=DataflowActionType.ESL,
                        src=AddrDomain.L3,
                        dst=AddrDomain.L3_REMOTE,
                        rw="w",
                        gcu_id=post_rank,  # self.get_die_id(),  # 0..8
                        relative_ts=rs_ref,
                        memory_access_list=temp_buffer_gen.send(size_per_slice),
                        remote_target_mem_access_list=temp_buffer_gen.send(size_per_slice),
                        bw_factor=esl_bw_factor,
                        name=f"RS 0 l3->esl->->gcu{rank}_l3",
                    )
                    yield rs_send
                    rs_send_latency = rs_ref + rs_send.latency
                    yield from send_barrier.wait(rs_send_latency)
                    yield from receive_barrier.wait(rs_ref)

    def fullmesh_allreduce(self) -> Generator[DataflowActionMemoryStat, None, None]:
        sic_id, sip_id = self.get_kernel_idx()
        if sic_id not in [0, 1] or sip_id != 0:
            return
        # sic 0 for mesh 0, sic 1 for mesh 1
        mesh_num = 2
        esl_bw_factor = 1 / mesh_num
        mesh_id = sic_id
        in_tensor = action_tensor_to_operand(self.inputs[0].tensor[0])
        rank_cnt = 8
        element_cnt = list_product(in_tensor.dim)
        element_cnt_per_mesh = element_cnt // 2
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

        rs_barrier = self.get_sync_barriers(mesh_id, rank_cnt)
        # address broadcast

        def write_flag(flag_ref):
            total_latency = 0
            flag_size = 128  # bytes
            for idx, peer in enumerate(peer_ranks):
                st_flag_gen = self._iter_access_gen(
                    [DataflowActionMemoryAccess(
                        flag_l3_base + rank * flag_size, flag_size, "w")]
                )
                next(st_flag_gen)
                ld_flag_gen = self._iter_access_gen(
                    [DataflowActionMemoryAccess(
                        flag_l3_base + rank * flag_size, flag_size, "r")]
                )
                next(ld_flag_gen)
                remote_flag_gen = self._iter_access_gen(
                    [DataflowActionMemoryAccess(
                        flag_l3_base + peer * flag_size, flag_size, "w")]
                )
                next(remote_flag_gen)

                flag_st_l3 = DataflowActionMemoryStat(
                    total_count=flag_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=AddrDomain.L3,
                    rw="w",
                    relative_ts=flag_ref,
                    memory_access_list=st_flag_gen.send(flag_size),
                    name=f"flag l0->l3",
                )
                yield flag_st_l3

                flag_broadcast_ref = flag_ref + flag_st_l3.latency
                flag_broadcast = DataflowActionMemoryStat(
                    total_count=flag_size,
                    master=DataflowActionType.ESL,
                    src=AddrDomain.L3,
                    dst=AddrDomain.L3_REMOTE,
                    rw="w",
                    src_gcu_id=self.config.gcu_id,
                    tar_gcu_id=peer,  # self.get_die_id(),  # 0..8
                    relative_ts=flag_broadcast_ref,
                    memory_access_list=ld_flag_gen.send(flag_size),
                    remote_target_mem_access_list=remote_flag_gen.send(
                        flag_size),
                    bw_factor=esl_bw_factor,
                    name=f"flag l3->esl->->gcu{peer}_l3",
                )
                yield flag_broadcast
                total_latency = max(
                    total_latency, flag_broadcast_ref + flag_broadcast.latency)
            return total_latency

        for idx, peer in enumerate(peer_ranks):
            st_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(addr_l3_base + rank * 128, 128, "w")])
            next(st_addr_gen)
            ld_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(addr_l3_base + rank * 128, 128, "r")])
            next(ld_addr_gen)
            peer_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(addr_l3_base + peer * 128, 128, "w")])
            next(peer_addr_gen)

            addr_st_l3 = DataflowActionMemoryStat(
                total_count=128,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=AddrDomain.L3,
                rw="w",
                relative_ts=stat_ref,
                memory_access_list=st_addr_gen.send(128),
                name=f"addr l0->l3",
            )
            yield addr_st_l3

            addr_broadcast_ref = stat_ref + addr_st_l3.latency
            addr_broadcast = DataflowActionMemoryStat(
                total_count=128,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=peer,
                relative_ts=addr_broadcast_ref,
                memory_access_list=ld_addr_gen.send(128),
                remote_target_mem_access_list=peer_addr_gen.send(128),
                bw_factor=esl_bw_factor,
                name=f"addr l3->esl->gcu{peer}_l3",
            )
            yield addr_broadcast
            total_latency = max(
                total_latency, addr_broadcast_ref + addr_broadcast.latency)

        # write flags
        flag_ref = total_latency
        total_latency = yield from write_flag(flag_ref)
        # reduce scatter
        reduce_scatter_ref = total_latency
        data_size = element_cnt_per_rank * bpe
        for idx, peer in enumerate(peer_ranks):
            rs_l3_base = data_l3_base + peer * data_size
            rs_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(rs_l3_base, data_size, "r")])
            next(rs_addr_gen)
            remote_rs_l3_base = data_l3_base + rank * data_size
            remote_rs_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(remote_rs_l3_base, data_size, "w")])
            next(remote_rs_addr_gen)
            reduce_scatter = DataflowActionMemoryStat(
                total_count=data_size,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=peer,
                relative_ts=reduce_scatter_ref,
                memory_access_list=rs_addr_gen.send(data_size),
                remote_target_mem_access_list=remote_rs_addr_gen.send(
                    data_size),
                bw_factor=esl_bw_factor,
                name=f"RS l3->esl->gcu{peer}_l3",
            )
            yield reduce_scatter
            total_latency = max(
                total_latency, reduce_scatter_ref + reduce_scatter.latency)

        # write flags
        flag_ref = total_latency
        total_latency = yield from write_flag(flag_ref)
        # all gather
        total_latency = yield from rs_barrier.wait(self.ref + total_latency)
        total_latency -= self.ref
        all_gather_ref = total_latency
        data_size = element_cnt_per_rank * bpe
        for idx, peer in enumerate(peer_ranks):
            al_l3_base = data_l3_base + rank * data_size
            al_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(al_l3_base, data_size, "r")])
            next(al_addr_gen)
            remote_al_l3_base = data_l3_base + peer * data_size
            remote_al_addr_gen = self._iter_access_gen(
                [DataflowActionMemoryAccess(remote_al_l3_base, data_size, "w")])
            next(remote_al_addr_gen)
            all_gather = DataflowActionMemoryStat(
                total_count=data_size,
                master=DataflowActionType.ESL,
                src=AddrDomain.L3,
                dst=AddrDomain.L3_REMOTE,
                rw="w",
                src_gcu_id=self.config.gcu_id,
                tar_gcu_id=peer,
                relative_ts=all_gather_ref,
                memory_access_list=al_addr_gen.send(data_size),
                remote_target_mem_access_list=remote_al_addr_gen.send(
                    data_size),
                bw_factor=esl_bw_factor,
                name=f"AG l3->esl->gcu{peer}_l3",
            )
            yield all_gather
            total_latency = max(
                total_latency, all_gather_ref + all_gather.latency)

        # write flags
        flag_ref = total_latency
        total_latency = yield from write_flag(flag_ref)

        self.core_cost.latency = total_latency

    def _basic_stat_info(self):
        self.core_cost.tensor_dtype = self.get_dtype()
