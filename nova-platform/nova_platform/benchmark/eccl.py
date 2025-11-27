from dataclasses import dataclass, field
from typing import Generator, List

from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
)
from nova_platform.executor.nova_platform_event import EventManager

em = EventManager()


@dataclass
class EcclPrimitives:
    def _iter_access_gen(self, mem_acc_list: List[DataflowActionMemoryAccess]):
        counter = 0
        batch: List[DataflowActionMemoryAccess] = []
        fetch_size = yield batch
        # history = [] # for debug
        # history_raw = [] # for debug
        while True:
            for acc in mem_acc_list:
                # history_raw.append(acc)
                batch.append(acc)
                counter += acc.size
                while True:
                    last_acc = batch.pop()

                    if counter >= fetch_size:
                        last_size_right = counter - fetch_size
                        last_size_left = last_acc.size - last_size_right
                        batch.append(
                            DataflowActionMemoryAccess(
                                base_addr=last_acc.base_addr, size=last_size_left, rw=last_acc.rw
                            )
                        )

                        new_base = last_acc.base_addr + last_size_left
                        new_size = last_acc.size - last_size_left
                        counter = fetch_size
                        # history.append(batch)
                        fetch_size = yield batch
                        if new_size > 0:
                            batch = [DataflowActionMemoryAccess(base_addr=new_base, size=new_size, rw=last_acc.rw)]
                            counter = new_size
                        else:
                            batch = []
                            counter = 0
                            break
                    else:
                        batch.append(last_acc)
                        break
            if batch:
                # history.append(batch)
                yield batch


@dataclass
class SimpleProtoPrimitives(EcclPrimitives):
    case_id: str = ""
    channel_id: str = ""
    recvPeers: List[int] = field(default_factory=list)
    sendPeers: List[int] = field(default_factory=list)
    sendBuff: int = 0
    recvBuff: int = 0
    rank: int = 0
    ref: int = 0
    useDirectMode: bool = False
    slice_num: int = 1
    esl_bw_factor: float = 1.0

    def _get_fifo_addr(self, recv_id):
        return 0x58000000000 + 0x01000000000 * (recv_id % 2)

    def _get_flag_event(self, from_rank: int, to_rank: int, trans_id: int):
        return em.get_event(f"{self.case_id}_{self.channel_id}_{from_rank}_{to_rank}_{trans_id}")

    def directRecvReduceCopySend(self, inpIx: int, outIx: int, nelem: int, postOp: bool = False):
        for recvPeerId in self.recvPeers:
            for sendPeerId in self.sendPeers:
                pass

    def _esl_master_send(self, peer: int, nelem: int, relative_ts: int, src_addr_gen, dst_addr_gen, send_id: int):
        master_send = DataflowActionMemoryStat(
            total_count=nelem,
            master=DataflowActionType.ESL,
            src=AddrDomain.L3,
            dst=AddrDomain.L3_REMOTE,
            rw="w",
            src_gcu_id=self.rank,
            tar_gcu_id=peer,
            relative_ts=relative_ts,
            memory_access_list=src_addr_gen.send(nelem),
            remote_target_mem_access_list=dst_addr_gen.send(nelem),
            bw_factor=self.esl_bw_factor,
            name=f"{self.channel_id}:l3->esl->gcu{peer}_l3",
        )
        yield master_send
        self._get_flag_event(self.rank, peer, send_id).set(self.ref + relative_ts + master_send.latency)
        print(f"!!!!send {self.case_id}_{self.channel_id}_{self.rank}_{peer}_{send_id}")
        return master_send.leading_latency, master_send.latency

    def directRecvCopySend(self, outIx: int, nelem: int, relative_ts: int, recv_id: int, send_id: int):
        latency = 0
        for recv_peer in self.recvPeers:
            for send_peer in self.sendPeers:
                recv_ts = yield from self._get_flag_event(recv_peer, self.rank, recv_id).wait(self.ref + relative_ts)
                send_ref = recv_ts - self.ref

                src_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(self.sendBuff, nelem, "r")])
                next(src_addr_gen)
                dst_addr = self.recvBuff + outIx if self.useDirectMode else self._get_fifo_addr(send_id)
                dst_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(dst_addr, nelem, "w")])
                next(dst_addr_gen)
                if not self.useDirectMode:
                    yield from self.memcpyGtoG(self.recvBuff, outIx, self._get_fifo_addr(recv_id), 0, nelem, send_ref)
                send_leading, send_latency = yield from self._esl_master_send(
                    send_peer, nelem, send_ref, src_addr_gen, dst_addr_gen, send_id
                )
                latency = max(latency, send_ref + send_latency)
        return latency

    def directRecv(self, outIx: int, nelem: int, relative_ts: int, recv_id: int):
        total_latency = 0
        for recv_peer in self.recvPeers:
            recv_ts = yield from self._get_flag_event(recv_peer, self.rank, recv_id).wait(self.ref + relative_ts)
            latency = recv_ts - self.ref
            if not self.useDirectMode:
                latency = yield from self.memcpyGtoG(
                    self.recvBuff, outIx, self._get_fifo_addr(recv_id), 0, nelem, latency
                )
            total_latency = max(latency, total_latency)
        return latency

    def directSend(self, inpIx: int, nelem: int, relative_ts: int, send_id: int):
        latency = 0
        for idx, peer in enumerate(self.sendPeers):
            src_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(self.sendBuff + inpIx, nelem, "r")])
            next(src_addr_gen)
            dst_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(self.recvBuff, nelem, "w")])
            next(dst_addr_gen)
            send_leading, send_latency = yield from self._esl_master_send(
                peer, nelem, relative_ts, src_addr_gen, dst_addr_gen, send_id
            )
            latency = max(latency, relative_ts + send_latency)
        return latency

    def directCopySend(self, inpIx: int, outIx: int, nelem: int, relative_ts: int, send_id: int):
        latency = 0
        for idx, peer in enumerate(self.sendPeers):
            src_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(self.sendBuff + inpIx, nelem, "r")])
            next(src_addr_gen)
            dst_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(self.recvBuff + outIx, nelem, "w")])
            next(dst_addr_gen)
            send_leading, send_latency = yield from self._esl_master_send(
                peer, nelem, relative_ts, src_addr_gen, dst_addr_gen, send_id
            )
            copy_latency = yield from self.memcpyGtoG(self.recvBuff, outIx, self.sendBuff, inpIx, nelem, relative_ts)
            latency = max(latency, relative_ts + send_latency)
        return latency

    def recvCopyAsync(self, offset: int, nelem: int, relative_ts: int):
        latency = 0
        for idx, peer in enumerate(self.recvPeers):
            recv_ts = yield from self._get_flag_event(peer, self.rank, 0).wait(self.ref + relative_ts)
            copy_latency = yield from self.memcpyGtoG(self.recvBuff, 0, 0, 0, nelem, recv_ts - self.ref)
            latency = max(latency, copy_latency)
        return latency

    def memcpyGtoG(self, dst: int, dstOffs: int, src: int, srcOffs: int, nelem: int, relative_ts: int):
        if self.useDirectMode:
            return relative_ts
        srcAddr = src + srcOffs
        dstAddr = dst + dstOffs
        if dstAddr == srcAddr:
            return relative_ts
        src_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(srcAddr, nelem, "r")])
        next(src_addr_gen)
        dst_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(dstAddr, nelem, "w")])
        next(dst_addr_gen)
        st_l3_l3 = DataflowActionMemoryStat(
            total_count=nelem,
            master=DataflowActionType.CDTE,
            src=AddrDomain.L3,
            dst=AddrDomain.L3,
            rw="w",
            relative_ts=relative_ts,
            memory_access_list=src_addr_gen.send(nelem),
            remote_target_mem_access_list=dst_addr_gen.send(nelem),
            name=f"{self.channel_id}:l3->l3",
        )
        yield st_l3_l3
        return relative_ts + st_l3_l3.latency

    def broadCast(self, src_addr: int, src_offset: int, nelem: int, relative_ts: int):
        total_latency = 0
        for idx, peer in enumerate(self.sendPeers):
            src_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(src_addr + src_offset, nelem, "r")])
            next(src_addr_gen)
            dst_addr_gen = self._iter_access_gen([DataflowActionMemoryAccess(src_addr + src_offset, nelem, "w")])
            next(dst_addr_gen)
            broadcast_ref = relative_ts
            leading, latency = yield from self._esl_master_send(peer, nelem, broadcast_ref, src_addr_gen, dst_addr_gen)
            total_latency = max(total_latency, broadcast_ref + latency)
        return latency

    def meshAllGather(self, offset: int, nelem: int, relative_ts: int):
        latency = yield from self.broadCast(self.sendBuff, offset, nelem, relative_ts)
        latency = yield from self.recvCopyAsync(offset, nelem, relative_ts)
        return latency
