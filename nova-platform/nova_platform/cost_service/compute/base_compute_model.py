from collections import defaultdict
import threading
from typing import _GenericAlias, DefaultDict, Dict, Generator, Generic, List, Tuple, TypeVar
from nova_platform.base_model import MAX_TIME, BaseActionStat, BaseESLSwitch, DType, DataflowActionComputeStat, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType, DataflowOpType, PostStat


from dataclasses import asdict, dataclass, field, fields

from nova_platform.base_model import BaseFrame
from nova_platform.config import BossaNovaConfig
from nova_platform.cost_service.power.base_power_model import PowerFrame, PowerL3Frame, PowerStat
from nova_platform.perfetto_protobuf._tgen import TraceGenerator
from nova_platform.utils.base_utils import BaseDataclass
from nova_platform.utils.config_utils import BaseEnum
import logging
logger = logging.getLogger(__file__)


class BWMode(str, BaseEnum):
    PORT = "port"
    BANDWIDTH = "bandwidth"


@dataclass
class BaseCoreStat(BaseActionStat):
    ld_l3_l3: int = 0
    st_l3_l3: int = 0
    tensor_macs: Dict[DType, int] = field(
        default_factory=dict)
    vector_ops: Dict[DType, int] = field(
        default_factory=dict)
    sfu_ops: int = 0
    scalar_cycle: int = 0
    ld_l0_local:  int = 0
    st_l0_local:  int = 0
    ld_l0_shared:  int = 0
    st_l0_shared:  int = 0
    ld_l0_l3: int = 0
    st_l0_l3: int = 0
    ld_local_l3: int = 0
    st_local_l3: int = 0
    ld_shared_l3: int = 0
    st_shared_l3: int = 0
    st_l3_l3_remote: int = 0
    st_esl_l3: int = 0

    def __iadd__(self, stat: DataflowActionComputeStat | DataflowActionMemoryStat):
        if isinstance(stat, DataflowActionComputeStat):
            for dt, v in stat.compute_1d_ops.items():
                if dt not in self.vector_ops:
                    self.vector_ops[dt] = 0.0
                self.vector_ops[dt] += v
            for dt, v in stat.compute_2d_ops.items():
                if dt not in self.tensor_macs:
                    self.tensor_macs[dt] = 0.0
                self.tensor_macs[dt] += v/2
            self.sfu_ops += stat.compute_msf_ops
            self.scalar_cycle += stat.compute_scalar_cycle
        elif isinstance(stat, DataflowActionMemoryStat):
            src, dst = stat.src.name.lower(), stat.dst.name.lower()
            if stat.rw == 'r':
                tar = f"ld_{src}_{dst}"
            else:
                tar = f"st_{src}_{dst}"

            v = getattr(self, tar)
            setattr(self, tar, v+stat.total_count)
        elif isinstance(stat, BaseCoreStat):
            for f in fields(BaseCoreStat):
                if f.type in (int, float):
                    v = getattr(self, f.name)+getattr(stat, f.name)
                    setattr(self, f.name, v)
                elif isinstance(f.type, _GenericAlias) and f.type.__origin__ is dict:
                    lhs_dict = getattr(self, f.name)
                    rhs_dict = getattr(stat, f.name)
                    for k, v in rhs_dict.items():
                        if k not in lhs_dict:
                            lhs_dict[k] = 0
                        lhs_dict[k] += rhs_dict[k]
                elif f.name in ('power_stat', 'name'):
                    pass
                else:
                    raise NotImplemented
        else:
            raise NotImplemented

        return self

    def to_dict(self):
        res = asdict(self)
        res["tensor_macs"] = sum([v for v in self.tensor_macs.values()])
        res["vector_ops"] = sum([v for v in self.vector_ops.values()])
        return res


@dataclass
class BWCost:
    start: float
    end: float
    action_id: int


@dataclass
class BWFrame(BaseFrame):
    allocated_bw: int = field(default=0)  # Byte/cycle


IFrame = TypeVar('IFrame', bound=BaseFrame)


@dataclass
class Timeline(BaseDataclass, Generic[IFrame]):
    data: List[IFrame] = field(default_factory=list)

    def get_frame(self, ref) -> Tuple[int, IFrame]:
        assert 0 <= ref < MAX_TIME, "ref time out of range [0,%d)" % (MAX_TIME)
        data_len = len(self.data)
        for i in reversed(range(data_len)):
            if self.data[i].begin <= ref:
                # ..., Frame left, ref, Frame right, ...
                return i, self.data[i]

    def insert(self, new_frame: IFrame):
        s_idx, s_frame = self.get_frame(new_frame.begin)
        # frame_cls: IFrame = new_frame.__class__
        if s_frame.begin < new_frame.begin:
            # split s_frame
            # rhs_frame = frame_cls(**s_frame.__dict__)
            rhs_frame = s_frame.clone()
            rhs_frame.begin = new_frame.begin
            self.data.insert(s_idx+1, rhs_frame)
            s_frame.end = new_frame.begin

            s_idx += 1
            s_frame = rhs_frame

        # now s_frame.begin==new_frame.begin

        if s_frame.end > new_frame.end:
            # split s_frame
            # rhs_frame = frame_cls(**s_frame.__dict__)
            rhs_frame = s_frame.clone()
            rhs_frame.begin = new_frame.end
            self.data.insert(s_idx+1, rhs_frame)
            s_frame.end = new_frame.end
            s_frame.incr(new_frame)
        elif s_frame.end == new_frame.end:
            s_frame.incr(new_frame)
        else:  # s_frame.end < new_frame.end
            # s_frame.incr(new_frame)
            # next_frame: IFrame = new_frame.clone()
            # next_frame.begin = s_frame.end
            # self.insert(next_frame)

            while s_frame.end < new_frame.end:
                s_frame.incr(new_frame)
                s_idx += 1
                s_frame = self.data[s_idx]

            next_frame: IFrame = new_frame.clone()
            next_frame.begin = s_frame.begin
            self.insert(next_frame)


@dataclass
class BWResource(BaseDataclass):
    name: str
    mode: BWMode
    max_bw: int
    timeline: Timeline[BWFrame] = field(
        default_factory=lambda: Timeline([BWFrame(0, MAX_TIME, 0)]))
    acc_read: int = 0
    acc_write: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_frame(self, ref):
        assert 0 <= ref < MAX_TIME, "ref time out of range [0,%d)" % (MAX_TIME)
        data_len = len(self.timeline.data)
        for i in reversed(range(data_len)):
            if self.timeline.data[i].begin <= ref:
                # ..., Frame left, ref, Frame right, ...

                if self.timeline.data[i].allocated_bw == self.max_bw:
                    # no available bw in this frame
                    start_i = i
                    end_i = data_len - 1
                    for j in range(i+1, data_len):
                        if self.timeline.data[j].allocated_bw < self.max_bw:
                            end_i = j
                            break
                    # compact timeline to avoid too deep recusion
                    self.timeline.data[i].end = self.timeline.data[end_i-1].end
                    del self.timeline.data[start_i+1:end_i]
                    # i = i+1  # use next allocatable frame

                return i, self.timeline.data[i]

    def get_last_valid_frame(self):
        frame_list = self.timeline.data
        if len(frame_list) == 0:
            return None
        elif frame_list[-1].end != MAX_TIME:
            return frame_list[-1]
        elif len(frame_list) > 1:
            return frame_list[-2]
        else:
            return None


@dataclass
class BWResourceContext:
    config: BossaNovaConfig
    esl_switch: BaseESLSwitch
    sic_io_dict: Dict = field(default_factory=dict)
    l3_dict: Dict = field(default_factory=dict)
    d2d_dict: Dict = field(default_factory=dict)
    esl_dict: Dict = field(default_factory=dict)

    def __post_init__(self):
        # for libra-like arch
        # TODO: extend when introducing new arch

        for die_id in range(self.config.inst_num.NUM_OF_DIE):
            # sic io
            for cluster_id in range(self.config.inst_num.NUM_OF_CLUSTER):
                sic_io_config = self.config.bw.sic_io
                sic_io_r = BWResource(
                    f"die{die_id}_sic_io{cluster_id}_r", BWMode.PORT, sic_io_config.noc.bw)
                sic_io_w = BWResource(
                    f"die{die_id}_sic_io{cluster_id}_w", BWMode.PORT, sic_io_config.noc.bw/sic_io_config.noc.rw_ratio)
                self.sic_io_dict[(die_id, cluster_id, 'r')] = sic_io_r
                self.sic_io_dict[(die_id, cluster_id, 'w')] = sic_io_w

            # l3
            l3_rw = BWResource(f"die{die_id}_l3",
                               BWMode.BANDWIDTH, self.config.bw.mc.l3.bw)
            self.l3_dict[die_id] = l3_rw

            # d2d
            if self.config.inst_num.NUM_OF_DIE == 2:
                # define Tx for each bw
                die0_d2d_bw_resource = BWResource(f"die0_d2d_tx",
                                                  BWMode.BANDWIDTH, self.config.bw.noc.llc_far.bw_per_second/self.config.freq.NOC/1e9)
                die1_d2d_bw_resource = BWResource(f"die1_d2d_tx",
                                                  BWMode.BANDWIDTH, self.config.bw.noc.llc_far.bw_per_second/self.config.freq.NOC/1e9)
                # Die 0 ST: transmit data from die0 to die1
                self.d2d_dict[(0, 'w')] = die0_d2d_bw_resource
                # Die 0 LD: transmit data from die1 to die0
                self.d2d_dict[(0, 'r')] = die1_d2d_bw_resource
                # Die 1 ST: transmit data from die1 to die0
                self.d2d_dict[(1, 'w')] = die1_d2d_bw_resource
                # Die 0 LD: transmit data from die0 to die1
                self.d2d_dict[(1, 'r')] = die0_d2d_bw_resource

        # esl
        self.esl_switch.build_bw_resource(self)

    def get_unique_bw_resource_list(self) -> Generator[Tuple[str, str, int, int, int, BWResource], None, None]:
        # TODO: need overwrite when introducing new arch
        for die_id in range(self.config.inst_num.NUM_OF_DIE):
            for cluster_id in range(self.config.inst_num.NUM_OF_CLUSTER):
                yield ("sic_io", "r", die_id, cluster_id, None, self.sic_io_dict[(die_id, cluster_id, 'r')])
                yield ("sic_io", "w", die_id, cluster_id, None, self.sic_io_dict[(die_id, cluster_id, 'w')])
            yield ("l3", "rw", die_id, None, None, self.l3_dict[(die_id)])
            if self.config.inst_num.NUM_OF_DIE > 1:
                yield ("d2d_tx", "rw", die_id, None, None, self.d2d_dict[(die_id, 'w')])
        # get esl unique bw_resource
        yield from self.esl_switch.get_unique_bw_resource(self)

    def get_bw_resource(self, src, dst, die_id, cluster_id, mem_stat, rw) -> BWResource | None:
        def _sic_io():
            key = (die_id, cluster_id, rw)
            return self.sic_io_dict[key]

        def _l3():
            key = (die_id)
            return self.l3_dict[key]

        def _d2d():
            key = (die_id, rw)
            return self.d2d_dict[key]

        def _esl():
            return self.esl_switch.get_bw_resource(self.config.gcu_id, mem_stat.src_gcu_id, mem_stat.tar_gcu_id, rw)

        res_dict = {
            ("sic_io", "noc"): _sic_io,
            ("mc", "l3"): _l3,
            ('esl', 'noc'): _esl,
            ("noc", "llc_far"): _d2d,
        }

        if (src, dst) in res_dict:
            return res_dict[(src, dst)]()
        else:
            return None


@dataclass
class CostBook(BaseDataclass):
    cache_stat_dict: Dict[str, any] = field(default_factory=dict)
    power_stat: PowerStat = None
    core_stat: BaseCoreStat = None
    latency: float = 0
    r_datasize: int = 0
    w_datasize: int = 0


@dataclass
class EDCFrame(BaseFrame):
    current: float = 0
    # triggered: bool = False

    def incr(self, frame: 'EDCFrame'):
        self.current += frame.current
        # self.triggered |= frame.triggered


@dataclass
class PowerContext:
    # (die_id,cluster_id,sip_id)
    sip_power_timeline_dict: Dict[Tuple[int, int, int],
                                  Timeline[PowerFrame]] = field(default_factory=lambda: {})
    l1_power_timeline_dict: Dict[Tuple[int, int, int],
                                 Timeline[PowerFrame]] = field(default_factory=lambda: {})
    dtu_power_timeline:  Timeline[PowerFrame] = field(
        default_factory=lambda: Timeline([PowerFrame()]))

    dtu_edc: Timeline[EDCFrame] = field(
        default_factory=lambda: Timeline([EDCFrame()]))

    soc_edc: Timeline[EDCFrame] = field(
        default_factory=lambda: Timeline([EDCFrame()]))

    l3_power: List[PowerL3Frame] = field(default_factory=list)


@dataclass
class BossaNovaContext():
    initial_ref: float = 0
    cost_dict: DefaultDict[int, CostBook] = field(
        default_factory=lambda: defaultdict(CostBook))

    bw_resource_context: BWResourceContext = None

    power_context: PowerContext = None

    post_stat: PostStat = None
    tgen: TraceGenerator = None

    def get_cluster_tgen(self, die_id: int, cid: int):
        return self.tgen.get_cluster_tgen(die_id, cid)

    def get_cost_book(self, action: 'DataflowAction') -> CostBook:
        return self.cost_dict[action.get_action_id()]


@dataclass
class DataflowAction():
    def __post_init__(self):
        self.ref = 0

    def get_die_id(self) -> int:
        raise NotImplemented()

    def get_action_id(self) -> int:
        raise NotImplemented()

    def get_engine_id(self) -> int:
        raise NotImplemented()

    def get_engine_sub_id(self) -> int:
        raise NotImplemented()

    def get_cluster_id(self) -> int:
        raise NotImplemented()

    def get_local_engine_id(self) -> int:
        raise NotImplemented()

    def get_port_id(self) -> int:
        raise NotImplemented()

    def get_memory_access(self) -> Generator[DataflowActionMemoryAccess, None, None]:
        # List[ addr, size, w/r ]
        raise NotImplemented()

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        # List[ count, src_domain, dst_domain, w/r ]
        raise NotImplemented()

    def get_child_ids(self):
        raise NotImplemented()

    def get_parent_ids(self):
        raise NotImplemented()

    def get_action_type(self) -> DataflowActionType:
        raise NotImplemented()

    def get_optype(self) -> DataflowOpType:
        raise NotImplemented()

    def get_dtype(self) -> DType:
        raise NotImplemented()

    def compute(self, context: 'BossaNovaContext') -> Generator[DataflowActionMemoryStat | DataflowActionComputeStat | None, None, BaseCoreStat]:
        raise NotImplemented()

    def get_core_stat(self) -> BaseActionStat:
        raise NotImplemented()


class BaseCostService():
    def __init__(self, config: BossaNovaConfig):
        self.config = config

    def process(self, action: DataflowAction, context: BossaNovaContext, ref: float) -> Generator[bool, None, None]:
        raise NotImplementedError()

    def post_process(self, context: BossaNovaContext):
        pass

    def post_stat(self, context: BossaNovaContext, dataflow):
        pass

    def get_report(self):
        return {}
