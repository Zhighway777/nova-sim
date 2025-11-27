from dataclasses import dataclass, field
from typing import Any, Dict, List

from nova_platform.utils.base_utils import BaseDataclass
from nova_platform.utils.config_utils import BaseEnum
MAX_TIME = 99999


class TensorFloat:
    pass


class DType(BaseEnum):
    FP4 = (float, 0.5)
    FP8 = (float, 1)
    FP16 = (float, 2)
    FP32 = (float, 4)
    INT8 = (int, 2)
    INT32 = (int, 4)
    INT64 = (int, 8)
    TF32 = (TensorFloat, 4)

    def get_bpe(self):
        return self.value[1]


class AddrDomain(str, BaseEnum):
    L0 = "L0"
    L1C = "L1C"
    LOCAL = "LOCAL"
    SHARED = "SHARED"
    LLC = "LLC"
    L3 = "L3"
    L3_FAR = "L3_FAR"
    L3_REMOTE = "L3_REMOTE"
    ESL = "ESL"

    @classmethod
    def get_addr_domain(cls, addr):
        # DSM: [4T, 4T+1GB]
        # L3: [5T, ]
        # List[ addr, size, w/r ]
        if 4 * 2**40 <= addr < 5 * 2**40:
            return AddrDomain.SHARED
        elif 5*2**40 <= addr:
            return AddrDomain.L3
        else:
            raise Exception("UNKNOWN addr domain %d", addr)


class DataflowActionType(str, BaseEnum):
    ODTE = "ODTE"
    CDTE = "CDTE"
    SDTE = "SDTE"
    XPU = "XPU"
    ESL = "ESL"


class DataflowOpType(str, BaseEnum):
    GEMM = "GEMM"
    ADD = "ADD"
    MUL = "MUL"
    SOFTMAX = "SOFTMAX"
    GELU = "GELU"
    SIGMOID = "SIGMOID"
    SILU = "SILU"
    RELU = "RELU"
    NOP = "NOP"
    DTE = "DTE"
    LAYERNORM = "LAYERNORM"
    GATHER = "GATHER"


@dataclass
class BaseActionStat:
    power_stat: any = None
    relative_ts: float = 0
    latency: float = 0
    name: str = ""


@dataclass
class EDCStat(BaseDataclass):
    dtu_edc_report: Any = None
    soc_edc_report: Any = None
    edc_total_latency: float = 0
    edc_acc_dict: Dict[float, float] = None
    edc_incr_percent: float = 0


@dataclass
class PostStat(BaseDataclass):
    total_latency: float = 0
    action_end_time: float = 0
    core_util: float = 0
    l3_rw_bw_util: float = 0
    sic_io_r_bw_util: float = 0
    sic_io_w_bw_util: float = 0
    sic_io_rw_bw_util: float = 0
    esl_bw_util: float = 0
    workload_balance: float = 0
    service_report_dict: Dict[str, Any] = field(default_factory=dict)
    edc: EDCStat = field(default_factory=EDCStat)
    d2d_tx_rw_bw_util: float = 0


@dataclass
class DataflowActionMemoryAccess:
    base_addr: int
    size: int
    rw: str


@dataclass
class DataflowActionMemoryStat(BaseActionStat):
    cache_stat: Dict[str, any] = None
    total_count: int = 0
    master: DataflowActionType = None
    src: AddrDomain = None
    dst: AddrDomain = None
    rw: str = None
    leading_latency: float = 0
    memory_access_list: List[DataflowActionMemoryAccess] = None
    remote_target_mem_access_list: List[DataflowActionMemoryAccess] = None
    write_through = False
    src_gcu_id: int = None
    tar_gcu_id: int = None
    is_done = False
    bw_factor: bool = 1.0


@dataclass
class DataflowActionComputeStat(BaseActionStat):
    compute_1d_ops: Dict[DType, float] = field(default_factory=dict)
    compute_2d_ops: Dict[DType, float] = field(default_factory=dict)
    compute_msf_ops: int = 0
    compute_scalar_cycle: int = 0
    compute_nop_cycle: int = 0
    compute_1d_efficiency: float = 1.0
    compute_2d_efficiency: float = 1.0
    compute_sfu_efficiency: float = 1.0

    def __iadd__(lhs: 'DataflowActionComputeStat', rhs: 'DataflowActionComputeStat'):
        for dt, val in rhs.compute_1d_ops.items():
            if dt not in lhs.compute_1d_ops:
                lhs.compute_1d_ops[dt] = 0
            lhs.compute_1d_ops[dt] += val
        for dt, val in rhs.compute_2d_ops.items():
            if dt not in lhs.compute_2d_ops:
                lhs.compute_2d_ops[dt] = 0
            lhs.compute_2d_ops[dt] += val

        lhs.latency += rhs.latency
        lhs.compute_scalar_cycle += rhs.compute_scalar_cycle
        lhs.compute_nop_cycle += rhs.compute_nop_cycle
        lhs.compute_1d_efficiency += rhs.compute_1d_efficiency
        lhs.compute_2d_efficiency += rhs.compute_2d_efficiency
        lhs.compute_sfu_efficiency+rhs.compute_sfu_efficiency

        return lhs


@dataclass
class BaseFrame(BaseDataclass):
    begin: float = field(default=0)  # second
    end:   float = field(default=MAX_TIME)  # second

    @property
    def dur(self):
        return self.end-self.begin

    def incr(self, frame: 'BaseFrame'):
        raise NotImplemented()

    def clone(self) -> 'BaseFrame':
        return self.__class__(**self.__dict__)


class BaseESLSwitch():

    def __init__(self, config, topo):
        self.gcu_map = {}
        self.port_map = {}
        self.config = config
        self.topo = topo

    def add_gcu(self, gcu_id, executor):
        self.gcu_map[gcu_id] = executor

    def build_bw_resource(self, bw_res_context):
        raise NotImplementedError

    def get_unique_bw_resource(self, bw_res_context):
        raise NotImplementedError

    def get_bw_resource(self, local_gpu_id: int, src_gcu_id: int, tar_gcu_id: int, rw):
        raise NotImplementedError

    def send(self, ref, src_gcu_id, tar_gcu_id, rw, data_size, memory_list=None):
        raise NotImplementedError
