from dataclasses import dataclass, field
from typing import Dict, Tuple
from nova_platform.base_model import DType
from nova_platform.utils.config_utils import BaseEnum, ConfigLoader, Deserializable

INF = 99999999


@dataclass
class AbstractCacheConfig:
    # Important: all below values are the value per cache bank

    CACHE_LINE_SIZE: int = field(default=128)
    CACHE_WAYS: int = field(default=None)
    CACHE_SIZE: int = field(default=None)
    MEM_LATENCY: int = field(default=0)
    NON_MEM_LATENCY: int = field(default=0)
    NUM_MSHR: int = field(default=INF)


@dataclass
class L1C_Config(AbstractCacheConfig):
    NUM_OF_CORE: int = field(default=None)
    CACHE_SIZE_PER_CORE: int = field(default=None)


@dataclass
class LLC_Config(AbstractCacheConfig):
    NUM_OF_PARTITIONS: int = field(default=None)
    NUM_OF_SLICES_PER_PARTITION: int = field(default=None)


@dataclass
class InstNumConfig:
    NUM_OF_CLUSTER:           int
    NUM_OF_CORE_PER_CLUSTER:  int
    NUM_OF_DSM:               int
    NUM_OF_DIE:               int = 1
    NUM_OF_ESL_PER_DIE:       int = 1


@dataclass
class FreqConfig:
    CORE:         float  # GHz
    INTERCONNECT: float
    NOC:          float
    LLC:          float
    MC:           float
    L3:           float
    ESL:          float = None

    def get_freq(self, domain: 'FreqDomain'):
        return getattr(self, domain.name)*1e9  # GHz => Hz


class FreqDomain(BaseEnum):
    CORE = "CORE"
    INTERCONNECT = "INTERCONNECT"
    NOC = "NOC"
    LLC = "LLC"
    MC = "MC"
    L3 = "L3"
    ESL = "ESL"


@dataclass
class ComputeConfig:
    thread_2d_throughput: Dict[DType, float]
    thread_1d_throughput: Dict[DType, float]
    thread_sfu_throughput: int = field(default=None)
    compute_parallel_capability: int = 0


@dataclass
class DTEConfig:
    THREAD_NUMBER:    int
    BW_PER_DTE_THEAD: int  # Byte/cycle


class BossaNovaBaseConfig:
    def deserialize(o: any):
        raise NotImplementedError()


@dataclass
class BWEle(Deserializable):
    pre_latency:        int = field(default=0)
    post_latency:       int = field(default=0)
    rw_ratio:         float = field(default=1)
    bw:                 int = field(default=0)
    bw_per_second:    float = field(default=0)

    def get_bw(self, freq):
        if self.bw:
            return self.bw
        else:
            return self.bw_per_second/freq

    def deserialize(o):
        # example 1: [[ 32,0], 256, 1]
        # example 2: [[355,0],   0, 1, !eval 8/2*10*2**40]
        # [[pre_latency,post_latency], Byte/Cycle, rw_ratio, Byte/s]
        # [[#1.1,#1.2],                #2,         #3,       #4]
        # 1.1: pre-latency/cycle
        # 1.2: post-latency/cycle
        # 2:   参数2和参数4互斥，带宽Byte/Cycle
        # 3:   rw_ratio=read bw/write bw
        # 4:   可选，当设置#4时，#2必须为0，带宽Byte/s
        if not isinstance(o, list):
            raise RuntimeError(f"bwele config is invalid {str(o)}")
        if len(o) == 3:
            pre, post = o[0]
            bw = o[1]
            rw = o[2]
            return BWEle(pre, post, rw, bw)
        elif len(o) == 4:
            if o[1] != 0:
                raise RuntimeError(
                    f"bwele config is invalid. when set #4, #2 must equal to 0. {str(o)}")
            pre, post = o[0]
            bw_per_second = o[3]
            rw = o[2]
            return BWEle(pre, post, rw, None, bw_per_second)

        raise RuntimeError(f"bwele config is invalid {str(o)}")


@dataclass
class BWFile:
    freq_domain:   FreqDomain = field(default=None)
    otsd:                 int = field(default=256)
    shared:             BWEle = field(default=None)
    local:              BWEle = field(default=None)
    sic_io:             BWEle = field(default=None)
    noc:                BWEle = field(default=None)
    llc:                BWEle = field(default=None)
    llc_far:            BWEle = field(default=None)
    mc:                 BWEle = field(default=None)
    l3:                 BWEle = field(default=None)


@dataclass
class BWConfig:
    xpu:        BWFile = field(default=None)
    cdte:       BWFile = field(default=None)
    l0:         BWFile = field(default=None)
    local:      BWFile = field(default=None)
    shared:     BWFile = field(default=None)
    sic_io:     BWFile = field(default=None)
    noc:        BWFile = field(default=None)
    llc:        BWFile = field(default=None)
    llc_far:    BWFile = field(default=None)
    mc:         BWFile = field(default=None)
    l3:         BWFile = field(default=None)
    esl:        BWFile = field(default=None)


@dataclass
class MemoryCommonConfig:
    DSM_PRIVATE_SHARED_RATIO: float


@dataclass
class MemoryL3Config:
    TOTAL_SIZE: int


@dataclass
class MemoryL1Config:
    SIZE_PER_CORE: int


@dataclass
class MemoryL2Config:
    SIZE_PER_SIC: int


@dataclass
class MemoryL0Config:
    IV_SIZE: int
    SMR_SIZE: int
    OA_SIZE: int


@dataclass
class MemoryConfig:
    common: MemoryCommonConfig
    l3: MemoryL3Config
    l2: MemoryL2Config
    l1: MemoryL1Config
    l0: MemoryL0Config

    llc: LLC_Config
    l1c: L1C_Config = None


@dataclass
class PowerSIPLibConfig:
    voltage:                          float
    voltage_scaling:                  float

    idle_power:                       float
    leakage_power:                    float
    plc_power:                        float

    xpu_l0_dsm_local_energy:          float
    xpu_l0_dsm_shared_energy:         float
    xpu_l0_l3_energy:                 float

    compute_2d_mac_energy:            Dict[DType, float]
    compute_1d_op_energy:             Dict[DType, float]
    compute_msf_op_energy:            float


@dataclass
class PowerL1LibConfig:
    voltage:                          float
    voltage_scaling:                  float
    idle_power:                       float
    leakage_power:                    float
    sip_master_energy:                float
    dte_master_energy:                float

    xpu_dsm_local_l3_energy:          float
    xpu_dsm_shared_l3_energy:         float
    xpu_dsm_shared_dsm_shared_energy: float
    xpu_l3_l3_energy:                 float

    dte_dsm_local_l3_energy:          float
    dte_dsm_shared_l3_energy:         float
    dte_dsm_shared_dsm_shared_energy: float
    dte_l3_l3_energy:                 float


@dataclass
class PowerSOCLibConfig:
    voltage:                     float
    voltage_scaling:             float
    mc_idle_power:               float
    dataflow_idle_power:         float
    esl_idle_power:              float
    d2d_idle_power:              float
    leakage_power:               float
    other_power:                 float
    cdte_master_energy:          float
    llc_hit_data_energy:         float
    llc_miss_data_energy:        float
    sic_io:                      float


@dataclass
class PowerMemLibConfig:
    voltage_scaling:             float
    leakage_power:               float
    hbm_active_power:            float
    hbm_idle_power:              float


@dataclass
class PowerD2DLibConfig:
    leakage_power:               float
    active_power:                float
    idle_power:                  float


@dataclass
class PowerESLLibConfig:
    idle_power:                  float
    leakage_power:               float


@dataclass
class PowerLibConfig:
    sip:                     PowerSIPLibConfig
    l1:                      PowerL1LibConfig
    soc:                     PowerSOCLibConfig
    mem:                     PowerMemLibConfig
    esl:                     PowerESLLibConfig
    d2d:                     PowerD2DLibConfig
    dtu_edc:                 float = 1000
    board_efficiency:        float = 1
    edc_filter_glitch:       float = 10e-9
    edc_current_interval:    int = 10


class TOPO(BaseEnum):
    STANDALONE = ("STANDALONE", 1)
    FULLMESH8 = ("FULLMESH8", 8)
    SUPERNODE4 = ("SUPERNODE4", 4)
    SUPERNODE8 = ("SUPERNODE8", 8)
    SUPERNODE16 = ("SUPERNODE16", 16)
    SUPERNODE32 = ("SUPERNODE32", 32)


@dataclass
class BossaNovaConfig:
    name: str
    arch_name: str
    inst_num: InstNumConfig
    freq: FreqConfig
    compute: ComputeConfig
    dte: DTEConfig
    # latency: LatencyConfig
    bw: BWConfig
    memory: MemoryConfig
    power: PowerLibConfig
    gather_mu: int = 1
    gcu_id: int = 0

    def __hash__(self):
        return hash(self.name)
