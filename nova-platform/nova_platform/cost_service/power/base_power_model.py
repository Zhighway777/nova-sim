from dataclasses import MISSING, Field, dataclass, field
import dataclasses
from typing import Callable, Dict, List

from nova_platform.base_model import AddrDomain, BaseFrame, DType, DataflowActionType
from nova_platform.config import PowerLibConfig
from nova_platform.base_model import DataflowActionComputeStat, DataflowActionMemoryStat

VALID_TYPE = Callable[[DataflowActionComputeStat |
                       DataflowActionMemoryStat], bool]
PW_FORMULA_TYPE = Callable[[DataflowActionComputeStat |
                            DataflowActionMemoryStat, PowerLibConfig], float]


class PowerField(Field):
    def __init__(self, default, default_factory, init, repr, hash, compare, metadata, kw_only, formula: PW_FORMULA_TYPE, pre_check: VALID_TYPE):
        super().__init__(default, default_factory, init,
                         repr, hash, compare, metadata, kw_only)
        self.formula = formula
        self.pre_check = pre_check


def power_field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
                hash=None, compare=True, metadata=None, kw_only=MISSING,
                formula: PW_FORMULA_TYPE = None, pre_check=None):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('cannot specify both default and default_factory')
    if default is MISSING and default_factory is MISSING:
        default = 0
    return PowerField(default, default_factory, init, repr, hash, compare,
                      metadata, kw_only, formula, pre_check)

# ops: factor=1, mac: factor=2


def sum_sub_items(ops_dict: Dict[DType, float], pw_cfg: Dict[DType, float]):
    sum_energy = 0
    for dtype, ops in ops_dict.items():
        # compute_tensor_fp16_energy
        energy_lib = pw_cfg[dtype]
        sum_energy += ops*energy_lib
    return sum_energy


@dataclass
class BasePowerDomain:
    def __add__(self, o: 'BasePowerDomain'):
        sum_obj = self.__class__()
        for f in dataclasses.fields(self):
            left = getattr(self, f.name)
            right = getattr(o, f.name)
            setattr(sum_obj, f.name, left+right)
        return sum_obj

    def __mul__(self, scalar):
        mul_obj = self.__class__()
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)*scalar
            setattr(mul_obj, f.name, val)
        return mul_obj


def check_stat_type(stat_type: DataflowActionMemoryStat | DataflowActionComputeStat):
    def _check(stat: DataflowActionMemoryStat | DataflowActionComputeStat):
        return isinstance(stat, stat_type)
    return _check


def check_master(master):
    def _check(stat: DataflowActionMemoryStat | DataflowActionComputeStat):
        return stat.master == master
    return _check


def check_list(*args):
    def _check(stat):
        for checker in args:
            if not checker(stat):
                return False
        return True
    return _check


CHECK_C = check_stat_type(DataflowActionComputeStat)
CHECK_M = check_stat_type(DataflowActionMemoryStat)
CHECK_XPU_M = check_list(
    check_stat_type(DataflowActionMemoryStat),
    check_master(DataflowActionType.XPU),
)
CHECK_DTE_M = check_list(
    check_stat_type(DataflowActionMemoryStat),
    check_master(DataflowActionType.CDTE),
)


def check_stat_src_dst(master, src, dst) -> VALID_TYPE:
    def _check(stat: DataflowActionMemoryStat | DataflowActionComputeStat):
        if isinstance(stat, DataflowActionComputeStat):
            return False
        if master != stat.master:
            return False
        if src == stat.src and dst == stat.dst:
            return True
        return False

    return _check


CHECK_XPU_L0_DSM_LOCAL = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.L0, AddrDomain.LOCAL)
CHECK_XPU_L0_DSM_SHARED = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.L0, AddrDomain.SHARED)

CHECK_XPU_L0_L3 = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.L0, AddrDomain.L3)
CHECK_XPU_DSM_LOCAL_L3 = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.LOCAL, AddrDomain.L3)
CHECK_XPU_DSM_SHARED_L3 = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.SHARED, AddrDomain.L3)
CHECK_XPU_L3_L3 = check_stat_src_dst(
    DataflowActionType.XPU, AddrDomain.L3, AddrDomain.L3)
CHECK_XPU_DSM_SHARED_DSM_SHARED = check_stat_src_dst(DataflowActionType.XPU,
                                                     AddrDomain.SHARED, AddrDomain.SHARED)

CHECK_DTE_L0_L3 = check_stat_src_dst(
    DataflowActionType.CDTE, AddrDomain.L0, AddrDomain.L3)
CHECK_DTE_DSM_LOCAL_L3 = check_stat_src_dst(
    DataflowActionType.CDTE, AddrDomain.LOCAL, AddrDomain.L3)
CHECK_DTE_DSM_SHARED_L3 = check_stat_src_dst(
    DataflowActionType.CDTE, AddrDomain.SHARED, AddrDomain.L3)
CHECK_DTE_L3_L3 = check_stat_src_dst(
    DataflowActionType.CDTE, AddrDomain.L3, AddrDomain.L3)
CHECK_DTE_DSM_SHARED_DSM_SHARED = check_stat_src_dst(DataflowActionType.CDTE,
                                                     AddrDomain.SHARED, AddrDomain.SHARED)


@dataclass
class PowerSIPDomain(BasePowerDomain):
    # autopep8: off
    xpu_2d_compute:                    float = power_field(
        pre_check=CHECK_C,
        formula=lambda stat, pw_cfg: sum_sub_items(
            stat.compute_2d_ops, pw_cfg.sip.compute_2d_mac_energy)/2*pw_cfg.sip.voltage_scaling/stat.latency
        # for 2d, the energy in power lib is energy/mac, thus a factor 2 is in above expression to convert ops to mac
    )
    xpu_1d_compute:                    float = power_field(
        pre_check=CHECK_C,
        formula=lambda stat, pw_cfg: sum_sub_items(
            stat.compute_1d_ops, pw_cfg.sip.compute_1d_op_energy)*pw_cfg.sip.voltage_scaling/stat.latency
    )
    xpu_msf_compute:                   float = power_field(
        pre_check=CHECK_C,
        formula=lambda stat, pw_cfg: stat.compute_msf_ops *
        pw_cfg.sip.compute_msf_op_energy*pw_cfg.sip.voltage_scaling/stat.latency
    )
    xpu_ld_st_l0_dsm_local:            float = power_field(
        pre_check=CHECK_XPU_L0_DSM_LOCAL,
        formula=lambda stat, pw_cfg: stat.total_count *
        pw_cfg.sip.xpu_l0_dsm_local_energy*pw_cfg.sip.voltage_scaling/stat.latency
    )
    xpu_ld_st_l0_dsm_shared:           float = power_field(
        pre_check=CHECK_XPU_L0_DSM_SHARED,
        formula=lambda stat, pw_cfg: stat.total_count *
        pw_cfg.sip.xpu_l0_dsm_shared_energy*pw_cfg.sip.voltage_scaling/stat.latency
    )
    xpu_ld_st_l0_l3:                   float = power_field(
        pre_check=CHECK_XPU_L0_L3,
        formula=lambda stat, pw_cfg: stat.total_count *
        pw_cfg.sip.xpu_l0_l3_energy*pw_cfg.sip.voltage_scaling/stat.latency
    )
    # autopep8: on


@dataclass
class PowerL1Domain(BasePowerDomain):
    # autopep8: off

    xpu_ld_st_dsm_local_l3:            float = power_field( 
        pre_check=CHECK_XPU_DSM_LOCAL_L3,
        formula=lambda stat, pw_cfg: stat.total_count*(pw_cfg.l1.xpu_dsm_local_l3_energy+pw_cfg.l1.sip_master_energy)*pw_cfg.l1.voltage_scaling/stat.latency
    )
    xpu_ld_st_dsm_shared_l3:           float = power_field( 
        pre_check=CHECK_XPU_DSM_SHARED_L3,
        formula=lambda stat, pw_cfg: stat.total_count*(pw_cfg.l1.xpu_dsm_shared_l3_energy+pw_cfg.l1.sip_master_energy)*pw_cfg.l1.voltage_scaling/stat.latency
    )
    xpu_ld_st_dsm_shared_dsm_shared:   float = power_field( 
        pre_check=CHECK_XPU_DSM_SHARED_DSM_SHARED,
        formula=lambda stat, pw_cfg: stat.total_count*(pw_cfg.l1.xpu_dsm_shared_dsm_shared_energy+pw_cfg.l1.sip_master_energy)*pw_cfg.l1.voltage_scaling/stat.latency
    )

    xpu_ld_st_l3_l3:                   float = power_field( 
        pre_check=CHECK_XPU_L3_L3,
        formula=lambda stat, pw_cfg: stat.total_count*(pw_cfg.l1.xpu_l3_l3_energy+pw_cfg.l1.sip_master_energy)*pw_cfg.l1.voltage_scaling/stat.latency
    )

    cdte_op_dsm_local_l3:              float = power_field( 
        pre_check=CHECK_DTE_DSM_LOCAL_L3,
        formula=lambda stat, pw_cfg: stat.total_count*pw_cfg.l1.dte_dsm_local_l3_energy*pw_cfg.l1.voltage_scaling/stat.latency
    )
    cdte_op_dsm_shared_l3:             float = power_field( 
        pre_check=CHECK_DTE_DSM_SHARED_L3,
        formula=lambda stat, pw_cfg: stat.total_count*pw_cfg.l1.dte_dsm_shared_l3_energy*pw_cfg.l1.voltage_scaling/stat.latency
    )
    cdte_op_dsm_shared_dsm_shared:     float = power_field( 
        pre_check=CHECK_DTE_DSM_SHARED_DSM_SHARED,
        formula=lambda stat, pw_cfg: stat.total_count*pw_cfg.l1.dte_dsm_shared_dsm_shared_energy*pw_cfg.l1.voltage_scaling/stat.latency
    )
    cdte_op_l3_l3:                     float = power_field( 
        pre_check=CHECK_DTE_L3_L3,
        formula=lambda stat, pw_cfg: stat.total_count*pw_cfg.l1.dte_l3_l3_energy*pw_cfg.l1.voltage_scaling/stat.latency
    )
    # autopep8: on


def get_hit_rate(stat: DataflowActionMemoryStat | DataflowActionComputeStat) -> float:
    llc_stat = stat.cache_stat.get("LLC", None)
    if llc_stat == None:
        return 0
    if stat.rw == 'r':
        hit_rate = llc_stat.read_hit_rate
    else:
        hit_rate = llc_stat.write_hit_rate
    return hit_rate


def get_miss_rate(stat) -> float:
    return 1-get_hit_rate(stat)


FORMULA_XPU_HIT_SOC_PW: PW_FORMULA_TYPE = (
    lambda stat, pw_cfg:
        stat.total_count
        * get_hit_rate(stat)
        * pw_cfg.soc.llc_hit_data_energy
        * pw_cfg.soc.voltage_scaling/stat.latency
)
FORMULA_XPU_MIS_SOC_PW: PW_FORMULA_TYPE = (
    lambda stat, pw_cfg:
        stat.total_count
        * get_miss_rate(stat)
        * pw_cfg.soc.llc_miss_data_energy
        * pw_cfg.soc.voltage_scaling/stat.latency
)

FORMULA_DTE_HIT_SOC_PW: PW_FORMULA_TYPE = (
    lambda stat, pw_cfg:
        stat.total_count
        * get_hit_rate(stat)
        * (pw_cfg.soc.llc_hit_data_energy+pw_cfg.soc.cdte_master_energy)
        * pw_cfg.soc.voltage_scaling/stat.latency
)

FORMULA_DTE_MIS_SOC_PW: PW_FORMULA_TYPE = (
    lambda stat, pw_cfg:
        stat.total_count
        * get_miss_rate(stat)
        * (pw_cfg.soc.llc_miss_data_energy+pw_cfg.soc.cdte_master_energy)
        * pw_cfg.soc.voltage_scaling/stat.latency
)


@dataclass
class PowerSOCDomain(BasePowerDomain):
    # autopep8: off
    xpu_ld_st_l0_l3_llc_hit:            float = power_field( 
        pre_check=CHECK_XPU_L0_L3,
        # LLC Hit Ratio * L3 Data Transcation * LLC Hit Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_HIT_SOC_PW
    )
    xpu_ld_st_l0_l3_llc_miss:           float = power_field( 
        pre_check=CHECK_XPU_L0_L3,
        # (1-LLC Hit Ratio )* L3 Data Transcation * LLC Miss Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_MIS_SOC_PW
    )
    xpu_ld_st_dsm_local_l3_llc_hit:     float = power_field( 
        pre_check=CHECK_XPU_DSM_LOCAL_L3,
        # LLC Hit Ratio * L3 Data Transcation * LLC Hit Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_HIT_SOC_PW
    )
    xpu_ld_st_dsm_local_l3_llc_miss:    float = power_field( 
        pre_check=CHECK_XPU_DSM_LOCAL_L3,
        # (1-LLC Hit Ratio) * LLC Data Transcation * LLC Miss Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_MIS_SOC_PW
    )
    xpu_ld_st_dsm_shared_l3_llc_hit:    float = power_field( 
        pre_check=CHECK_XPU_DSM_SHARED_L3,
        # LLC Hit Ratio * L3 Data Transcation * LLC Hit Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_HIT_SOC_PW
    )
    xpu_ld_st_dsm_shared_l3_llc_miss:   float = power_field( 
        pre_check=CHECK_XPU_DSM_SHARED_L3,
        # (1-LLC Hit Ratio) * LLC Data Transcation * LLC Miss Data energy lib  *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_XPU_MIS_SOC_PW
    )
    xpu_ld_st_l3_l3:                    float = power_field( 
        pre_check=CHECK_XPU_L3_L3,
        # (LLC read HIT Data Transcation *  LLC Hit Data energy lib + LLC read Miss Data Transcation *  LLC Miss Data energy lib + LLC write HIT Data Transcation *  LLC Hit Data energy lib + LLC write Miss Data Transcation *  LLC Miss Data energy lib) *Voltage Scaling / Data Transcation execution time 
        # we treat L3-L3 as two stat whcih are L3-L3 read mem stat and L3-L3 write mem stat
        formula=(
            lambda stat, pw_cfg:
                FORMULA_XPU_HIT_SOC_PW(stat,pw_cfg)+FORMULA_XPU_MIS_SOC_PW(stat,pw_cfg)
        )
    )

    cdte_op_dsm_local_l3_llc_hit:       float = power_field( 
        pre_check=CHECK_DTE_DSM_LOCAL_L3,
        # LLC Hit Ratio * LLC Data Transcation * ( LLC Hit Data energy lib + CDTE Master energy lib ) *Voltage Scaling/ Data Transcation execution time 
        formula=FORMULA_DTE_HIT_SOC_PW
    )
    cdte_op_dsm_local_l3_llc_miss:      float = power_field( 
        pre_check=CHECK_DTE_DSM_LOCAL_L3,
        formula=FORMULA_DTE_MIS_SOC_PW
    )
    cdte_op_dsm_shared_l3_llc_hit:      float = power_field( 
        pre_check=CHECK_DTE_DSM_SHARED_L3,
        formula=FORMULA_DTE_HIT_SOC_PW
    )
    cdte_op_dsm_shared_l3_llc_miss:     float = power_field( 
        pre_check=CHECK_DTE_DSM_SHARED_L3,
        formula=FORMULA_DTE_MIS_SOC_PW
    )
    cdte_op_dsm_shared_dsm_shared:      float = power_field( 
        pre_check=CHECK_DTE_DSM_SHARED_DSM_SHARED,
        # (LLC read HIT Data Transcation *  LLC Hit Data energy lib + LLC read Miss Data Transcation *  LLC Miss Data energy lib + LLC write HIT Data Transcation *  LLC Hit Data energy lib + LLC write Miss Data Transcation *  LLC Miss Data energy lib + CDTE Master energy lib* LLC Data Transcation ) *Voltage Scaling / Data Transcation execution time 
        # we treat L3-L3 as two stat whcih are L3-L3 read mem stat and L3-L3 write mem stat
        formula=(
            lambda stat, pw_cfg:
                FORMULA_DTE_HIT_SOC_PW(stat,pw_cfg)+FORMULA_DTE_HIT_SOC_PW(stat,pw_cfg)
        )
    )

    # autopep8: on


@dataclass
class PowerMemDomain(BasePowerDomain):
    hbm: float = power_field(
        pre_check=lambda stat: True,
        formula=lambda stat, pw_stat: 0
    )


@dataclass
class PowerStat:
    sip: PowerSIPDomain = field(default_factory=lambda: PowerSIPDomain())
    l1: PowerL1Domain = field(default_factory=lambda: PowerL1Domain())
    soc: PowerSOCDomain = field(default_factory=lambda: PowerSOCDomain())
    mem: PowerMemDomain = field(default_factory=lambda: PowerMemDomain())

    def __add__(self, o: 'PowerStat'):
        return PowerStat(
            sip=self.sip+o.sip,
            l1=self.l1+o.l1,
            soc=self.soc+o.soc,
            mem=self.mem+o.mem
        )

    def __mul__(self, scalar):
        return PowerStat(
            sip=self.sip*scalar,
            l1=self.l1*scalar,
            soc=self.soc*scalar,
            mem=self.mem*scalar
        )


@dataclass
class PowerFrame(BaseFrame):
    power: float = 0

    def incr(self, frame: 'PowerFrame'):
        self.power += frame.power


@dataclass
class PowerActiveFrame(BaseFrame):
    active: bool = False

    def incr(self, frame: 'PowerActiveFrame'):
        self.active |= frame.active


@dataclass
class ActiveFrame(BaseFrame):
    active: bool = False

    def incr(self, frame: 'ActiveFrame'):
        self.active |= frame.active


@dataclass
class PowerL3Frame(BaseFrame):
    l3_power: int = field(default=0)  # Byte/cycle
