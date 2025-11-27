import pytest
from bossa_nova.cost_service.cache.arch.libra import GCU_Libra
from bossa_nova.cost_service.cache.base_cache_model import CacheStat
from bossa_nova.cost_service.cache.cache_cost_service import CacheCostService
from bossa_nova.utils.config_utils import load_config
from bossa_nova.config import BossaNovaConfig
from bossa_nova.cost_service.cache.arch import RequestType, SimpleAccess


def _access(cache_svc, base_addr, size):
    for i in range(int(size//128)):
        access = SimpleAccess(
            0,
            base_addr+128*i,
            1,
            128,
            base_addr+128*i+128-1,
            request_type=RequestType.LLC,
            sip_id=0
        )
        cache_svc.cache_model.reuse_distance(access)
    cache_svc.cache_model.finish_queue()
    return cache_svc.get_cache_stat_dict()["LLC"]


config = load_config(f"config/libra_1DIE_3.2TB_24SIP_256OST.yaml", BossaNovaConfig)
config.memory.llc.NON_MEM_LATENCY = 0
config.memory.llc.MEM_LATENCY = 0
config.memory.llc.CACHE_SIZE = 16*128*2  # 2 cache sets per slice
config.memory.llc.NUM_OF_PARTITIONS = 4
config.memory.llc.NUM_OF_SLICES_PER_PARTITION = 4


@pytest.mark.ci
def test_cache_0_5_hit():

    cache_svc = CacheCostService(config)

    SLICE_NUMS = config.memory.llc.NUM_OF_PARTITIONS * \
        config.memory.llc.NUM_OF_SLICES_PER_PARTITION

    base_addr = 5*10**4//128*128
    size = config.memory.llc.CACHE_SIZE*SLICE_NUMS  # +(128*CACHE_SETS*4)

    stat: CacheStat = _access(
        cache_svc=cache_svc, base_addr=base_addr, size=size)

    assert stat.read_count == int(size//128)
    assert stat.read_hit_rate == 0.0

    stat: CacheStat = _access(
        cache_svc=cache_svc, base_addr=base_addr, size=size)
    assert stat.read_count == int(size//128)*2
    assert stat.read_hit_rate == 0.5


@pytest.mark.ci
def test_cache_cap_miss():
    cache_svc = CacheCostService(config)
    SLICE_NUMS = (
        config.memory.llc.NUM_OF_PARTITIONS
        * config.memory.llc.NUM_OF_SLICES_PER_PARTITION
    )
    CACHE_SETS = (config.memory.llc.CACHE_SIZE
                  / config.memory.llc.CACHE_LINE_SIZE
                  / config.memory.llc.CACHE_WAYS
                  * SLICE_NUMS
                  )
    base_addr = 5*10**4//128*128
    size = config.memory.llc.CACHE_SIZE*SLICE_NUMS + (128*CACHE_SETS)

    stat: CacheStat = _access(
        cache_svc=cache_svc, base_addr=base_addr, size=size)

    assert stat.read_count == int(size//128)
    assert stat.read_hit_rate == 0.0

    stat: CacheStat = _access(
        cache_svc=cache_svc, base_addr=base_addr, size=size)
    assert stat.read_count == int(size//128)*2
    assert stat.read_hit_rate == 0.0
