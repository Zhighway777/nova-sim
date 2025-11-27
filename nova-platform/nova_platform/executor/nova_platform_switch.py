from queue import Queue
from typing import List

from nova_platform.base_model import AddrDomain, DataflowActionMemoryAccess, DataflowActionMemoryStat, DataflowActionType
from nova_platform.base_model import BaseESLSwitch
from nova_platform.config import TOPO, BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BWMode, BWResource, BWResourceContext


class DefaultESLSwitch(BaseESLSwitch):
    def __init__(self, config: BossaNovaConfig, topo):
        super().__init__(config, topo)

    def calculate_esl_esl_latency():
        raise NotImplementedError

    def add_gcu(self, gcu_id, executor):
        self.gcu_map[gcu_id] = executor

    def send(self, ref, src_gcu_id, tar_gcu_id, rw, data_size, memory_list=None):
        tar_executor = self.gcu_map[tar_gcu_id]
        tar_compute_svc = tar_executor.compute_svc
        tar_cache_svc = tar_executor.compute_svc.cache_svc

        context = tar_executor.context
        sw_lat = self.calculate_esl_esl_latency(src_gcu_id, tar_gcu_id)
        edc_freq_cfg = tar_executor.power_svc.get_edc_freq(
            ref+sw_lat, context) if tar_executor.power_svc else None
        mem_stat = DataflowActionMemoryStat(
            total_count=data_size,
            master=DataflowActionType.ESL,
            src=AddrDomain.ESL,
            dst=AddrDomain.L3,
            rw=rw,
            relative_ts=sw_lat,
            src_gcu_id=src_gcu_id,
            tar_gcu_id=tar_gcu_id,
            memory_access_list=memory_list,
            name="esl_remote",
        )

        def get_cache_gen(action, context, ref):
            while True:
                mem_stat = yield
                if not mem_stat:
                    break
                if hasattr(tar_cache_svc, 'get_cache_stat_dict'):
                    memory_access_list: List[DataflowActionMemoryAccess] = mem_stat.memory_access_list
                    start_stat = tar_cache_svc.get_cache_stat_dict()
                    tar_cache_svc._process_access(action, memory_access_list)
                    end_stat = tar_cache_svc.get_cache_stat_dict()
                    cache_stat = {}
                    for k, v in end_stat.items():
                        cache_stat[k] = v-start_stat[k]
                else:
                    cache_stat = {}

                mem_stat.cache_stat = cache_stat
            pass

        cache_gen = get_cache_gen(None, context, ref)
        next(cache_gen)
        die_id = tar_gcu_id // self.config.inst_num.NUM_OF_ESL_PER_DIE
        latency_stat_list = yield from tar_compute_svc.process_memory_stat(
            mem_stat, context, edc_freq_cfg, cache_gen, die_id, None, None, f'esl gcu {src_gcu_id:02d}->{tar_gcu_id:02d}', ref)

        return mem_stat.latency+sw_lat, {
            "sw_latency": sw_lat,
            "mem_stat_latency": mem_stat.latency,
            "mem_stat_leading": mem_stat.leading_latency,
            "stat_list": latency_stat_list
        }


class FullmeshESLSwitch(DefaultESLSwitch):
    def __init__(self, config: BossaNovaConfig):
        super().__init__(config, TOPO.FULLMESH8)
        self.logic_port_num = TOPO.FULLMESH8.value[1]
        self.logic_port_per_die_num = self.logic_port_num//self.config.inst_num.NUM_OF_DIE
        # src gcu_id, port_id => tar gcu_id,port_id
        # self.port_map[(0, n)] = (n, 0)
        self.resource_map = {}
        for gcu_id in range(8):
            for port_id in range(8):
                if gcu_id == port_id:
                    continue
                die_id, local_port_id = self.__get_id(port_id)
                self.resource_map[(gcu_id, die_id, local_port_id, 'tx')] = BWResource(f"esl tx gcu {gcu_id} die_id {die_id} port {local_port_id}", BWMode.BANDWIDTH,
                                                                                      self.config.bw.esl.noc.bw_per_second*self.config.freq.ESL/1e9)
                self.resource_map[(gcu_id, die_id, local_port_id, 'rx')] = BWResource(f"esl rx gcu {gcu_id} die_id {die_id} port {local_port_id}", BWMode.BANDWIDTH,
                                                                                      self.config.bw.esl.noc.bw_per_second*self.config.freq.ESL/1e9)

    def calculate_esl_esl_latency(self, src_gcu_id, dst_gcu_id):
        # no sw
        return 0.7e-6

    def __get_id(self, tar_gcu_id):
        die_id = tar_gcu_id//self.logic_port_per_die_num
        port_id = tar_gcu_id % self. logic_port_per_die_num
        return (die_id, port_id)

    def build_bw_resource(self, bw_res_context: BWResourceContext):
        for port_id in range(self.logic_port_num):
            if port_id == bw_res_context.config.gcu_id:
                continue
            die_id, local_port_id = self.__get_id(port_id)
            bw_res_context.esl_dict[(die_id, port_id, 'tx')] = self.resource_map[(
                bw_res_context.config.gcu_id, die_id, local_port_id, 'tx')]
            bw_res_context.esl_dict[(die_id, port_id, 'rx')] = self.resource_map[(
                bw_res_context.config.gcu_id, die_id, local_port_id, 'rx')]

    def get_bw_resource(self, local_gpu_id: int, src_gcu_id: int, tar_gcu_id: int, rw):
        if local_gpu_id == src_gcu_id:
            die_id, local_port_id = self.__get_id(tar_gcu_id)
            key = (local_gpu_id, die_id, local_port_id,
                   'tx' if rw == 'r' else 'rx')
        elif local_gpu_id == tar_gcu_id:
            die_id, local_port_id = self.__get_id(src_gcu_id)
            key = (local_gpu_id, die_id, local_port_id,
                   'tx' if rw == 'r' else 'rx')
        return self.resource_map[key]

    def get_unique_bw_resource(self, context: BWResourceContext):
        for i in range(self.logic_port_num):
            die_id, port_id = self.__get_id(i)
            if (die_id, port_id, 'tx') in context.esl_dict:
                yield (f"esl_tx", "r", die_id, None, port_id, context.esl_dict[die_id, port_id, 'tx'])
            if (die_id, port_id, 'rx') in context.esl_dict:
                yield (f"esl_rx", "w", die_id, None, port_id, context.esl_dict[die_id, port_id, 'rx'])


class SupernodeESLSwitch(DefaultESLSwitch):
    node_num_per_rank = 16

    def __init__(self, config: BossaNovaConfig, topo):
        super().__init__(config, topo)
        self.node_num = topo.value[1]
        self.resource_map = {}
        node_port_num = self.config.inst_num.NUM_OF_ESL_PER_DIE * \
            self.config.inst_num.NUM_OF_DIE
        # 25GB/s * 16ports = 400GB/s
        node_bw = self.config.bw.esl.noc.bw_per_second * \
            self.config.freq.ESL*node_port_num/1e9
        self.rank_num = self.node_num//self.node_num_per_rank

        for gcu_id in range(self.node_num):
            key = (gcu_id, 'tx')
            tx_bw = BWResource(f"esl tx supernode", BWMode.BANDWIDTH, node_bw)
            self.resource_map[key] = tx_bw
            key = (gcu_id, 'rx')
            rx_bw = BWResource(f"esl rx supernode", BWMode.BANDWIDTH, node_bw)
            self.resource_map[key] = rx_bw

    def get_parted_id(self, global_id):
        rank_id = global_id // self.node_num_per_rank
        local_id = global_id % self.node_num_per_rank
        return (rank_id, local_id)

    def calculate_esl_esl_latency(self, src_global_id, tar_global_id):
        src_rank_id, _ = self.get_parted_id(src_global_id)
        tar_rank_id, _ = self.get_parted_id(tar_global_id)
        if src_rank_id == tar_rank_id:
            return 5e-6
        else:
            return 5e-6*2

    def build_bw_resource(self, bw_res_context: BWResourceContext):
        global_gcu_id = bw_res_context.config.gcu_id
        bw_res_context.esl_dict[('tx')] = self.resource_map.get(
            global_gcu_id, 'tx')
        bw_res_context.esl_dict[('rx')] = self.resource_map.get(
            global_gcu_id, 'rx')

    def get_bw_resource(self, local_gpu_id: int, src_gcu_id: int, tar_gcu_id: int, rw):
        if local_gpu_id == src_gcu_id:
            key = (local_gpu_id, 'tx' if rw == 'r' else 'rx')
        elif local_gpu_id == tar_gcu_id:
            key = (tar_gcu_id, 'tx' if rw == 'r' else 'rx')
        return self.resource_map[key]

    def get_unique_bw_resource(self, context: BWResourceContext):
        global_gcu_id = context.config.gcu_id
        tx = self.resource_map.get((global_gcu_id, 'tx'))
        rx = self.resource_map.get((global_gcu_id, 'rx'))
        yield (f"esl_tx", "r", None, None, 0, tx)
        yield (f"esl_rx", "w", None, None, 0, rx)


class StandaloneSwitch(DefaultESLSwitch):
    def __init__(self, config: BossaNovaConfig):
        super().__init__(config, TOPO.STANDALONE)

    def build_bw_resource(self, context):
        pass

    def get_unique_bw_resource(self, context: BWResourceContext):
        if False:
            yield
        return


class ESLSwitchManager():
    @classmethod
    def build_switch(cls, config: BossaNovaConfig, topo: TOPO) -> BaseESLSwitch:
        switch_map = {
            TOPO.STANDALONE: lambda: StandaloneSwitch(config),
            TOPO.FULLMESH8: lambda: FullmeshESLSwitch(config),
            TOPO.SUPERNODE4: lambda: SupernodeESLSwitch(config, TOPO.SUPERNODE4),
            TOPO.SUPERNODE8: lambda: SupernodeESLSwitch(config, TOPO.SUPERNODE8),
            TOPO.SUPERNODE16: lambda: SupernodeESLSwitch(config, TOPO.SUPERNODE16),
            TOPO.SUPERNODE32: lambda: SupernodeESLSwitch(config, TOPO.SUPERNODE32),
        }
        if topo not in switch_map:
            raise NotImplementedError
        fun = switch_map[topo]
        return fun()
