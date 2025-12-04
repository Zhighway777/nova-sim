from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Generator, List, Tuple
from nova_platform.base_model import MAX_TIME, AddrDomain, DType, DataflowActionComputeStat, DataflowActionMemoryStat, DataflowActionType
from nova_platform.config import BossaNovaConfig, FreqConfig
from nova_platform.cost_service.cache.cache_cost_service import CacheCostService
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext, BaseCostService
from nova_platform.cost_service.compute.data_transport_service import DataTransportService
from nova_platform.cost_service.power.power_cost_service import PowerCostService
from nova_platform.dataflow.action.diag_action import DiagDataflowAction
from nova_platform.dataflow.dataflow import Dataflow
import logging

from nova_platform.executor.nova_platform_barrier import BARRIER
from nova_platform.executor.nova_platform_event import BossaNovaEvent
from nova_platform.executor.nova_platform_switch import BaseESLSwitch

logger = logging.getLogger(__name__)


@dataclass
class EngineStat:
    start_op_ref: float = MAX_TIME
    end_op_ref: float = 0
    engine_end_ref: float = 0


class ComputeCostService(BaseCostService):
    def __init__(
        self,
        config: BossaNovaConfig,
        power_svc: PowerCostService,
        dump_addr,
        cache_svc: CacheCostService | None = None,
        esl_switch: BaseESLSwitch = None,
    ):
        super().__init__(config)
        if cache_svc:
            self.cache_svc = cache_svc
        else:
            # make an empty cache_gen
            class EmptyCacheSvc:
                def process(self, action, context, ref):
                    while True:
                        stat: DataflowActionMemoryStat = yield
                        if not stat:
                            break
                        stat.cache_stat = {}
                    yield  # using yield instead of return to avoid from StopIterator exception
            self.cache_svc = EmptyCacheSvc()
        self.data_transport_service = DataTransportService(
            config,  esl_switch)
        self.power_svc = power_svc

        self.engine_stat_dict: Dict[Tuple[DataflowActionType, int, int, int], EngineStat] = defaultdict(
            EngineStat)  # key: (act_type,die_id,cluster_id,sip_id)

        self.dump_addr = dump_addr

    def process_compute_stat(
        self,
        stat: DataflowActionComputeStat,
        edc_freq_cfg: FreqConfig,
    ):
        def _process_throughput(ops_dict: Dict[DType, float], throughput_cfg: Dict[DType, float], factor=1):
            cyc = 0
            for dtype, ops in ops_dict.items():
                throughput = throughput_cfg.get(dtype, 0)
                cyc = max(cyc, ops/throughput)
            return cyc/factor

        cyc_1d = _process_throughput(
            stat.compute_1d_ops, self.config.compute.thread_1d_throughput, factor=stat.compute_1d_efficiency
        )
        cyc_2d = _process_throughput(
            stat.compute_2d_ops, self.config.compute.thread_2d_throughput, factor=2 *
            stat.compute_2d_efficiency
        )  # config unit [mac]
        cyc_sfu = stat.compute_msf_ops / \
            self.config.compute.thread_sfu_throughput / stat.compute_sfu_efficiency
        cyc_scalar = stat.compute_scalar_cycle  # TODO: config
        cyc_nop = stat.compute_nop_cycle
        cycle = max(cyc_1d, cyc_2d, cyc_sfu, cyc_scalar, cyc_nop)
        stat.latency = cycle/edc_freq_cfg.CORE/1e9
        return stat.latency

    def process_memory_stat(
        self,
        stat: DataflowActionMemoryStat,
        context: BossaNovaContext,
        edc_freq_cfg,
        cache_gen,
        die_id,
        cluster_id,
        engine_id,
        action_name,
        ref: float,
    ):
        try:
            cache_gen.send(stat)
        except (StopIteration, RuntimeError) as e:
            stat.cache_stat = {}
            logger.warning(
                "cache gen stopped before mem stat. %s", e)
        if stat.total_count > 10 * 2**10:  # >10kB
            chunk_size = (stat.total_count // 10 // 128 + 1)*128
            # chunk_size = stat.total_count // 10
        # if stat.total_count > 10*1024*1024:  # >10kB
        #     chunk_size = 10*1024*1024
        else:
            chunk_size = stat.total_count
        # chunk_size = stat.total_count

        def split_array_with_remainder_first(total_count, chunk_size):
            remainder = total_count % chunk_size  # 计算余数部分大小
            offset = 0

            # 如果有余数部分，先生成余数部分
            if remainder > 0:
                yield offset, remainder
                offset += remainder

            # 生成完整的 chunk
            for start in range(remainder, total_count, chunk_size):
                yield start, chunk_size
                offset += chunk_size

        def _iter_stat(_stat: DataflowActionMemoryStat):
            chunk_stat = DataflowActionMemoryStat(**_stat.__dict__)
            if chunk_stat.dst == 'L3_REMOTE':
                chunk_stat.rw = 'r' if _stat.rw == 'w' else 'w'
            # for offset in range(0, _stat.total_count, chunk_size):
            #     chunk_stat.is_done = False
            #     chunk_stat.total_count = min(
            #         chunk_size, _stat.total_count - offset)
            #     yield chunk_stat
            for offset, size in split_array_with_remainder_first(_stat.total_count, chunk_size):
                chunk_stat.is_done = False
                chunk_stat.total_count = size
                yield chunk_stat

        def _iter_stat_list(stat_list: List[DataflowActionMemoryStat]):
            iter_list = [_iter_stat(s) for s in stat_list]
            while iter_list:
                _stat_g = iter_list.pop(0)
                try:
                    yield next(_stat_g)
                    iter_list.append(_stat_g)
                except StopIteration as si:
                    pass

        stat_list = []
        if stat.dst == AddrDomain.L3:
            _dst = AddrDomain.L3_FAR
            for i in range(self.config.inst_num.NUM_OF_DIE):
                _stat = DataflowActionMemoryStat(**stat.__dict__)
                _stat.total_count = int(
                    _stat.total_count/self.config.inst_num.NUM_OF_DIE)
                if i != die_id:
                    _stat.dst = _dst
                stat_list.append(_stat)
            bw_factor = 1/self.config.inst_num.NUM_OF_DIE
        elif (stat.master, stat.src, stat.dst, stat.rw) == (DataflowActionType.ESL, AddrDomain.L3, AddrDomain.L3_REMOTE, 'w'):  # esl master write
            _stat = DataflowActionMemoryStat(**stat.__dict__)
            _stat.src = AddrDomain.ESL
            _stat.dst = AddrDomain.L3
            _stat.rw = 'r'
            for i in range(self.config.inst_num.NUM_OF_DIE):
                __stat = DataflowActionMemoryStat(**_stat.__dict__)
                __stat.total_count = int(
                    __stat.total_count/self.config.inst_num.NUM_OF_DIE)
                if i != die_id:
                    __stat.dst = AddrDomain.L3_FAR
                stat_list.append(__stat)
            bw_factor = 1/self.config.inst_num.NUM_OF_DIE
        elif (stat.master, stat.src, stat.dst, stat.rw) == (DataflowActionType.ESL, AddrDomain.L3, AddrDomain.L3_REMOTE, 'r'):  # esl master read
            raise NotImplemented
        elif (stat.master, stat.src, stat.dst, stat.rw) == (DataflowActionType.XPU, AddrDomain.L3, AddrDomain.L3_REMOTE, 'w'):  # esl slave store
            raise NotImplemented
        elif (stat.master, stat.src, stat.dst, stat.rw) == (DataflowActionType.XPU, AddrDomain.L0, AddrDomain.L3_REMOTE, 'r'):  # esl slave load
            raise NotImplemented
        else:
            stat_list.append(stat)
            bw_factor = 1

        # consider mem stat bw
        bw_factor *= stat.bw_factor

        latency_stat_list = []

        # track_detail1 = context.get_cluster_tgen(die_id, cluster_id).create_track(
        #     f"{action_type}:{engine_id}:dataflow_detail1")
        # track_detail2 = context.get_cluster_tgen(die_id, cluster_id).create_track(
        #     f"{action_type}:{engine_id}:dataflow_detail2")
        # track_detail3 = context.get_cluster_tgen(die_id, cluster_id).create_track(
        #     f"{action_type}:{engine_id}:dataflow_detail3")
        for idx, chunk_stat in enumerate(_iter_stat_list(stat_list)):
            # calculate latency
            while not chunk_stat.is_done:
                _latency, leading_latency, latency_stat = yield from self.data_transport_service.compute_latency(
                    context, chunk_stat,
                    die_id,
                    cluster_id,

                    ref + chunk_stat.relative_ts, stat.cache_stat, edc_freq_cfg,
                    bw_factor=bw_factor
                )
                if not chunk_stat.is_done:
                    next_ref = ref + chunk_stat.relative_ts
                    yield next_ref, chunk_stat

            latency_stat_list.append(latency_stat)

            last_leading_dur = latency_stat[-1]["data_start_ref"] - \
                latency_stat[-1]["ref"]
            next_ref = (
                latency_stat[-1]["data_end_ref"] -
                last_leading_dur
            )

            if stat.master == DataflowActionType.ESL and stat.name != 'esl_remote':
                next_ref = latency_stat[-1]['raw_data_lat'] + \
                    stat.relative_ts+ref

            chunk_stat.relative_ts = next_ref-ref
            # i += 1
            yield next_ref, stat

        stat_far_data_start_ref = 0
        stat_end_ref = 0
        # latency_stat_list  >>> [
        #   [near_die_chunk0_subpath0, near_die_chunk0_path1, ...], [far_die_chunk0_subpath0, far_die_chunk1_subpath1, ...],
        #   [near_die_chunk1_subpath0, near_die_chunk1_path1, ...], [far_die_chunk1_subpath0, far_die_chunk1_subpath1, ...],
        # ]
        first_chunk_list = latency_stat_list[0:len(stat_list)]
        last_chunk_list = latency_stat_list[-len(stat_list):]
        for i in range(len(stat_list)):
            stat_far_data_start_ref = max(
                stat_far_data_start_ref, first_chunk_list[i][0]['data_start_ref'])

            _stat_end_ref = max([s['data_end_ref']
                                for s in last_chunk_list[i]])

            stat_end_ref = max(stat_end_ref, _stat_end_ref)

        # for _s in latency_stat_list:
        #     _stat_end_ref = max([_sub['data_end_ref']
        #                         for _sub in _s])
        #     stat_end_ref = max(stat_end_ref, _stat_end_ref)

        if (stat.rw == 'w' and stat.write_through == False and stat.master != DataflowActionType.ESL) or \
           (stat.rw == 'w' and stat.write_through == False and stat.master == DataflowActionType.ESL and stat.name == 'esl_remote'):
            # 当write back且经过l3时, 寻找llc/llc_far数据传输end time作为respond时间
            _stat_far_data_start_ref = 0
            for i in range(len(stat_list)):
                for _sub_path_stat in first_chunk_list[i]:
                    _src, _dst = _sub_path_stat['src'], _sub_path_stat['dst']
                    if (_src.lower(), _dst.lower()) in [('noc', 'llc'), ('noc', 'llc_far')]:
                        _stat_far_data_start_ref = max(
                            _stat_far_data_start_ref, _sub_path_stat["data_start_ref"])
                        break
            if _stat_far_data_start_ref > 0:
                stat_far_data_start_ref = min(
                    stat_far_data_start_ref, _stat_far_data_start_ref)
            _stat_end_ref = 0
            for i in range(len(stat_list)):
                for _sub_path_stat in last_chunk_list[i]:
                    _src, _dst = _sub_path_stat['src'], _sub_path_stat['dst']
                    if (_src.lower(), _dst.lower()) in [('noc', 'llc'), ('noc', 'llc_far')]:
                        _stat_end_ref = max(
                            _stat_end_ref, _sub_path_stat["data_end_ref"])
                        break
            if _stat_end_ref > 0:
                stat_end_ref = min(stat_end_ref, _stat_end_ref)

        stat.latency = stat_end_ref-(stat.relative_ts+ref)

        # stat.leading_latency = leading_latency
        # stat.leading_latency = stat_data_start_ref - \
        #     (ref+stat.relative_ts)
        stat.leading_latency = stat_far_data_start_ref - \
            (ref+stat.relative_ts)
        track = context.get_cluster_tgen(die_id, cluster_id).create_track(
            f"{action_name}:{engine_id}:dataflow", tid=engine_id)
        stat_name = stat.name if stat.name else f"m"
        track.duration(
            ref + stat.relative_ts,
            stat.leading_latency,
            f"{stat_name}:leading:{stat.rw}",
            latency_stat_list,
            category_list=["memory", "leading"],
        )
        track.duration(
            ref + stat.relative_ts + stat.leading_latency,
            stat.latency - stat.leading_latency,
            f"{stat_name}:latency:{stat.rw}",
            stat,
            category_list=["memory", "latency"],
        )
        return latency_stat_list
        # for debug
        # for c, chunk in enumerate(latency_stat_list):
        #     for i, layer in enumerate(chunk):
        #         seq = layer['seq']
        #         if c % 2 == 0:
        #             near = 'near'
        #         else:
        #             near = 'far'
        #         name = f"c{c}:l{i}:{layer['src']}->{layer['dst']}:{seq}"
        #         _track = context.get_cluster_tgen(die_id, cluster_id).create_track(
        #             f"{action_type}:{engine_id}:dataflow_{near}:{stat_name}", tid=engine_id)
        #         # start = layer['ref']
        #         # dur = layer['end_ts']-start
        #         start = layer['data_start_ref']
        #         esl_dur = layer['esl_data_lat'] if 'esl_data_lat' in layer else 0
        #         total_dur = layer['data_lat']
        #         _track.duration(
        #             start,
        #             total_dur,
        #             name,
        #             layer,
        #             category_list=["memory", "detail"],
        #         )
        #         if (layer['src'], layer['dst']) == ('ESL', 'NOC'):
        #             name = f"c{c}:l{i}:{layer['src']}->{layer['dst']}->remote:{seq}"
        #             _track.duration(
        #                 start+total_dur-esl_dur,
        #                 esl_dur,
        #                 name,
        #                 layer,
        #                 category_list=["memory", "detail"],
        #             )

    def process(self, action: DiagDataflowAction, context: BossaNovaContext, ref: float, trace_label: str | None = None) -> Generator[float, None, None]:
        die_id = action.get_die_id()
        cid = action.get_cluster_id()
        engine_id = action.get_local_engine_id()
        action_type = action.get_action_type()
        trace_label = trace_label or action_type.name

        cost_book = context.get_cost_book(action)
        stat_gen = action.compute(context)

        cache_gen = self.cache_svc.process(
            action, context, ref)
        next(cache_gen)

        power_gen = self.power_svc.process(
            action, context, ref)
        next(power_gen)

        engine_stat = self.engine_stat_dict[(
            action_type, die_id, cid, engine_id)]

        core_stat = BaseCoreStat()
        try:
            i = 0
            while stat := next(stat_gen):
                if isinstance(stat, BARRIER) or isinstance(stat, BossaNovaEvent):
                    yield stat.max_t, stat
                    continue

                # temp_context = None
                # llc_stat = self.cache_svc.post_stat(temp_context)
                # logging.info(f"LLC in every action: {llc_stat}")
                # addr_stat, data_size = self.cache_svc.get_access_addr(stat)
                edc_freq_cfg = self.power_svc.get_edc_freq(
                    ref+stat.relative_ts, context)
                if issubclass(type(stat), DataflowActionComputeStat):
                    compute_latency = self.process_compute_stat(
                        stat, edc_freq_cfg)
                    track = context.get_cluster_tgen(die_id, cid).create_track(
                        f"{trace_label}:{engine_id}:compute", tid=engine_id)
                    stat_name = stat.name if stat.name else f"compute"
                    track.duration(
                        ref + stat.relative_ts, stat.latency, stat_name, stat, category_list=["compute"]
                    )

                    # update stat
                    start_ref = ref + stat.relative_ts
                    if start_ref < engine_stat.start_op_ref:
                        engine_stat.start_op_ref = start_ref
                    end_ref = ref + stat.relative_ts+stat.latency
                    if end_ref > engine_stat.end_op_ref:
                        engine_stat.end_op_ref = end_ref

                    yield ref+stat.relative_ts+compute_latency, stat
                elif issubclass(type(stat), DataflowActionMemoryStat):

                    # calculate cache
                    # TODO: need review
                    yield from self.process_memory_stat(stat, context, edc_freq_cfg, cache_gen, die_id, cid, engine_id, trace_label, ref)
                    self.dump_addr(action, stat, ref)

                    # track = context.tgen._create_track(
                    #     track._uuid, f"{action_type}:{engine_id}:Detail", 0)
                # if edc_freq_cfg.CORE_CLOCK_DOMAIN < self.config.freq.CORE_CLOCK_DOMAIN:
                #     track.instant(ref+stat.relative_ts, "edc triggered", {
                #                   "freq": edc_freq_cfg.CORE_CLOCK_DOMAIN})
                power_gen.send(stat)
                core_stat += stat

                engine_end_ref = ref + stat.relative_ts+stat.latency
                if engine_end_ref > engine_stat.engine_end_ref:
                    engine_stat.engine_end_ref = engine_end_ref

            next(stat_gen)
        except StopIteration as res:
            # cost_book.core_stat = res.value
            cost_book.core_stat = core_stat

            cost_book.latency = res.value.latency
        finally:
            # must send None at the end to notify cache gen finish work of this action
            try:
                cache_gen.send(None)
            except StopIteration as e:
                pass
            try:
                power_gen.send(None)
            except StopIteration as e:
                pass

        logger.debug("cost servcie - action %03d:%025s done",
                     action.get_action_id(), action.get_action_type())

    def _longest_sip_stat(self, context: BossaNovaContext, per_sip_stat):
        index = None
        max_end_ref = 0
        for _index, _stat in self.engine_stat_dict.items():
            if _index[0] == DataflowActionType.XPU:
                engine_end_ref = _stat.engine_end_ref
                if engine_end_ref > max_end_ref:
                    max_end_ref = engine_end_ref
                    index = _index

        if index:
            _stat = self.engine_stat_dict[index]
            initial_ref = context.initial_ref
            return {
                'die_id': index[1],
                'cluster_id': index[2],
                'sip_id': index[3],
                'ops': per_sip_stat[index[1:]].to_dict(),
                'prolog': f"{_stat.start_op_ref-initial_ref:.3E}",
                'epilog': f"{context.post_stat.total_latency-(_stat.end_op_ref-initial_ref):.3E}",
                'main_body': f"{_stat.end_op_ref-_stat.start_op_ref:.3E}"
            }
        return None

    def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):

        stat = BaseCoreStat()
        per_sip_stat = {}

        for action_id, cost_book in context.cost_dict.items():
            action = dataflow._action_map[action_id]
            die_id = action.get_die_id()
            cid = action.get_cluster_id()
            engine_id = action.get_local_engine_id()
            per_sip_stat.setdefault(
                (die_id, cid, engine_id), BaseCoreStat())
            per_sip_stat[(die_id, cid, engine_id)] += cost_book.core_stat
            if cost_book.core_stat:
                stat += cost_book.core_stat

        report = stat.to_dict()

        sip_max_vector_ops = 0
        sip_max_tensor_macs = 0
        sip_max_sfu_ops = 0
        for die_id, cid, engine_id in per_sip_stat.keys():
            _vector_ops = per_sip_stat[(
                die_id, cid, engine_id)].vector_ops.values()
            _vector_ops = max(_vector_ops) if _vector_ops else 0
            _tensor_macs = per_sip_stat[(
                die_id, cid, engine_id)].tensor_macs.values()
            _tensor_macs = max(_tensor_macs) if _tensor_macs else 0
            _sfu_ops = per_sip_stat[(die_id, cid, engine_id)].sfu_ops
            sip_max_vector_ops = max(sip_max_vector_ops, _vector_ops)
            sip_max_tensor_macs = max(sip_max_tensor_macs, _tensor_macs)
            sip_max_sfu_ops = max(sip_max_sfu_ops, _sfu_ops)

        num_of_cores = self.config.inst_num.NUM_OF_CORE_PER_CLUSTER * \
            self.config.inst_num.NUM_OF_CLUSTER*self.config.inst_num.NUM_OF_DIE
        workload_balance = {
            "sip_max_vector_ops": sip_max_vector_ops,
            "sip_max_tensor_macs": sip_max_tensor_macs,
            "sip_max_sfu_ops": sip_max_sfu_ops,
            "vector_ops_rate": max(stat.vector_ops.values())/sip_max_vector_ops / num_of_cores if sip_max_vector_ops else 0,
            "tensor_macs_rate": max(stat.tensor_macs.values())/sip_max_tensor_macs / num_of_cores if sip_max_tensor_macs else 0,
            "sfu_ops_rate": stat.sfu_ops/sip_max_sfu_ops / num_of_cores if sip_max_sfu_ops else 0,
        }
        report.update({"workload_balance": workload_balance})
        longest_sip_stat = self._longest_sip_stat(context, per_sip_stat)
        report.update({"longest_sip_stat": longest_sip_stat})

        # context.post_stat.core_util = stat['action_time'] / \
        #     num_of_cores/context.post_stat.total_latency
        if workload_balance["tensor_macs_rate"] > 0:
            context.post_stat.workload_balance = workload_balance["tensor_macs_rate"]
        elif workload_balance["vector_ops_rate"] > 0:
            context.post_stat.workload_balance = workload_balance["vector_ops_rate"]
        elif workload_balance["sfu_ops_rate"] > 0:
            context.post_stat.workload_balance = workload_balance["sfu_ops_rate"]
        # report.update({
        #     "action_count": f"{stat['action_count']:,d}",
        #     "action_time(ns)": f"{int(stat['action_time']*1e9):,d}",
        # })
        # report = dict(sorted(report.items()))

        for (lvl, rw, die_id, cluster_id, esl_port_id, bw_resource) in context.bw_resource_context.get_unique_bw_resource_list():
            key = f"{lvl}_{rw}_total"
            if key not in report:
                report[key] = 0
            if rw == 'r':
                report[key] += bw_resource.acc_read
            elif rw == 'w':
                report[key] += bw_resource.acc_write
            elif rw == 'rw':
                report[key] += bw_resource.acc_read + bw_resource.acc_write
            else:
                raise Exception(f"unsupported rw: {rw}")

        for k, v in report.items():
            if type(v) == int:
                report[k] = f"{v:,d}"

        return report
