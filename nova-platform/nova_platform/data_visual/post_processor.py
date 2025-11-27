from typing import Dict, List, Tuple, Any
import yaml
from dataclasses import Field, asdict, fields, field
from collections import Counter, defaultdict
from pathlib import Path
from nova_platform.base_model import PostStat
from nova_platform.config import BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BaseCostService, BossaNovaContext, CostBook, DataflowAction, EDCFrame, Timeline
from nova_platform.cost_service.power.base_power_model import PowerL1Domain, PowerMemDomain, PowerSIPDomain, PowerSOCDomain
from nova_platform.data_visual import AbstractPostProcessor
from nova_platform.dataflow.dataflow import Dataflow
from nova_platform.base_model import EDCStat, PostStat

import logging

from nova_platform.perfetto_protobuf._tgen import CounterTrack, TraceGenerator
logger = logging.getLogger(__name__)


def ts_convert(ns):
    return round(ns*1e6, 6)


class PostProcessor(AbstractPostProcessor):
    def __init__(self, outdir="./", tgen=None) -> None:
        self.outdir = outdir
        self.tgen = tgen
        self.visited_ref = {}
        self.post_stat: PostStat = None

    def get_trace_generator(self) -> TraceGenerator:
        return self.tgen

    def _get_post_stat(self, dataflow, context, config):
        post_stat = PostStat()
        cost_dict = context.cost_dict
        action_end_time, longest_path = 0, []
        visited = {}
        for action_id in dataflow._roots:
            l, p = self._get_longest(
                action_id, cost_dict, dataflow.dag, visited)
            if l > action_end_time:
                action_end_time = l
                longest_path = p

        post_stat.action_end_time = action_end_time
        l3_end_time = 0
        for die_id in range(config.inst_num.NUM_OF_DIE):
            bw_resource = context.bw_resource_context.l3_dict[die_id]
            last_frame = bw_resource.get_last_valid_frame()
            if last_frame:
                l3_end_time = max(l3_end_time, last_frame.end) - \
                    context.initial_ref

        total_latency = max(action_end_time, l3_end_time)
        post_stat.total_latency = total_latency

        self.longest_path = longest_path
        self.edge_pairs = [(longest_path[i], longest_path[i+1])
                           for i in range(len(longest_path)-1)]

        bw_total_stat = defaultdict(float)
        bw_max_bw = defaultdict(float)
        for (lvl, rw, die_id, cluster_id, esl_port_id, bw_resource) in context.bw_resource_context.get_unique_bw_resource_list():
            key = f"{lvl}_{rw}_bw_util"
            bw_max_bw[key] += bw_resource.max_bw
            for frame in bw_resource.timeline.data:
                bw_total_stat[key] += frame.dur*frame.allocated_bw

            if rw != 'rw':
                key_rw = f"{lvl}_rw_bw_util"
                bw_max_bw[key_rw] += bw_resource.max_bw
                for frame in bw_resource.timeline.data:
                    bw_total_stat[key_rw] += frame.dur*frame.allocated_bw
        for k, v in bw_total_stat.items():
            util = bw_total_stat[k]/total_latency / \
                bw_max_bw[k] if total_latency > 0 else 0

            if not hasattr(post_stat, k):
                logger.warning(
                    "no attr %s(util=%.1f%%) in post_stat, skipped", k, util*100)
                continue
            setattr(post_stat, k, util)

        return post_stat

    def _get_longest(self, action_id, cost_dict, dag,  visited={}):
        stack = [action_id]
        if action_id in visited:
            return visited[action_id]
        curr_latency = cost_dict[action_id].latency
        next_max_latency = 0
        next_child_stack = []
        for child_id in dag.successors(action_id):
            # child = _action_map[child_id]
            latency, child_stack = self._get_longest(
                child_id, cost_dict, dag,  visited)
            if latency > next_max_latency:
                next_max_latency = latency
                next_child_stack = child_stack
        visited[action_id] = curr_latency + \
            next_max_latency, stack
        stack.extend(next_child_stack)
        return curr_latency+next_max_latency, stack

    def _get_ref(self, action_id, dag, cost_dict):
        if action_id in self.visited_ref:
            return self.visited_ref[action_id]

        ref = 0
        parents = dag.predecessors(action_id)
        if parents:
            refs = [self._get_ref(parent_action_id, dag, cost_dict)
                    for parent_action_id in parents]
            ref = max(refs) if refs else 0

        self.visited_ref[action_id] = cost_dict[action_id].latency+ref

        return self.visited_ref[action_id]

    def _prepare_key_frame(self, context, dataflow, config):
        cost_dict = context.cost_dict
        dag = dataflow.dag
        _action_map = dataflow._action_map
        key_frame_per_cluster = defaultdict(set)
        key_frame_global = set()
        for id, cost_book in cost_dict.items():
            if id not in _action_map:
                continue
            action = _action_map[id]
            action: DataflowAction

            x, dx = (self._get_ref(id, dag, cost_dict) -
                     cost_book.latency, cost_book.latency)

            die_id = action.get_die_id()
            cluster_id = action.get_cluster_id()

            key_frame_global.add(x)
            key_frame_global.add(x+dx)
            key_frame_per_cluster[(die_id, cluster_id)].add(x)
            key_frame_per_cluster[(die_id, cluster_id)].add(x+dx)

        key_frame_global_dict = defaultdict(list)
        key_frame_per_cluster_dict = defaultdict(lambda: defaultdict(list))
        for id, cost_book in cost_dict.items():
            if id not in _action_map:
                continue
            action = _action_map[id]
            action: DataflowAction

            x, dx = (self._get_ref(id, dag, cost_dict) -
                     cost_book.latency, cost_book.latency)

            for key_frame in sorted(key_frame_global):
                if x <= key_frame < x+dx:
                    key_frame_global_dict[key_frame].append(cost_book)
            for (die_id, cluster_id), sub_key_frame_list in key_frame_per_cluster.items():
                for key_frame in sorted(sub_key_frame_list):
                    if x <= key_frame < x+dx:
                        key_frame_per_cluster_dict[(die_id, cluster_id)][key_frame].append(
                            cost_book)

        self.key_frame_global_dict = key_frame_global_dict
        self.key_frame_per_cluster_list = key_frame_per_cluster_dict

    def _post_trace(self, context, dataflow, config):
        # trace_arr = []
        # if not hasattr(context, 'track_dict'):
        #    context.track_dict = {}
        # track_dict = context.track_dict

        _action_map = dataflow._action_map
        dag = dataflow.dag
        cost_dict = context.cost_dict
        # key_frame_per_cluster = defaultdict(set)
        # key_frame_global = set()
        # total_per_domain_cluster = defaultdict(lambda: defaultdict(list))
        # action_by_cluster = defaultdict(set)

        # for id, cost_book in cost_dict.items():
        #     if id not in _action_map:
        #         continue
        #     action = _action_map[id]
        #     action: DataflowAction

        #     x, dx = (self._get_ref(id, dag, cost_dict) -
        #              cost_book.latency, cost_book.latency)
        #     die_id = action.get_die_id()
        #     cluster_id = action.get_cluster_id()
        #     args = {
        #         "parent_ids": action.get_parent_ids(),
        #         "child_ids": action.get_child_ids(),
        #         "action_id": action.get_action_id(),
        #     }

        #     for k, v in asdict(cost_book).items():
        #         args[k] = v

        #     key_frame_global.add(x)
        #     key_frame_global.add(x+dx)
        #     key = (die_id, cluster_id)
        #     key_frame_per_cluster[key].add(x)
        #     key_frame_per_cluster[key].add(x+dx)
        #     action_by_cluster[key].add(id)

        # init track dict
        # 1. power
        # track_dict: Dict[Tuple[int, int, str], CounterTrack] = {}
        # die_ids = list(range(config.inst_num.NUM_OF_DIE))
        # cids = list(range(config.inst_num.NUM_OF_CLUSTER))
        # for cls in [PowerSIPDomain, PowerL1Domain, PowerSOCDomain, PowerMemDomain]:
        #     pat = r'Power(.*)Domain'
        #     import re
        #     domain_name = re.match(
        #         pat, cls.__name__).group(1).lower()
        #     for die_id in die_ids:
        #         for cluster_id in cids:
        #             for field in fields(cls):
        #                 field_name = field.name
        #                 _name = f"pw_stat {domain_name}:{field_name}"
        #                 key = (die_id, cluster_id,
        #                        f"{domain_name}:{field_name}")
        #                 track_dict[key] = context.get_cluster_tgen(
        #                     die_id, cluster_id).create_counter_track(_name)

        #             # add cluster total track
        #             key = (die_id, cluster_id, f"{domain_name}:cluster_total")
        #             track_dict[key] = context.get_cluster_tgen(
        #                 die_id, cluster_id).create_counter_track(f"pw_stat {domain_name}:cluster_total")

        #         # global
        #         key = (None, None, f"{domain_name}:total")
        #         _name = f"pw_stat:total {domain_name}:total"
        #         track_dict[key] = context.get_cluster_tgen(
        #             None, None).create_counter_track(_name)
        # # power
        # for (die_id, cluster_id), frame_set in key_frame_per_cluster.items():
        #     action_list = action_by_cluster[(die_id, cluster_id)]
        #     key_frame_dict = {k: [] for k in frame_set}
        #     for id in action_list:
        #         cost_book = cost_dict[id]
        #         x, dx = (self._get_ref(id, dag, cost_dict) -
        #                  cost_book.latency, cost_book.latency)
        #         for key_frame in sorted(frame_set):
        #             if x <= key_frame < x+dx:
        #                 key_frame_dict[key_frame].append(cost_book)

        #     for key_frame in sorted(key_frame_dict):
        #         cost_book_list = key_frame_dict[key_frame]
        #         sum_dict = defaultdict(Counter)

        #         for cost_book in cost_book_list:
        #             cost_book: CostBook
        #             for k, v in asdict(cost_book.power_stat).items():
        #                 if k == 'mem':  # skip mem domain as it is counted in global
        #                     continue
        #                 sum_dict[k].update(v)

        #         args = {}
        #         for pd_domain_name, pd_domain in sum_dict.items():
        #             sum_domain = 0
        #             for pd_sub_domain_name, v in pd_domain.items():
        #                 name = f"{pd_domain_name}:{pd_sub_domain_name}"
        #                 args[name] = round(v/1e12, 3)
        #                 track_dict[die_id, cluster_id,
        #                            name].count(key_frame, round(v/1e12, 3))

        #                 sum_domain += v
        #             name = f"{pd_domain_name}:cluster_total"
        #             args[name] = round(sum_domain/1e12, 3)
        #             track_dict[die_id, cluster_id, name].count(
        #                 key_frame, round(v/1e12, 3))
        #             total_per_domain_cluster[pd_domain_name][cluster_id].append(
        #                 (key_frame, sum_domain))

        # bw_util
        for (lvl, rw, die_id, cluster_id, esl_port_id, bw_resource) in context.bw_resource_context.get_unique_bw_resource_list():
            name = f"bw:{lvl}_{rw}"
            if esl_port_id is not None:
                name += f"_{esl_port_id}"
            last_end = 0
            key_frame = None
            max_bw = bw_resource.max_bw

            track = context.get_cluster_tgen(
                die_id, cluster_id).create_counter_track(f"{name} bw_util")

            for frame in bw_resource.timeline.data:
                if frame.begin > last_end:
                    track.count(last_end, 0)
                track.count(frame.begin, frame.allocated_bw/max_bw)
                last_end = frame.end

        # global total
        # cluster_idx = defaultdict(int)
        # for key_frame in sorted(key_frame_global):
        #     for domain, c in total_per_domain_cluster.items():
        #         domain_sum = defaultdict(float)
        #         key = f"{domain}:total"
        #         for cluster_id, domain_total_list in c.items():
        #             ts, domain_val = domain_total_list[cluster_idx[cluster_id]]

        #             ts_next, domain_val_next = domain_total_list[cluster_idx[cluster_id]+1] if cluster_idx[cluster_id]+1 < len(
        #                 domain_total_list) else (999999, domain_val)
        #             if ts <= key_frame < ts_next:
        #                 domain_sum[key] += round(domain_val/1e12, 3)
        #             else:
        #                 domain_sum[key] += round(domain_val_next/1e12, 3)
        #                 if cluster_idx[cluster_id]+1 < len(domain_total_list):
        #                     cluster_idx[cluster_id] += 1
        #         for k, v in domain_sum.items():
        #             track_dict[(None, None, key)].count(key_frame, v)

        # l3 power
        if context.power_context:
            track = context.get_cluster_tgen(
                None, None).create_counter_track("pw:l3 l3_power")
            for key_frame in context.power_context.l3_power:

                track.count(key_frame.begin, round(key_frame.l3_power/1e12, 3))

            track.count(min(key_frame.end, context.post_stat.total_latency),
                        round(key_frame.l3_power/1e12, 3))

            # edc frame current
            dtu_track = context.get_cluster_tgen(
                None, None).create_counter_track("EDC: DTU [A]")
            dtu_timeline: Timeline[EDCFrame] = context.power_context.dtu_edc
            for key_frame in dtu_timeline.data:
                if key_frame.begin > 100:
                    continue
                dtu_track.count(key_frame.begin, key_frame.current)

            soc_track = context.get_cluster_tgen(
                None, None).create_counter_track("EDC: SOC [A]")
            soc_timeline: Timeline[EDCFrame] = context.power_context.soc_edc
            for key_frame in soc_timeline.data:
                if key_frame.begin > 100:
                    continue
                soc_track.count(key_frame.begin, key_frame.current)

        # update track_dict
        # context.track_dict = track_dict
        logger.info("trace file generated")

    def generate_report(
        self,
        context: BossaNovaContext,
        dataflow: Dataflow,
        config: BossaNovaConfig,
        service_list: List[BaseCostService],
    ):
        # BossaNovaContext类型的对象context记录post_stat生成的信息
        post_stat = self._get_post_stat(dataflow, context, config)
        context.post_stat = post_stat
        # pass out for fusion_report
        self.post_stat = post_stat
        DIE_NUM = config.inst_num.NUM_OF_DIE

        for cost_svc in service_list:
            post_stat.service_report_dict[cost_svc.__class__.__name__] = cost_svc.post_stat(
                context, dataflow)
        self._post_trace(context, dataflow, config)

        report_path = f"{self.outdir}/report.yaml"
        self._prepare_key_frame(context, dataflow, config)

        logger.info("total_latency:%.2fns,len(longest_path):%s",
                    post_stat.total_latency*1e9, len(self.longest_path))

        INTERVAL = config.power.edc_current_interval
        FILTER_GLITCH = config.power.edc_filter_glitch

        # context.power_context.dtu_edc
        if context.power_context:
            def edc_stat(current_timeline: Timeline[EDCFrame]):
                max_current_lvl = 0
                current_dict = defaultdict(lambda: [])
                for key_frame in current_timeline.data:
                    if key_frame.end > 100:
                        continue
                    current = int(key_frame.current // INTERVAL * INTERVAL)
                    max_current_lvl = max(max_current_lvl, current)
                    for lvl in range(0, max_current_lvl+INTERVAL, INTERVAL):
                        if not current_dict[lvl]:
                            current_dict[lvl].append({
                                "begin": key_frame.begin,
                            })
                        if current >= lvl:
                            current_dict[lvl][-1]["end"] = key_frame.end
                        else:
                            if "end" not in current_dict[lvl][-1]:
                                current_dict[lvl][-1]["begin"] = key_frame.end
                            else:
                                current_dict[lvl].append({
                                    "begin": key_frame.end,
                                })
                # filter duration < FILTER_GLITCH and stat count and total duration
                current_dict_agg = {}
                for lvl in range(0, max_current_lvl+INTERVAL, INTERVAL):
                    dur = 0
                    count = 0
                    min_dur = 9999
                    max_dur = 0
                    for current in current_dict[lvl]:
                        if "end" not in current:
                            continue
                        _dur = current["end"]-current["begin"]
                        if _dur < FILTER_GLITCH:
                            continue
                        count += 1
                        dur += _dur
                        min_dur = min(min_dur, _dur)
                        max_dur = max(max_dur, _dur)
                    if count == 0:
                        continue  # 没有符合要求的数据, e.g. 所有的dur<FILTER_GLITCH
                    current_dict_agg[lvl] = {
                        "count": count,
                        "duration(tot)(us)": ts_convert(dur),
                        "duration(min)(us)": ts_convert(min_dur),
                        "duration(max)(us)": ts_convert(max_dur),
                        "duration(avg)(us)": ts_convert(dur/count if count > 0 else 0),
                        "current(A)": lvl,
                    }

                return current_dict_agg
            dtu_edc_stat = edc_stat(context.power_context.dtu_edc)
            soc_edc_stat = edc_stat(context.power_context.soc_edc)

            dtu_edc_report = [dtu_edc_stat[lvl]
                              for lvl in sorted(dtu_edc_stat.keys(), reverse=True)[:5]]
            soc_edc_report = [soc_edc_stat[lvl]
                              for lvl in sorted(soc_edc_stat.keys(), reverse=True)[:5]]
            post_stat.edc.dtu_edc_report = dtu_edc_report
            post_stat.edc.soc_edc_report = soc_edc_report

        report = asdict(post_stat)
        report_str = yaml.dump(report, sort_keys=False, indent=2)

        Path(report_path).write_text(report_str)
        logger.debug(report)
        logger.info("report generated at %s", report_path)
        return report


class FusionPostProcessor:
    '''
    record last post_stat and merge it with current one
    '''

    def __init__(self, outdir="./") -> None:
        self.outdir = outdir

    def fusion_post_stats(self, post_stats: List[PostStat]) -> PostStat:
        if not post_stats:
            return PostStat()  # 返回一个默认的 PostStat
        # 过滤掉 None 的情况
        post_stats = [ps for ps in post_stats if ps is not None]
        # 如果过滤后为空，返回一个默认的 PostStat 实例
        if not post_stats:
            return PostStat()
        # TODO: need to sync with the latest version of post_stat
        total_latency = sum(
            post_stat.total_latency for post_stat in post_stats)
        action_end_time = sum(
            post_stat.action_end_time for post_stat in post_stats)

        def w_avg(attr: str) -> float:
            if total_latency == 0:
                raise ValueError("Total latency is zero, cannot compute weighted average.")
            return sum(getattr(post_stat, attr) * post_stat.total_latency for post_stat in post_stats) / total_latency
        
        core_util = w_avg('core_util')
        l3_rw_bw_util = w_avg('l3_rw_bw_util')
        sic_io_r_bw_util = w_avg('sic_io_r_bw_util')
        sic_io_w_bw_util = w_avg('sic_io_w_bw_util')
        sic_io_rw_bw_util = w_avg('sic_io_rw_bw_util')
        esl_bw_util = w_avg('esl_bw_util')
        d2d_tx_rw_bw_util = w_avg('d2d_tx_rw_bw_util')

        # merge edc
        edc = self.merge_edc_stats([post_stat.edc for post_stat in post_stats])
        # merge service report
        service_report_dict = self.merge_service_report_dicts(
            [post_stat.service_report_dict for post_stat in post_stats])

        return PostStat(
            total_latency=total_latency,
            action_end_time=action_end_time,
            core_util=core_util,
            l3_rw_bw_util=l3_rw_bw_util,
            sic_io_r_bw_util=sic_io_r_bw_util,
            sic_io_w_bw_util=sic_io_w_bw_util,
            sic_io_rw_bw_util=sic_io_rw_bw_util,
            esl_bw_util=esl_bw_util,
            service_report_dict=service_report_dict,
            edc=edc,
            d2d_tx_rw_bw_util=d2d_tx_rw_bw_util
        )
    
    def fusion_post_stats_list(self, fus_post_stats: List[PostStat], caseinfo_post_stats: List[PostStat]) -> List[PostStat]:
        """
        Fusion multiple lists of post_stats for different GCUs
        Input: 
            arg0: List[self.post_stat0, ...],
            arg1: List[caseinfo.post_stat0, ...]

        """
        num_gcus = len(fus_post_stats)  # Get GCU count from first list
        
        # Fusion post_stats for each GCU
        for gcu_id in range(num_gcus):
            fus_post_stats[gcu_id] = self.fusion_post_stats([fus_post_stats[gcu_id], caseinfo_post_stats[gcu_id]])
                
        return fus_post_stats

    @staticmethod
    def merge_service_report_dicts(service_report_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
            merged_service_report = {}
            last_cache_service_report = None
            for service_report_dict in service_report_dicts:
                for service_name, metrics in service_report_dict.items():
                    if service_name in ["CacheCostService", "ParallelCacheCostService"]:
                        # 记录最后一个 CacheCostService 的报告
                        last_cache_service_report = service_report_dict
                    else:
                        service_metrics = merged_service_report.setdefault(service_name, {})
                        for metric_key, metric_value in metrics.items():
                            try:
                                metric_value_num = float(str(metric_value).replace(',', ''))
                            except ValueError:
                                continue
                            if metric_value_num != 0:
                                service_metrics[metric_key] = service_metrics.get(metric_key, 0) + metric_value_num
                            else:
                                service_metrics.setdefault(metric_key, 0)
            if last_cache_service_report:
                if "CacheCostService" in last_cache_service_report:
                    merged_service_report["CacheCostService"] = last_cache_service_report["CacheCostService"]
                if "ParallelCacheCostService" in last_cache_service_report:
                    merged_service_report["ParallelCacheCostService"] = last_cache_service_report["ParallelCacheCostService"]

            return merged_service_report

    @staticmethod
    def merge_edc_stats(edc_stats: List[EDCStat]) -> EDCStat:
        def merge_reports(reports: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            merged_report = []
            for report in reports:
                for entry in report:
                    if entry is None:
                        continue
                    current = entry["current(A)"]
                    existing_entry = next(
                        (e for e in merged_report if e["current(A)"] == current), None)
                    if existing_entry:
                        existing_entry["count"] += entry["count"]
                        existing_entry["duration(tot)(us)"] += entry["duration(tot)(us)"]
                        existing_entry["duration(max)(us)"] = max(
                            existing_entry["duration(max)(us)"], entry["duration(max)(us)"])
                        existing_entry["duration(min)(us)"] = min(
                            existing_entry["duration(min)(us)"], entry["duration(min)(us)"])
                        existing_entry["duration(avg)(us)"] = (
                            existing_entry["duration(tot)(us)"] / existing_entry["count"])
                    else:
                        merged_report.append(entry.copy())
            return merged_report

        def get_top_currents(report: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
            return sorted(report, key=lambda x: x["current(A)"], reverse=True)[:top_n]

        dtu_edc_reports = [edc.dtu_edc_report for edc in edc_stats]
        soc_edc_reports = [edc.soc_edc_report for edc in edc_stats]

        merged_dtu_edc_report = merge_reports(dtu_edc_reports)
        merged_soc_edc_report = merge_reports(soc_edc_reports)

        top_dtu_edc_report = get_top_currents(merged_dtu_edc_report, 5)

        # 累加 edc_total_latency
        edc_total_latency = sum(edc.edc_total_latency for edc in edc_stats)

        # 累加 edc_acc_dict
        edc_acc_dict = {}
        for edc in edc_stats:
            if edc.edc_acc_dict is not None:
                for key, value in edc.edc_acc_dict.items():
                    edc_acc_dict[key] = edc_acc_dict.get(key, 0) + value

        # 求均值 edc_incr_percent
        edc_incr_percent = sum(
            edc.edc_incr_percent for edc in edc_stats) / len(edc_stats) if edc_stats else 0

        return EDCStat(
            dtu_edc_report=top_dtu_edc_report,
            soc_edc_report=merged_soc_edc_report,
            edc_total_latency=edc_total_latency,
            edc_acc_dict=edc_acc_dict,
            edc_incr_percent=edc_incr_percent
        )

    def generate_fus_reports(self, post_stats):
        """Generate fusion reports for multiple GCUs"""
        for gcu_id, post_stat in enumerate(post_stats):
            if post_stat is not None:
                fus_report = asdict(post_stat)
                fus_report_str = yaml.dump(fus_report, sort_keys=False, indent=2)
                
                fus_report_dir = Path(self.outdir) / "fus_report"
                fus_report_dir.mkdir(parents=True, exist_ok=True)

                fus_report_path = f"{self.outdir}/fus_report/fus_report_gcu{gcu_id}.yaml"
                Path(fus_report_path).write_text(fus_report_str)
                logger.debug(f"GCU{gcu_id} fusion report: {fus_report}")
                logger.info("report generated at %s", fus_report_path)
