import functools
from typing import Dict

from nova_platform.base_model import AddrDomain as AD, DataflowActionMemoryStat, DataflowActionType
from nova_platform.config import BWEle, BWFile, BossaNovaConfig, FreqConfig, FreqDomain
from nova_platform.cost_service.compute.base_compute_model import CostBook, BossaNovaContext, DataflowAction, BWFrame, BWMode, BWResource
from nova_platform.cost_service.cache.base_cache_model import CacheStat
import math

import logging

from nova_platform.base_model import BaseESLSwitch
logger = logging.getLogger(__name__)

seq = 0


class DataTransportService():
    def __init__(self, config: BossaNovaConfig, esl_switch: BaseESLSwitch):
        self.config = config
        self.esl_switch = esl_switch

    def _compute_competition_latency_wrapper(self, resource_ctx: BWResource, data_size, cluster_id, bw, freq, ref):
        if resource_ctx:
            resource_ctx._lock.acquire()
            res = self._compute_competition_latency(
                resource_ctx, data_size, cluster_id, bw, freq, ref)
            resource_ctx._lock.release()
        else:
            res = self._compute_competition_latency(
                resource_ctx, data_size, cluster_id, bw, freq, ref)
        return res
        # assume default frame [0,MAX_TIME)
        # frames are continue without any gap

    def _compute_competition_latency(self, resource_ctx: BWResource, data_size, cluster_id, bw, freq, ref):
        if not resource_ctx:
            return data_size/bw/freq
        # look for left_frame, ref, right_frame
        # frame_index, frame = resource_ctx.timeline.get_frame(ref)
        frame_index, frame = resource_ctx.get_frame(ref)

        def get_allocatable_bw(frame: BWFrame):
            return min(resource_ctx.max_bw-frame.allocated_bw, bw)
        allocateable_bw = get_allocatable_bw(frame)
        affordable_data_size = int(
            allocateable_bw*(frame.end-ref)*freq)  # need review precision

        if ref > frame.begin:
            right_frame = BWFrame(ref, frame.end, frame.allocated_bw)
            if ref == frame.end:
                pass
            resource_ctx.timeline.data.insert(frame_index+1, right_frame)
            dur = ref-frame.begin
            frame.end = ref
            latency = self._compute_competition_latency(
                resource_ctx, data_size, cluster_id, bw, freq, ref)
            logger.debug(
                "%02d, case   I, ref > frame.begin, split, nested call, current frame=%s, new ref=%.3E", cluster_id, frame, ref)
        elif allocateable_bw == 0:
            # case I : get_allocatable_bw==0
            dur = frame.end-ref
            ref = frame.end
            logger.debug(
                "%02d, case  II, current frame allocateable_bw==0, nested call by set ref=frame.end, current begin=%.3E, new ref=%.3E", cluster_id, frame.begin, ref)
            latency = dur + \
                self._compute_competition_latency(
                    resource_ctx, data_size, cluster_id, bw, freq, ref)
        elif data_size <= affordable_data_size:
            # case II: current frame can afford all data transportation
            dur = data_size/allocateable_bw/freq
            if data_size < affordable_data_size:
                right_frame = BWFrame(ref+dur,
                                      frame.end, frame.allocated_bw)
                if ref+dur == frame.end:
                    pass
                resource_ctx.timeline.data.insert(frame_index+1, right_frame)
                frame.end = right_frame.begin
                logger.debug(
                    "%02d, case III, split frame left=%s, right=%s", cluster_id, frame, right_frame)
            frame.allocated_bw += allocateable_bw
            latency = dur
            logger.debug(
                "%02d, case  IV, estimate_lat <= frame.dur, current frame=%s", cluster_id, frame)
        else:
            # case V: allocate partial data in current frame and nested call remain
            frame.allocated_bw += allocateable_bw
            remaining_data_size = data_size-int(allocateable_bw*frame.dur*freq)
            ref += frame.dur
            logger.debug(
                "%02d, case V, estimate_lat > frame.dur, allocate current frame and nested call remaining data, new ref=%.3E, curr=%s, remain=%d", cluster_id, ref, frame, remaining_data_size)
            latency = frame.dur
            if remaining_data_size > 0:
                latency += self._compute_competition_latency(
                    resource_ctx, remaining_data_size, cluster_id, bw, freq, ref)

        return latency

    def _path_latency(self, src, dst, freq_config: FreqConfig):
        config = self.config
        bw_file: BWFile = getattr(config.bw, src.lower())
        bw_ele: BWEle = getattr(bw_file, dst.lower())
        freq = freq_config.get_freq(bw_file.freq_domain)  # GHz=>Hz

        pre_lat, post_lat = bw_ele.pre_latency/freq, bw_ele.post_latency/freq
        return pre_lat, post_lat

    def _get_route(self, master: DataflowActionType, src: AD, dst: AD):
        route_map = {
            (DataflowActionType.CDTE, AD.L0, AD.L3): ["L0", "L1C", "SIC_IO", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.CDTE, AD.L0, AD.L3_FAR): ["L0", "L1C", "SIC_IO", "NOC", "LLC_FAR", "MC", "L3"],
            (DataflowActionType.CDTE, AD.SHARED, AD.L3): ["SHARED", "SIC_IO", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.CDTE, AD.SHARED, AD.L3_FAR): ["SHARED", "SIC_IO", "NOC", "LLC_FAR", "MC", "L3"],
            (DataflowActionType.CDTE, AD.LOCAL, AD.L3): ["LOCAL", "SIC_IO", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.CDTE, AD.LOCAL, AD.L3_FAR): ["LOCAL", "SIC_IO", "NOC", "LLC_FAR", "MC", "L3"],
            (DataflowActionType.CDTE, AD.L3, AD.L3): ["SIC_IO", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.CDTE, AD.L3, AD.L3_FAR): ["SIC_IO", "NOC", "LLC_FAR", "MC", "L3"],
            (DataflowActionType.XPU, AD.L0, AD.L3): ["L0", "SIC_IO", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.XPU, AD.L0, AD.L3_FAR): ["L0", "SIC_IO", "NOC", "LLC_FAR", "MC", "L3"],
            (DataflowActionType.XPU, AD.L0, AD.LOCAL): ["L0", "LOCAL"],
            (DataflowActionType.XPU, AD.L0, AD.SHARED): ["L0", "SHARED"],
            (DataflowActionType.ESL, AD.ESL, AD.L3): ["ESL", "NOC", "LLC", "MC", "L3"],
            (DataflowActionType.ESL, AD.ESL, AD.L3_FAR): ["ESL", "NOC", "LLC_FAR", "MC", "L3"],
        }
        return route_map.get((master, src, dst))

    def _get_bw(self, master: DataflowActionType, src, dst, rw, last_bw_per_second, bw_resource: BWResource | None, freq_cfg: FreqConfig):
        bw_dict = self.config.bw
        bw_file: BWFile = getattr(bw_dict, src.lower())
        bw_ele: BWEle = getattr(bw_file, dst.lower())
        freq = freq_cfg.get_freq(bw_file.freq_domain)
        # bw_Bps = bw * freq

        last_bw = int(last_bw_per_second/freq)
        if bw_resource and bw_resource.mode == BWMode.BANDWIDTH:
            bw = last_bw
            logger.debug("bw bandwidth mode, last bw/s=%.2f, bw=%d",
                         last_bw_per_second, bw)
        else:
            bw = bw_ele.get_bw(freq)
            if rw == 'w':
                # TODO: check rw_ratio apply to private or global, now assume private
                rw_ratio = bw_ele.rw_ratio
                bw = bw/rw_ratio
                logger.debug("bw ratio %d, bw= %.2f", rw_ratio, bw)

            # bw = 128
            if master == DataflowActionType.CDTE:
                # TODO: workaround for SPMD _bw*4
                SPMD = self.config.dte.THREAD_NUMBER
                bw = bw*SPMD
                logger.debug("bw SPMD %d, bw= %.2f", SPMD, bw)
            bw = min(bw, last_bw)

        return bw, freq

    # @functools.lru_cache
    def _compute_leading(self, master, src, dst, freq_cfg):
        router = self._get_route(master, src, dst)
        if router is None:
            raise ValueError(
                f"Invalid route: master={master}, src={src}, dst={dst}")
        leading_latency_group = []
        for i in range(len(router)-1):
            _src = router[i]
            _dst = router[i+1]
            # leading
            pre_lat, post_lat = self._path_latency(_src, _dst, freq_cfg)
            leading_latency_group.append((_src, _dst, pre_lat+post_lat))

        return leading_latency_group

    def get_edc_freq_by_src(self, freq_cfg: FreqConfig, src):
        freq_domain_map = {
            "L0":     freq_cfg.CORE,
            "L1C":    freq_cfg.CORE,
            "LOCAL":  freq_cfg.CORE,
            "SHARED": freq_cfg.CORE,
            "SIC_IO": freq_cfg.INTERCONNECT,
            "NOC":    freq_cfg.INTERCONNECT,
            "LLC":    freq_cfg.LLC,
            "LLC_FAR": freq_cfg.LLC,
            "MC":     freq_cfg.MC,
            "L3":     freq_cfg.L3,
        }
        return freq_domain_map[src.upper()]*1e9

    def _compute_sub_path(self, mem_stat: DataflowActionMemoryStat, master, route, index, count, last_bw_per_second, freq_cfg: FreqConfig, die_id, cluster_id, rw, hit_stat_dict: Dict[str, CacheStat], context: BossaNovaContext, ref, latency_stat):
        src = route[index]
        dst = route[index+1]
        pre_lat, post_lat = self._path_latency(src, dst, freq_cfg)
        pre_ref = ref+pre_lat

        resource_ctx = context.bw_resource_context.get_bw_resource(
            src.lower(), dst.lower(), die_id, cluster_id, mem_stat, rw)
        bw, freq = self._get_bw(
            master, src, dst, rw, last_bw_per_second, resource_ctx, freq_cfg)  # bw: byte/cycle
        _last_bw_per_second = bw*freq
        end_ts = 0
        _post_ref = pre_ref
        if index < len(route)-1:
            if dst.upper() == "LLC_FAR":
                _dst = "LLC"
            else:
                _dst = dst
            is_cache_layer = _dst.upper() in hit_stat_dict
            if is_cache_layer:
                cache_stat = hit_stat_dict.get(_dst)
                if rw == 'w':
                    # hit_count = cache_stat.write_hit_count
                    write_hit_rate = cache_stat.write_hit_rate
                    hit_count = min(int(count*write_hit_rate), count)
                else:
                    # hit_count = cache_stat.read_hit_count
                    # check fixed hit_rate
                    read_hit_rate = cache_stat.read_hit_rate
                    # logging.info(hit_rate)
                    hit_count = min(int(count*read_hit_rate), count)
            else:
                hit_count = 0
                hit_rate = 0

            next_count = count - hit_count
            if next_count > 0 and index < len(route)-2:
                _die_id = die_id if route[index+1][-4:] != '_FAR' else (
                    die_id+1) % self.config.inst_num.NUM_OF_DIE
                _ref, _pre_ref, _post_ref, _end_ts, _count = yield from self._compute_sub_path(
                    mem_stat, master, route, index+1, next_count, _last_bw_per_second, freq_cfg, _die_id, cluster_id, rw, hit_stat_dict, context, pre_ref, latency_stat)
                end_ts = max(end_ts, _end_ts)

        data_start_ts = 99999
        data_end_ts = 0

        if rw == 'r':
            # no sub path
            post_ref = _post_ref + post_lat  # compute backward latency for read
        else:
            post_ref = pre_ref

        data_start_ts = min(data_start_ts, post_ref)
        global seq
        seq += 1
        data_lat = self._compute_competition_latency_wrapper(
            resource_ctx, count, cluster_id, bw, freq, post_ref)
        _stat = {}

        def __compute_esl(ref, src_gcu_id, tar_gcu_id, rw, count, memory_list):
            lat, detail = yield from self.esl_switch.send(
                ref,
                src_gcu_id=src_gcu_id,
                tar_gcu_id=tar_gcu_id,
                rw=rw, data_size=count, memory_list=memory_list)
            return lat, detail

        if src == 'ESL' and rw == 'r':
            _lat, esl_detail = yield from __compute_esl(post_ref, mem_stat.src_gcu_id, mem_stat.tar_gcu_id, 'w', count, mem_stat.remote_target_mem_access_list)
            _stat['raw_data_lat'] = data_lat
            _stat['esl_data_lat'] = _lat
            data_lat = max(_lat, data_lat)
            _stat['esl_detail'] = esl_detail

        end_ts = max(end_ts, post_ref+data_lat)
        data_end_ts = max(data_end_ts, post_ref+data_lat)

        if resource_ctx:  # TODO: consider move below code to compute_competition_latency
            if rw == 'r':
                resource_ctx.acc_read += count
            else:
                resource_ctx.acc_write += count

        _stat.update(
            **{"src": src,
               "dst": dst,
               "ref": ref,
               "data_lat": data_end_ts-data_start_ts,
               "data_start_ref": data_start_ts,
               "data_end_ref": data_end_ts,
               "pre_ref": pre_ref,
               "post_ref": post_ref,
               "end_ts": end_ts,
               "count": count,
               "last_bw_per_second": last_bw_per_second,
               "_last_bw_per_second": _last_bw_per_second,
               "required_bw": bw*freq,
               "actual_bw": count/data_lat,
               "seq": seq,
               }
        )

        latency_stat.append(_stat)
        return ref, pre_ref, post_ref, end_ts, count

    def compute_latency(
        self,
        context: BossaNovaContext,
        mem_stat: DataflowActionMemoryStat,
        die_id,
        cluster_id,
        ref: float,
        hit_stat_dict: Dict[str, CacheStat],
        freq_cfg: FreqConfig,
        bw_factor=1
    ):
        master = mem_stat.master

        src, dst, rw, total_count = mem_stat.src, mem_stat.dst, mem_stat.rw, mem_stat.total_count

        leading_latency_group = self._compute_leading(
            master, src, dst, freq_cfg)
        total_leading_latency = 0
        total_latency = 0
        count = total_count
        router = self._get_route(mem_stat.master, src, dst)
        logger.debug(f"({src},{dst},{rw})=>{router}")
        # calculate and accumlate latency for each sub path
        # e.g. ["L0","L1C","LLC","MC"]=> ("L0","L1C"),("L1C","LLC"),("LLC","MC")
        # bw_min = 9999
        # last_bw_per_second = 0

        # master_freq = self.freq_domain_map[src.upper()]
        master_file: BWFile = getattr(
            self.config.bw, master.value.lower())  # TODO: need review
        master_freq = freq_cfg.get_freq(master_file.freq_domain)
        _src, _dst, _ = leading_latency_group[0]
        bw_file: BWFile = getattr(self.config.bw, _src.lower())
        bw_ele: BWEle = getattr(bw_file, _dst.lower())
        if rw == 'w':
            spmd = self.config.dte.THREAD_NUMBER
            # bw_dict = getattr(self.config.bw, master.value.lower())
            bw_per_cycle = bw_ele.get_bw(
                freq_cfg.get_freq(bw_file.freq_domain))
            last_bytes_per_cycle = bw_per_cycle if master == DataflowActionType.XPU else bw_per_cycle*spmd
        else:
            # TODO: how to fix?
            otsd = master_file.otsd
            _count = total_count
            round_chip_latency = 0
            for _src, _dst, leading_latency in leading_latency_group:
                is_cache_layer = _dst.upper() in hit_stat_dict
                hit_rate = 0
                if is_cache_layer:
                    cache_stat = hit_stat_dict.get(_dst)
                    hit_rate = cache_stat.read_hit_rate
                round_chip_latency += _count*leading_latency  # count*[s]
                _hit_count = min(int(_count*hit_rate), count)
                _count -= _hit_count
                if _count == 0:
                    break
            round_chip_latency = (round_chip_latency /
                                  total_count)*master_freq  # [s]*[cycle/s]=[cycle]
            # 256 * 128 / 5xx # 128：TODO: 取src到dst上最小的bw_per_cycle
            last_bytes_per_cycle = otsd*128/round_chip_latency
            bw_per_cycle = bw_ele.get_bw(
                freq_cfg.get_freq(bw_file.freq_domain))
            last_bytes_per_cycle = min(last_bytes_per_cycle, bw_per_cycle)
            logger.debug("round_chip_latency=%f, effective bpc=%f",
                         round_chip_latency, last_bytes_per_cycle)
        last_bw_per_second = last_bytes_per_cycle*master_freq*bw_factor

        route = self._get_route(master, src, dst)
        latency_stat = []
        start_ts, pre_ref, post_ref, end_ts, _count = yield from self._compute_sub_path(
            mem_stat, master, route, 0, count, last_bw_per_second, freq_cfg,
            die_id, cluster_id, rw, hit_stat_dict, context, ref, latency_stat
        )
        total_leading_latency = post_ref-start_ts
        total_latency = end_ts-start_ts

        mem_stat.is_done = True
        return total_latency, total_leading_latency, latency_stat
