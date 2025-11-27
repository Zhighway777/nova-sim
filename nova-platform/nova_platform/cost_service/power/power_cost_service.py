from collections import defaultdict
from dataclasses import asdict
import dataclasses
import itertools
import time
from typing import Callable, Generator
from nova_platform.base_model import MAX_TIME, AddrDomain as AD, BaseActionStat, DataflowActionComputeStat, DataflowActionMemoryStat, DataflowActionType as DT
from nova_platform.config import BossaNovaConfig, FreqConfig, PowerLibConfig
from nova_platform.cost_service.compute.base_compute_model import BaseCostService, BossaNovaContext, DataflowAction, BWResource, EDCFrame, PowerContext, Timeline
from nova_platform.cost_service.cache.base_cache_model import CacheStat
from nova_platform.cost_service.power.base_power_model import BasePowerDomain, PowerL1Domain, PowerField, PowerFrame, PowerL3Frame, PowerStat
from nova_platform.dataflow.dataflow import Dataflow
import logging
logger = logging.getLogger(__file__)


def duration(func):
    def wrap(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        print(func.__name__, f"{(end-start)/1e6:.2f}ms")
        return result
    return wrap


@duration
def merge_timeline(pairs, agg_fun, init_fac):
    ts_map = defaultdict(list)
    for key, timeline in pairs:
        for frame in timeline.data:
            ts_map[frame.begin].append((0, key, frame))
            ts_map[frame.end].append((1, key, frame))

    ts_list = sorted(ts_map.keys())
    ts_res = [init_fac() for _ in range(len(ts_list))]
    ts_len = len(ts_list)
    progress = 0
    for i, ts in enumerate(ts_list):
        frame_list = ts_map[ts]
        for flag, key, frame in frame_list:
            if flag == 1:
                continue
            e = frame.end
            val = agg_fun(key, frame)
            for offset, ts_end in enumerate(ts_list[i:]):
                if ts_end < e:
                    for k, _v in enumerate(val):
                        ts_res[i+offset][k] += _v
                else:
                    break
        _progress = int((i+1)/ts_len*100)
        if _progress > progress:
            progress = _progress
            logger.info("merge_timeline progress %d%%" % progress)
    return ts_list, ts_res


# fmt:off
class PowerCostService(BaseCostService):
    def __init__(self, config: BossaNovaConfig):
        super().__init__(config)

    def init_context(self):
        die_num=self.config.inst_num.NUM_OF_DIE
        cluster_num=self.config.inst_num.NUM_OF_CLUSTER
        sip_num=self.config.inst_num.NUM_OF_CORE_PER_CLUSTER
        sip_power_timeline_dict={}
        l1_power_timeline_dict={}
        for die_id,cluster_id,sip_id in itertools.product(
            range(die_num),
            range(cluster_num),
            range(sip_num),
        ):
            key=die_id,cluster_id,sip_id
            sip_power_timeline_dict[key]=Timeline([PowerFrame()])
            l1_power_timeline_dict[key]=Timeline([PowerFrame()])


        power_context=PowerContext(
            sip_power_timeline_dict=sip_power_timeline_dict,
            l1_power_timeline_dict=l1_power_timeline_dict,
            # soc_power_timeline_dict={die_id:Timeline([PowerFrame()]) for die_id in range(die_num)}
        )
        return power_context

    def _compute_domain_power(self,power_domain_stat, act_stat: DataflowActionComputeStat | DataflowActionMemoryStat):
        sum_power = 0
        for f in dataclasses.fields(power_domain_stat):
            f: PowerField
            if f.pre_check(act_stat):
                fun = f.formula
                val=fun(act_stat, self.config.power) if act_stat.latency else 0
                setattr(power_domain_stat,f.name,val)
                sum_power += val
        sum_power /= 1e12  # [pW] -> [W]
        return sum_power

    def process(self, action: DataflowAction, context: BossaNovaContext, ref: float) -> Generator[bool, BaseActionStat, None]:
        die_id=action.get_die_id()
        cluster_id=action.get_cluster_id()
        engine_id = action.get_local_engine_id() if action.get_action_type()!=DT.CDTE else 0
        action_type = action.get_action_type()
        key=(die_id,cluster_id,engine_id)
        cost_book=context.get_cost_book(action)
        cost_book.power_stat=PowerStat()
        sub_power_stat=[]
        while True:
            act_stat = yield
            if not act_stat:
                break
            power_stat=PowerStat()
            sub_power_stat.append((power_stat,act_stat.latency))
            act_stat.power_stat=power_stat

            # sip
            sum_sip_power=self._compute_domain_power(power_stat.sip,act_stat)
            # volt = self.config.power.dtu.voltage
            context.power_context.sip_power_timeline_dict[key].insert(PowerFrame(
                ref+act_stat.relative_ts,
                ref+act_stat.relative_ts+act_stat.latency,
                power=sum_sip_power
            ))

            # l1
            sum_l1_power=self._compute_domain_power(power_stat.l1,act_stat)
            # volt = self.config.power.dtu.voltage
            context.power_context.l1_power_timeline_dict[key].insert(PowerFrame(
                ref+act_stat.relative_ts,
                ref+act_stat.relative_ts+act_stat.latency,
                power=sum_l1_power
            ))

            # soc
            sum_soc_energy=self._compute_domain_power(power_stat.soc,act_stat)
            # context.power_context.soc_power_timeline_dict[die_id].insert(PowerFrame(
            #     ref+act_stat.relative_ts,
            #     ref+act_stat.relative_ts+act_stat.latency,
            #     power=sum_soc_power
            # )) 

            soc_volt = self.config.power.soc.voltage
            soc_current = sum_soc_energy/soc_volt
            context.power_context.soc_edc.insert(EDCFrame(
                ref+act_stat.relative_ts,
                ref+act_stat.relative_ts+act_stat.latency,
                soc_current
            ))

        tot=PowerStat()
        if cost_book.latency:
            for sub,lat in sub_power_stat:
                sub: PowerStat
                tot+=sub*lat
            tot=tot*(1/(cost_book.latency*1e12))
        cost_book.power_stat=tot

    def _derive_sip_power_timeline(self, context: BossaNovaContext):
        sip_idle_power=self.config.power.sip.idle_power/1e12
        sip_leakage_power=self.config.power.sip.leakage_power/1e12
        sip_plc_power=self.config.power.sip.plc_power/1e12

        DIE_NUM=self.config.inst_num.NUM_OF_DIE
        sip_idle_energy=[0]*DIE_NUM
        sip_active_energy=[0]*DIE_NUM
        sip_plc_energy=[0]*DIE_NUM
        sip_leakage_energy=[0]*DIE_NUM
        total_latency=context.post_stat.total_latency
        total_items=len(context.power_context.sip_power_timeline_dict)
        item_count=0
        tot_sip_dur=0
        def agg_fun(key,pw_frame):
            nonlocal tot_sip_dur
            die_id,cluster_id,sip_id=key
            # sip 
            sip_power=pw_frame.power
            sip_dur=pw_frame.dur if pw_frame.end<MAX_TIME else total_latency-pw_frame.begin
            if sip_power:
                # active
                sip_active_energy[die_id]+=sip_power*sip_dur
                sip_plc_energy[die_id]+=sip_plc_power*sip_dur
                sip_power+=sip_plc_power+sip_leakage_power
                tot_sip_dur+=sip_dur
            else:
                sip_idle_energy[die_id]+=sip_idle_power*sip_dur
                sip_power+=sip_idle_power+sip_leakage_power

            sip_leakage_energy[die_id]+=sip_leakage_power*sip_dur
            current=sip_power/self.config.power.sip.voltage
            return sip_power, current

        logger.info("start to merge sip power timeline")
        ts_list,ts_res=merge_timeline(context.power_context.sip_power_timeline_dict.items(),agg_fun,lambda:[0,0])
        dtu_power_timeline_data_part1=[PowerFrame(ts_list[i],ts_list[i+1],ts_res[i][0]) for i in range(len(ts_list)-1)]
        dtu_edc_timeline_data_part1=[EDCFrame(ts_list[i],ts_list[i+1],ts_res[i][1]) for i in range(len(ts_list)-1)]

        sip_util=tot_sip_dur/total_latency/len(context.power_context.sip_power_timeline_dict)
        # update edc
        # for dtu_frame in context.power_context.dtu_power_timeline.data:
        #     volt = self.config.power.dtu.voltage
        #     current = dtu_frame.power/volt
        #     context.power_context.dtu_edc.insert(EDCFrame(
        #         dtu_frame.begin,
        #         dtu_frame.end,
        #         current
        #     ))
        sip_active_power_tot=[e/total_latency for e in sip_active_energy]
        sip_idle_power_tot=[e/total_latency for e in sip_idle_energy]
        sip_leakage_power_tot=[e/total_latency for e in sip_leakage_energy]
        sip_plc_power_tot=[e/total_latency for e in sip_plc_energy]
        return sip_active_power_tot,sip_idle_power_tot,sip_leakage_power_tot,sip_plc_power_tot,sip_util,dtu_power_timeline_data_part1,dtu_edc_timeline_data_part1

    def _derive_l1_power_timeline(self, context: BossaNovaContext):
        l1_idle_power=self.config.power.l1.idle_power/1e12
        l1_leakage_power=self.config.power.l1.leakage_power/1e12
        DIE_NUM=self.config.inst_num.NUM_OF_DIE

        l1_active_energy=[0]*DIE_NUM
        l1_leakage_energy=[0]*DIE_NUM
        l1_idle_energy=[0]*DIE_NUM
        total_latency=context.post_stat.total_latency
        total_items=len(context.power_context.sip_power_timeline_dict)
        item_count=0
        # with open("l1_power_timeline.timeline","wb") as f:
        #     import pickle
        #     pickle.dump(context.power_context.l1_power_timeline_dict,f)


        def agg_fun(key,pw_frame):
            die_id,cluster_id,sip_id=key
            l1_power=pw_frame.power
            l1_dur=pw_frame.dur if pw_frame.end<MAX_TIME else total_latency-pw_frame.begin
            if l1_power:
                # active
                l1_active_energy[die_id]+=l1_power*l1_dur
                l1_power+=l1_leakage_power
            else:
                l1_idle_energy[die_id]+=l1_idle_power*l1_dur
                l1_power+=l1_idle_power+l1_leakage_power
            l1_leakage_energy[die_id]+=l1_leakage_power*l1_dur

            l1_power/=1e12
            current=l1_power/self.config.power.l1.voltage
            return l1_power,current

        logger.info("start to merge l1 power timeline")

        ts_list,ts_res=merge_timeline(context.power_context.l1_power_timeline_dict.items(),agg_fun,lambda:[0,0])
        dtu_power_timeline_data_part2=[PowerFrame(ts_list[i],ts_list[i+1],ts_res[i][0]) for i in range(len(ts_list)-1)]
        dtu_edc_timeline_data_part2=[EDCFrame(ts_list[i],ts_list[i+1],ts_res[i][1]) for i in range(len(ts_list)-1)]

        l1_active_power_tot=[e/total_latency for e in l1_active_energy]
        l1_idle_power_tot=[e/total_latency for e in l1_idle_energy]
        l1_leakage_power_tot=[e/total_latency for e in l1_leakage_energy]
        return l1_active_power_tot,l1_idle_power_tot,l1_leakage_power_tot, dtu_power_timeline_data_part2,dtu_edc_timeline_data_part2


    def _derive_soc_power_timeline(self, context: BossaNovaContext):
        soc_idle_power=self.config.power.soc.mc_idle_power/1e12
        soc_leakage_power=self.config.power.dtu.leakage_power/1e12
        for key,pw_timeline in context.power_context.soc_power_timeline_dict.items():
            for pw_frame in pw_timeline.data:
                # sip 
                if pw_frame.power:
                    # active
                    pw_frame.power+=xpu_plc_power+soc_leakage_power
                else:
                    pw_frame.power+=xpu_idle_power+soc_leakage_power
                context.power_context.soc_power_timeline.insert(pw_frame) 


    def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):
        # TODO: dual_die
        die_id=0
        mem_bw: BWResource = context.bw_resource_context.get_bw_resource(
            'mc', 'l3', die_id, 0, None, 'r')

        sip_active_power,sip_idle_power,sip_leakage_power,sip_plc_power,sip_util,dtu_pw_part1,dtu_edc_part1=self._derive_sip_power_timeline(context)
        l1_active_power,l1_idle_power,l1_leakage_power,dtu_pw_part2,dtu_edc_part2=self._derive_l1_power_timeline(context)
        # self._derive_soc_power_timeline(context)
        logger.info("sip_active_power=%s",[f"{e/1e12:03.2f}" for e in sip_active_power])
        logger.info("sip_idle_power=%s",[f"{e/1e12:03.2f}" for e in sip_idle_power])
        logger.info("sip_leakage_power=%s",[f"{e/1e12:03.2f}" for e in sip_leakage_power])
        logger.info("sip_plc_power=%s",[f"{e/1e12:03.2f}" for e in sip_plc_power])
        logger.info("l1_active_power=%s",[f"{e/1e12:03.2f}" for e in l1_active_power])
        logger.info("l1_idle_power=%s",[f"{e/1e12:03.2f}" for e in l1_idle_power])
        logger.info("l1_leakage_power=%s",[f"{e/1e12:03.2f}" for e in l1_leakage_power])

        logger.info("start to merge dtu power timeline")
        dtu_pw_dict={
            "part1": Timeline(dtu_pw_part1),
            "part2": Timeline(dtu_pw_part2),
        }
        ts_list,ts_res=merge_timeline(dtu_pw_dict.items(),lambda k,frame:[frame.power],lambda:[0])
        context.power_context.dtu_power_timeline.data=[PowerFrame(ts_list[i],ts_list[i+1],ts_res[i][0]) for i in range(len(ts_list)-1)]

        logger.info("start to merge dtu edc timeline")
        dtu_edc_dict={
            "part1": Timeline(dtu_edc_part1),
            "part2": Timeline(dtu_edc_part2),
        }
        ts_list,ts_res=merge_timeline(dtu_edc_dict.items(),lambda k,frame:[frame.current],lambda:[0])
        context.power_context.dtu_edc.data=[EDCFrame(ts_list[i],ts_list[i+1],ts_res[i][0]) for i in range(len(ts_list)-1)]

        for frame in mem_bw.timeline.data:
            util = frame.allocated_bw/mem_bw.max_bw
            total_power = util*self.config.power.mem.hbm_active_power + \
                (1-util)*self.config.power.mem.hbm_idle_power
            total_power /= 1e12
            context.power_context.l3_power.append(PowerL3Frame(
                frame.begin, frame.end, total_power))

        domain_total_energy = defaultdict(float)
        for id, cost_book in context.cost_dict.items():
            for domain_name, domain in asdict(cost_book.power_stat).items():
                domain_total_energy[domain_name] += sum(domain.values()) * \
                    cost_book.latency
        for frame in context.power_context.l3_power:
            dur = frame.dur if frame.dur < 999 else 0
            domain_total_energy["mem"] += dur*frame.l3_power

        # TODO: need to implement ESL power calculation
        domain_total_energy["esl"] = 0

        # total_avg_power = 0
        for k in list(domain_total_energy):
            energy = domain_total_energy.pop(k)
            domain_total_energy[k+":active_energy(J)"] = energy
            domain_total_energy[k+":active_power(W)"] = energy / \
                context.post_stat.total_latency

        # DTU Domain
        DIE_NUM=self.config.inst_num.NUM_OF_DIE
        for die_id in range(DIE_NUM):
            die_key=f'die:{die_id}'
            domain_total_energy[die_key]={}
            data_dict=domain_total_energy[die_key]
            data_dict["xpu:compute_power(W)"] = sip_active_power[die_id]+sip_plc_power[die_id]+sip_leakage_power[die_id]+sip_idle_power[die_id]
            data_dict["l1:dataflow_power(W)"] = l1_active_power[die_id]+l1_leakage_power[die_id]+l1_idle_power[die_id]
            data_dict["dtu:idle_power(W)"] = sip_idle_power[die_id]+l1_idle_power[die_id]
            data_dict["dtu:leakage_power(W)"] = sip_leakage_power[die_id]+l1_leakage_power[die_id]
            data_dict["dtu:active_power(W)"] = sip_active_power[die_id]+l1_active_power[die_id]
            data_dict["dtu:total_power(W)"] = data_dict["xpu:compute_power(W)"]+data_dict["l1:dataflow_power(W)"]

        domain_total_energy["dtu:idle_power(W)"] = sum(sip_idle_power)+sum(l1_idle_power)
        domain_total_energy["dtu:active_power(W)"] = sum(sip_active_power)+sum(l1_active_power)
        domain_total_energy["dtu:leakage_power(W)"] = sum(sip_leakage_power)+sum(l1_leakage_power)
        domain_total_energy["dtu:total_power(W)"] = sum([domain_total_energy[f'die:{die_id}']["dtu:total_power(W)"] for die_id in range(DIE_NUM)])


        # SOC
        domain_total_energy["soc:idle_power(W)"] = (
            (1-context.post_stat.l3_rw_bw_util) *
            self.config.power.soc.mc_idle_power
            + (1-context.post_stat.sic_io_rw_bw_util) *
            self.config.power.soc.dataflow_idle_power
            + (1-context.post_stat.esl_bw_util) *
            self.config.power.soc.esl_idle_power
        )*DIE_NUM/1e12
        domain_total_energy["soc:leakage_power(W)"] = self.config.power.soc.leakage_power*DIE_NUM/1e12
        domain_total_energy["soc:other_power(W)"] = self.config.power.soc.other_power*DIE_NUM/1e12
        domain_total_energy["soc:total_power(W)"] = (domain_total_energy["soc:active_power(W)"]
                                                     + domain_total_energy["soc:idle_power(W)"]
                                                     + domain_total_energy["soc:leakage_power(W)"]
                                                     + domain_total_energy["soc:other_power(W)"]
                                                     )

        domain_total_energy["mem:idle_power(W)"] = (
            1-context.post_stat.l3_rw_bw_util)*self.config.power.mem.hbm_idle_power/1e12
        domain_total_energy["mem:leakage_power(W)"] = self.config.power.mem.leakage_power/1e12
        domain_total_energy["mem:total_power(W)"] = (domain_total_energy["mem:active_power(W)"]
                                                     + domain_total_energy["mem:idle_power(W)"]
                                                     + domain_total_energy["mem:leakage_power(W)"]
                                                     )

        domain_total_energy["esl:idle_power(W)"] = (
            1-context.post_stat.esl_bw_util)*self.config.power.esl.idle_power/1e12
        domain_total_energy["esl:leakage_power(W)"] = self.config.power.esl.leakage_power/1e12
        domain_total_energy["esl:total_power(W)"] = (domain_total_energy["esl:active_power(W)"]
                                                     + domain_total_energy["esl:idle_power(W)"]
                                                     + domain_total_energy["esl:leakage_power(W)"]
                                                     )

        # d2d
        if DIE_NUM>1:
            domain_total_energy["d2d:active_power(W)"] = (
                context.post_stat.d2d_tx_rw_bw_util*self.config.power.d2d.active_power/1e12
            )
            domain_total_energy["d2d:idle_power(W)"] = (
                1-context.post_stat.d2d_tx_rw_bw_util)*self.config.power.d2d.idle_power/1e12
            domain_total_energy["d2d:leakage_power(W)"] = self.config.power.d2d.leakage_power/1e12
            domain_total_energy["d2d:total_power(W)"] = (domain_total_energy["d2d:active_power(W)"]
                                                        + domain_total_energy["d2d:idle_power(W)"]
                                                        + domain_total_energy["d2d:leakage_power(W)"]
                                                        )     

        domain_total_energy["total:asic_power(W)"] = (
            domain_total_energy["dtu:total_power(W)"]
            + domain_total_energy["soc:total_power(W)"]
            + domain_total_energy["mem:total_power(W)"]
            + domain_total_energy["esl:total_power(W)"]
            + domain_total_energy["d2d:total_power(W)"]
        )
        domain_total_energy["total:board_power(W)"] = domain_total_energy["total:asic_power(W)"] / \
            self.config.power.board_efficiency

        for k in domain_total_energy:
            if "(J)" in k:
                domain_total_energy[k] = round(domain_total_energy[k]/1e12, 6)
        domain_total_energy = sorted(domain_total_energy.items())
        res = dict(domain_total_energy)
        res['sip_util']=sip_util
        # calculate total latency affected by edc
        EDC_LIMIT = context.bw_resource_context.config.power.dtu_edc
        base_freq = context.bw_resource_context.config.freq.CORE
        sum_edc_total_latency = 0
        edc_acc_dict = defaultdict(lambda: 0)
        raw_edc_total_latency=0
        for edc_frame in context.power_context.dtu_edc.data[:-1]:
            edc_dtu_freq = min(EDC_LIMIT/edc_frame.current, 1) * \
                base_freq if edc_frame.current else base_freq  # MHz
            edc_dtu_freq = ((edc_dtu_freq*1000)//50)*50 / 1000
            sum_edc_total_latency += edc_frame.dur*base_freq/edc_dtu_freq if edc_dtu_freq else 0
            raw_edc_total_latency += edc_frame.dur
            edc_acc_dict[edc_dtu_freq] += edc_frame.dur
        context.post_stat.edc.edc_total_latency = sum_edc_total_latency
        context.post_stat.edc.edc_acc_dict = dict(edc_acc_dict)
        if raw_edc_total_latency != 0:
             context.post_stat.edc.edc_incr_percent = (
            sum_edc_total_latency-raw_edc_total_latency)/raw_edc_total_latency*100
        else:
            context.post_stat.edc.edc_incr_percent = 0  # or another appropriate default value
        return res

    def get_edc_freq(self, ref, context: BossaNovaContext) -> FreqConfig:
        base_freq_cfg = context.bw_resource_context.config.freq
        return base_freq_cfg
        # _, edc_frame = context.dtu_edc.get_frame(ref)
        # base_freq = context.bw_resource_context.config.freq.CORE
        # EDC_LIMIT = context.bw_resource_context.config.power.dtu.edc
        # edc_dtu_freq = min(EDC_LIMIT/edc_frame.current, 1) * \
        #     base_freq if edc_frame.current else base_freq  # MHz
        # edc_dtu_freq = ((edc_dtu_freq*1000)//50)*50 / 1000  # GHz
        # if edc_dtu_freq < base_freq:
        #     track = context.get_cluster_tgen(
        #         None).create_track("EDC: DTU [A]")
        #     track.instant(ref, "edc triggered", {
        #         "base_freq": base_freq,
        #         "edc_dtu_freq": edc_dtu_freq,
        #         "edc_dtu_current": edc_frame.current
        #     })

        # logger.debug("edc dtu, ref=%09d, current= %.2f, freq=%.3fGHz",
        #              int(ref*1e9), edc_frame.current, edc_dtu_freq)
        # edc_freq = FreqConfig(
        #     **context.bw_resource_context.config.freq.__dict__)
        # edc_freq.CORE = edc_dtu_freq
        # return edc_freq
