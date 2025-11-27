import functools
import os
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class _DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def update(self, n=1):
            pass

        def close(self):
            pass

        def __iter__(self):
            if self.iterable is None:
                return iter([])
            for item in self.iterable:
                yield item

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def tqdm(iterable=None, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)
import yaml
import subprocess
from dataclasses import asdict
from glob import glob
from pathlib import Path
from types import GeneratorType
from typing import Dict, Generator

from nova_platform.base_model import MAX_TIME, DType, DataflowActionMemoryStat
from nova_platform.config import BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BossaNovaContext, DataflowAction
from nova_platform.cost_service.compute.base_compute_model import BaseCostService
from nova_platform.cost_service.cache.cache_cost_service import CacheCostService
from nova_platform.cost_service.compute.base_compute_model import BWResourceContext
from nova_platform.cost_service.compute.compute_cost_service import ComputeCostService
from nova_platform.cost_service.power.power_cost_service import PowerCostService
from nova_platform.data_visual.post_processor import PostProcessor
from nova_platform.dataflow.action.diag_action import DiagDataflow, DiagDataflowAction, parse_shape_tile_info
from nova_platform.dataflow.action.dte_action import CDTEDesliceAction, CDTEReshapeAction, CDTESliceAction
from nova_platform.dataflow.action.nop_action import XPUNopAction
from nova_platform.dataflow.action.xpu_activation_action import XPUGeluAction, XPUReluAction, XPUSigmoidAction, XPUSiluAction
from nova_platform.dataflow.action.xpu_elementwise_action import XPUAddAction, XPUMulAction
from nova_platform.dataflow.action.xpu_gather_action import XPUGatherAction
from nova_platform.dataflow.action.xpu_scatter_action import XPUScatterAction
from nova_platform.dataflow.action.xpu_gemm_action_2 import XPUGemmAction
from nova_platform.dataflow.action.xpu_layernorm_action import XPULayernormAction
from nova_platform.dataflow.action.xpu_softmax_action import XPUSoftmaxAction
from nova_platform.dataflow.action.xpu_softmax_backward_action import XPUSoftmaxBackwardAction
from nova_platform.dataflow.action.xpu_sdpa_action import XPUSdpaAction
from nova_platform.dataflow.action.xpu_local_gemm_action import XPULocalGemmAction, XPUSharedGemmAction
from nova_platform.dataflow.action.xpu_allreduce_action import XPUAllReduceAction
from nova_platform.dataflow.action.xpu_allgather_action import XPUAllGatherAction
from nova_platform.dataflow.action.xpu_allgather_gemm_action import XPUAllGatherGemmAction
from nova_platform.dataflow.dataflow import Dataflow
from nova_platform.executor.nova_platform_barrier import BARRIER
from nova_platform.base_model import BaseESLSwitch
from nova_platform.executor.nova_platform_event import BossaNovaEvent
from nova_platform.perfetto_protobuf._tgen import TraceGenerator
from nova_platform.utils.config_utils import dict_to_dataclass
from nova_platform.utils.cuda_utils import get_gpu_count
from nova_platform.executor.dataflow_gen import generate_dataflow, DATAFLOW_GENERATOR_MAPPING

import logging

logger = logging.getLogger(__name__)

SUPPORTED_OP_TYPES = [
    "gemm",
    "softmax",
    "softmaxbackward",
    "add",
    "mul",
    "transpose",
    "gather",
    "sigmoid",
    "gelu",
    "silu",
    "relu",
    "layernorm",
    "scatter",
    "sdpa",  # scaled dot product attention
    "allreduce",
    "allgather",
    "allgather_gemm",
]


def get_data_flow_gen_cmd(dataflow_config: dict, arch_config: BossaNovaConfig):
    if dataflow_config["bench_op_type"] not in SUPPORTED_OP_TYPES:
        raise RuntimeError(
            f'Not supported op type {dataflow_config["bench_op_type"]}')
    binary_path = Path(__file__).resolve().parents[2] / "build" / "dataflow_gen"
    sic_num = arch_config.inst_num.NUM_OF_CLUSTER * arch_config.inst_num.NUM_OF_DIE
    l3_mc_bw = arch_config.bw.mc.l3.bw * arch_config.freq.MC * \
        arch_config.inst_num.NUM_OF_DIE * 1e9
    args = [
        str(binary_path),
        f"--arch.sic_num={sic_num}",
        f"--arch.xpu_num_per_sic={arch_config.inst_num.NUM_OF_CORE_PER_CLUSTER}",
        f"--arch.l2_bytes_per_sic={arch_config.memory.l2.SIZE_PER_SIC}",
        f"--arch.l1_bytes_per_xpu={arch_config.memory.l1.SIZE_PER_CORE}",
        f"--arch.l3_mc_bandwidth={l3_mc_bw}",
        f"--arch.l3_mc_cnt=1",
    ]

    for dtype, v in arch_config.compute.thread_2d_throughput.items():
        dtype: DType
        # TODO: need review as now using libra default value when v is not set
        if not v or dtype == DType.FP4:
            continue
        args.append(f"--arch.xpu_2d_macs.{dtype.name.lower()}={v}")

    for dtype, v in arch_config.compute.thread_1d_throughput.items():
        dtype: DType
        # TODO: need review as now using libra default value when v is not set
        if not v or dtype == DType.FP4:
            continue
        args.append(f"--arch.xpu_1d_throughput.{dtype.name.lower()}={v}")
    args.append(
        f"--arch.xpu_sfu_throughput={arch_config.compute.thread_sfu_throughput}")

    for k, v in dataflow_config.items():
        if isinstance(v, (list, tuple)):
            v = ",".join(str(i) for i in v)
        args.append(f"--{k}={v}")

    cmd = " ".join(args)
    return cmd


def check_cuda():
    try:
        import pycuda
        return True
    except Exception as e:
        return False


action_map = {
    "xpu": {
        "gemm": XPUGemmAction,
        "gemm.local": XPULocalGemmAction,
        "gemm.shared": XPUSharedGemmAction,
        "Nop": XPUNopAction,
        "sigmoid": XPUSigmoidAction,
        "gelu": XPUGeluAction,
        "relu": XPUReluAction,
        "silu": XPUSiluAction,
        "add": XPUAddAction,
        "mul": XPUMulAction,
        "gather": XPUGatherAction,
        "scatter": XPUScatterAction,
        "layernorm": XPULayernormAction,
        "softmax": XPUSoftmaxAction,
        "softmaxbackward": XPUSoftmaxBackwardAction,
        "sdpa": XPUSdpaAction,
        "allreduce": XPUAllReduceAction,
        "allgather": XPUAllGatherAction,
        "allgather_gemm": XPUAllGatherGemmAction,
    },
    "cdte": {
        "slice": CDTESliceAction,
        "deslice": CDTEDesliceAction,
        "slice-reshape": CDTEReshapeAction,
    },
}


def customize_loader(d, config, dataflow_config, topo, case_id):
    if "action_type" not in d:
        raise Exception("action_type is required")
    action_type = d["action_type"]
    d["config"] = config
    d["topo"] = topo
    d["dataflow_config"] = dataflow_config
    d["case_id"] = case_id
    if action_type == "xpu":
        code = d["code"]
        optype = code.rsplit("_", 1)[0]
        if optype in action_map["xpu"]:
            cls = action_map["xpu"][optype]
            return cls
        else:
            raise Exception(f"{action_type}:{optype} not supported")
    elif action_type == "cdte":
        op = d["op"]
        if op in action_map["cdte"]:
            cls = action_map["cdte"][op]
            return cls
        else:
            raise Exception(f"{action_type}:{op} not supported")
    else:
        raise Exception(f"action_type={d['action_type']} is not supported")


class BossaNovaExecutor():

    def __init__(
        self,
        config: BossaNovaConfig,
        outdir: str,
        tgen: TraceGenerator,
        op_shape,
        optype,
        dtype,
        input_addr,
        output_addr,
        dataflow_config,
        mu=1.0,
        device="auto",
        device_id=0,
        enable_cache=False,
        cache_svc=None,
        enable_power_svc=False,
        enable_dump_addr=False,
        initial_ref=0,
        esl_switch: BaseESLSwitch = None,
        case_idx=None,
    ):
        self.config = config
        self.cache_svc = None
        self.outdir = outdir
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.dataflow_config = dataflow_config
        self.esl_switch = esl_switch
        self.case_idx = case_idx
        if not enable_cache:
            logger.info("cache model disabled")

        elif cache_svc:
            self.cache_svc = cache_svc
        else:
            if device == 'cpu':
                self.cache_svc = CacheCostService(config)
                logger.info("device=cpu, default cache svc inited")
            elif device == 'gpu':
                device_count = get_gpu_count()
                if device_count:
                    from nova_platform.cost_service.cache.parallel_cache_cost_service import ParallelCacheCostService
                    self.cache_svc = ParallelCacheCostService(
                        config, device_id=device_id)
                    logger.info("device=auto, parallel cache svc inited")
                else:
                    msg = "device=gpu, BossaNova failed to init parallel cache svc"
                    logger.error(msg)
                    raise Exception(msg)
            elif device == 'auto':
                device_count = get_gpu_count()
                if device_count:
                    from nova_platform.cost_service.cache.parallel_cache_cost_service import ParallelCacheCostService
                    self.cache_svc = ParallelCacheCostService(
                        config, device_id=device_id)
                    logger.info("device=auto, parallel cache svc inited")
                else:
                    self.cache_svc = CacheCostService(config)
                    logger.info("device=auto, default cache svc inited")
            else:
                raise Exception("device=%s is not supported", device)
            logger.info("cache model enabled")

        # make an empty power_gen
        class EmptyPowerSvc:
            def process(self, action, context, ref):
                while True:
                    stat: DataflowActionMemoryStat = yield
                    if not stat:
                        break
                yield

            def get_edc_freq(self, ref, context):
                base_freq_cfg = context.bw_resource_context.config.freq
                return base_freq_cfg

            def init_context(self):
                pass

            def post_process(self, context: BossaNovaContext):
                pass

            def post_stat(self, context: BossaNovaContext, dataflow: Dataflow):
                return {}

        self.power_svc = PowerCostService(
            config) if enable_power_svc else EmptyPowerSvc()

        def get_dump_addr():
            self.f = open(f"{self.outdir}/addr_dump.txt", "w")
            self.f.write(
                "start,end,master,die_id,cluster_id,engine_id,address,size,rw\n")

            def dump_addr(action: DiagDataflowAction, stat, ref):
                master = action.get_action_type()
                die_id = action.get_die_id()
                cluster_id = action.get_cluster_id()
                engine_id = action.get_local_engine_id()
                for access in stat.memory_access_list:
                    if access.base_addr >= 2**40*5:
                        s = ref+stat.relative_ts
                        e = ref+stat.relative_ts+stat.latency
                        self.f.write(
                            f"{s:0.9f},{e:0.9f},{master}, {die_id}, {cluster_id}, {engine_id}, {hex(access.base_addr)}, {hex(access.size)}, {access.rw}\n")
            return dump_addr

        self.compute_svc = ComputeCostService(
            config,
            power_svc=self.power_svc,
            cache_svc=self.cache_svc if self.cache_svc else None,
            esl_switch=esl_switch,
            dump_addr=get_dump_addr() if enable_dump_addr else lambda action, stat, ref: None
        )
        # if not tgen:
        #     self.tgen = TraceGenerator(
        #         f"{outdir}/trace.perfetto-trace", self.config.gcu_id)
        # else:
        #     self.tgen = tgen

        self.tgen = tgen
        self.post_processor = PostProcessor(outdir=outdir, tgen=self.tgen)
        self.context = BossaNovaContext(
            initial_ref=initial_ref,
            bw_resource_context=BWResourceContext(
                config=self.config, esl_switch=self.esl_switch),
            power_context=self.power_svc.init_context(),
            tgen=self.tgen,
        )
        self.dataflow = self.generate_dataflow(op_shape, optype, dtype, mu)

    def get_service_list(self) -> Generator[BaseCostService, None, None]:
        if self.cache_svc:
            yield self.cache_svc
        yield self.compute_svc
        if self.power_svc:
            yield self.power_svc

    def get_ref(self, dataflow: Dataflow, action: DataflowAction, visited):
        '''
        caculate the start time at the last action in a dataflow
        '''
        dag = dataflow.dag
        if action.get_action_id() in visited:
            return visited[action.get_action_id()]
        ref = 0
        for parent_id in dag.predecessors(action.get_action_id()):
            parent = dataflow._action_map[parent_id]
            latency = self.get_ref(dataflow, parent, visited)
            latency += self.context.cost_dict[parent_id].latency
            if latency > ref:
                ref = latency
        visited[action.get_action_id()] = ref
        return ref

    def _process_action(self, action: DataflowAction, ref):
        cid = action.get_cluster_id()
        engine_id = action.get_local_engine_id()
        action_type = action.get_action_type()
        action.ref = ref
        track = self.context.get_cluster_tgen(action.get_die_id(), cid).create_track(
            f"{action_type}:{engine_id}", tid=engine_id)
        yield from self.compute_svc.process(action, self.context, ref)
        # yield from self.power_svc.process(action, self.context, ref)
        self.compute_svc.post_process(self.context)
        if self.power_svc:
            self.power_svc.post_process(self.context)

        # core_stat = action.get_core_stat()

        cost_book = self.context.get_cost_book(action)

        track.duration(ref, cost_book.latency, action_type, cost_book)
        return cost_book.latency

    def execute(self) -> Generator[float, None, None]:
        visited = {}
        action_gen = self.dataflow.execute_dataflow()
        action_id, action = next(action_gen)
        action_map: Dict[int, Generator] = {}

        initial_ref = self.context.initial_ref
        max_ref: float = 0
        action_len = len(self.dataflow._action_map)

        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

        position = int(worker_id.replace("gw", "")
                       ) if worker_id != "master" else 0  # 根据 worker ID 计算位置

        pbar = tqdm(total=action_len,
                    desc=f"Worker {worker_id}", position=position, leave=True, unit="act")
        next_ref = 0
        event_barrier_map: Dict[int, BARRIER | BossaNovaEvent] = {}

        def _check_barrier():
            for _action_id in list(event_barrier_map.keys()):
                _event_or_barrier = event_barrier_map[_action_id]
                if _event_or_barrier.is_done:
                    self.dataflow.queue.put(
                        (_event_or_barrier.max_t, _action_id))
                    event_barrier_map.pop(_action_id)

        while True:
            try:
                if action:
                    local_latency = self.get_ref(
                        self.dataflow, action, visited)
                    ref = local_latency+initial_ref
                    is_done = False
                    stat = None
                    if id(action) in action_map:
                        action_process = action_map[id(action)]
                        try:
                            next_ref, stat = next(action_process)
                            if isinstance(stat, BARRIER) or isinstance(stat, BossaNovaEvent):
                                event_barrier_map[action_id] = stat
                        except StopIteration as e:
                            next_ref = ref + e.value
                            action_map.pop(id(action))
                            pbar.update(1)
                            is_done = True
                        finally:
                            yield next_ref, 'running'
                    else:
                        action_process = self._process_action(action, ref)
                        assert isinstance(
                            action_process, GeneratorType), "process must be an generator"
                        action_map[id(action)] = action_process
                        next_ref = ref
                        # continue

                    action_latency = self.context.cost_dict[action.get_action_id(
                    )].latency
                    max_ref = max(max_ref, ref + action_latency)

                    _check_barrier()
                    # send back current action status and get the next
                    action_id, action = action_gen.send(
                        (next_ref, is_done, stat))
                    pass
                elif len(event_barrier_map) > 0:
                    # print("!!!!!event_barrier_map")
                    next_ref = MAX_TIME
                    for b in event_barrier_map.values():
                        next_ref = min(next_ref, b.max_t)
                    # wait for barrier
                    yield next_ref, 'wait'
                    _check_barrier()
                    action_id, action = action_gen.send(
                        (next_ref, False, None))

                else:
                    action_id, action = action_gen.send(
                        (None, None, None))
                    # 当action和barrier_map 均无待处理工作时表示完成
                    break

            except StopIteration:
                break
        self.end_ref = max_ref
        pbar.close()

    def generate_dataflow(self, op_shape, optype, dtype, mu) -> Dataflow:
        dataflow_config = {
            "bench_out_dir": self.outdir,
            "bench_op_type": optype,
            "bench_basic_data_type": dtype.name.lower(),
        }

        def _resolve_gemm_op_type(version: int) -> str:
            if version in (4, 6):
                return "gemm.local"
            if version == 5:
                return "gemm.shared"
            return "gemm"

        if optype == 'linear':
            op_shape = [1, op_shape[0] * op_shape[1],
                        op_shape[2], op_shape[3]]
            op_version = self.dataflow_config.get("bench_gemm_op_version", 2)
            dataflow_config["bench_op_type"] = _resolve_gemm_op_type(op_version)
            dataflow_config["bench_gemm_shape_b_m_k_n"] = op_shape
        elif optype == 'matmul':
            op_version = self.dataflow_config.get("bench_gemm_op_version", 2)
            dataflow_config["bench_op_type"] = _resolve_gemm_op_type(op_version)
            dataflow_config["bench_gemm_shape_b_m_k_n"] = op_shape
        elif optype == "gemm":
            op_version = self.dataflow_config.get("bench_gemm_op_version", 2)
            dataflow_config["bench_op_type"] = _resolve_gemm_op_type(op_version)
            dataflow_config["bench_gemm_shape_b_m_k_n"] = op_shape
        elif optype in ["add", "mul", "gelu", "relu", "silu", "sigmoid"]:
            dataflow_config["bench_elementwise_shape"] = op_shape
        elif optype == "layernorm":
            dataflow_config["bench_layernorm_shape"] = op_shape
        elif optype in ["softmax", "softmaxbackward"]:
            dataflow_config["bench_softmax_shape_b_c"] = op_shape
        elif optype == "gather":
            dataflow_config["bench_gather_shape"] = op_shape
            self.config.gather_mu = mu
        elif optype == "scatter":
            dataflow_config["bench_scatter_shape"] = op_shape
        elif optype == "transpose":
            dataflow_config["bench_transpose_tensor_b_h_w_c"] = op_shape[0]
            dataflow_config["bench_transpose_dim_map"] = op_shape[1]
        elif optype == "sdpa":
            dataflow_config["bench_sdpa_q_shape"] = op_shape[0]
            dataflow_config["bench_sdpa_kv_shape"] = op_shape[1]
            if len(op_shape) > 2:
                assert isinstance(op_shape[2], (bool, int))
                dataflow_config["bench_sdpa_with_dropout"] = 1 if op_shape[2] else 0
        elif optype == "allreduce":
            dataflow_config["bench_all_reduce_shape"] = op_shape
        elif optype == "allgather":
            dataflow_config["bench_allgather_in_shape"] = op_shape[1]
            dataflow_config["bench_allgather_out_shape"] = op_shape[0]
        elif optype == "allgather_gemm":
            dataflow_config["bench_allgather_gemm_shape_b_m_k_n"] = op_shape
        elif optype == "gemm.shared":
            dataflow_config["bench_gemm_shape_b_m_k_n"] = op_shape
        elif optype == "gemm.local":
            dataflow_config["bench_gemm_shape_b_m_k_n"] = op_shape
        else:
            raise Exception("check if introducing new optype, %s", optype)

        addr_input_list, addr_output_list = self.input_addr, self.output_addr
        if addr_input_list or addr_output_list:
            dataflow_config["bench_input_tensor_addr"] = addr_input_list
            dataflow_config["bench_output_tensor_addr"] = addr_output_list
        else:
            print("-------------------------------------------")
            logger.info("No input/output addresses provided for this case")

        dataflow_config = {**dataflow_config, **self.dataflow_config}
        bench_op_type = dataflow_config["bench_op_type"]
        python_supported = bench_op_type in DATAFLOW_GENERATOR_MAPPING

        if python_supported:
            return generate_dataflow(self.config, dataflow_config, self.esl_switch.topo, self.case_idx)

        repo_root = Path(__file__).resolve().parents[2]
        dataflow_binary = repo_root / "build" / "dataflow_gen"

        if not dataflow_binary.exists():
            if python_supported:
                logger.warning(
                    "dataflow_gen binary not found at %s; falling back to in-process generator for bench_op_type=%s",
                    dataflow_binary, bench_op_type
                )
                return generate_dataflow(self.config, dataflow_config, self.esl_switch.topo, self.case_idx)
            raise FileNotFoundError(
                f"Required dataflow generator binary not found at {dataflow_binary}. "
                f"Bench op type '{bench_op_type}' has no built-in fallback."
            )
        dataflow_path = dataflow_config["bench_out_dir"]
        cmd = get_data_flow_gen_cmd(dataflow_config, self.config)
        process = subprocess.run(cmd.encode(
            "utf-8"), shell=True, encoding="utf-8", capture_output=True)
        ret_code = process.returncode
        logger.info("\n" + process.stdout)
        if ret_code != 0:
            logger.error(process.stderr)
            if python_supported:
                logger.warning(
                    "External dataflow generation failed; retrying with in-process generator for bench_op_type=%s",
                    bench_op_type
                )
                return generate_dataflow(self.config, dataflow_config, self.esl_switch.topo, self.case_idx)
            raise Exception(
                f"Failed to generate dataflow: {process.stderr}")

        df_files = glob(f"{dataflow_path}/*dataflow.yaml")
        assert len(df_files) > 0, "Can't find generated dataflow yaml file!"
        logger.info(f"Found dataflow files: {df_files}")  # print the files
        assert len(
            df_files) == 1, "Found more than one dataflow yaml which is not supported now!"

        p = Path(df_files[0])
        if not p.exists():
            raise Exception(f"{df_files[0]} not exists!")
        with open(p) as f:
            data_dict = yaml.safe_load(f)

        dataflow = dict_to_dataclass(
            data_dict,
            DiagDataflow,
            {
                DiagDataflowAction: functools.partial(
                    customize_loader,
                    config=self.config,
                    topo=self.esl_switch.topo,
                    dataflow_config=dataflow_config,
                    case_id=self.case_idx
                )
            },
            restrict_mode=False,
        )
        if dataflow_config["bench_op_type"] in [
            "transpose",
            "sdpa",
            "allreduce",
            "allgather",
            "allgather_gemm",
        ] or dataflow_config.get("bench_gemm_op_version", 1) in [4, 5]:
            return dataflow
        shape_files = glob(f"{dataflow_path}/*shape_list.txt")
        assert len(
            shape_files) > 0, "Can't find generated grid_shape_list file!"
        assert len(
            shape_files) == 1, "Found more than one grid_shape_list file which is not supported now!"
        tile_info = parse_shape_tile_info(shape_files[0])
        for act in dataflow.action_list:
            act.tile_info = tile_info
        return dataflow

    def generate_report(self):
        report = self.post_processor.generate_report(
            self.context, self.dataflow, self.config, self.get_service_list())
        # self.post_processor.tgen.flush()
        # self.tgen.block_until_all_tasks_done()
        return report
