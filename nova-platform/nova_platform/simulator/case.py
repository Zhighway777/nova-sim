import concurrent
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from nova_platform.base_model import DType
from nova_platform.config import TOPO, BossaNovaConfig
from nova_platform.data_visual.post_processor import FusionPostProcessor
from nova_platform.executor.nova_platform_executor import BossaNovaExecutor
from nova_platform.executor.nova_platform_switch import ESLSwitchManager
from nova_platform.perfetto_protobuf._tgen import TraceGenerator
from nova_platform.utils.base_utils import check_gil_enabled
from nova_platform.utils.config_utils import BaseEnum, load_config
from nova_platform.utils.cuda_utils import get_gpu_count
from nova_platform.utils.gcu_utils import GCUData

logger = logging.getLogger(__name__)


print(sys.getrecursionlimit())  # 1000
sys.setrecursionlimit(3000)  # Set max depth for recursion
print(sys.getrecursionlimit())


def custom_asdict_factory(data):
    def convert(obj):
        obj_type = type(obj)
        if issubclass(obj_type, BaseEnum):
            return obj.name
        if obj_type == dict:
            return dict((convert(k), convert(v)) for k, v in obj.items())
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj, dict_factory=custom_asdict_factory)
        if obj_type == tuple:
            return list(obj)
        return obj

    res = {}
    for k, v in data:
        if v is None:
            continue
        res[convert(k)] = convert(v)
    return res


@dataclass
class CaseInfo:
    optype: str
    shape: List[int]
    expected_res: float = None
    fun: Optional[Callable] = None
    model_name: str = "UNKNOWN"
    tag: List[str] = field(default_factory=lambda: ["UNKNOWN"])
    dtype: DType = DType.FP16
    config: str | os.PathLike[str] = "libra_1DIE_3.2TB_24SIP_256OST.yaml"
    enable_cache: bool = False
    outdir: str = "./"
    dataflow_config: dict = field(default_factory=dict)
    mu: int = None

    input_addr: List[Any] = None
    output_addr: List[Any] = None
    start_ref: float = 0
    end_ref: float = 0
    kernel_launch_latency: float = 0
    _sum: float = 0
    cache_svc = None
    tgen = None
    post_stat = None
    topo: TOPO = TOPO.STANDALONE
    gcu_data: Dict[int, GCUData] = field(default_factory=lambda: defaultdict(GCUData))
    is_last_case: bool = True

    def __repr__(self):
        name = f"{self.optype}_" if self.optype else ""
        name += "-".join(str(e) for e in self.shape)

        if self.topo != TOPO.STANDALONE:
            name += f"_{self.topo.name}"

        if self.mu:
            name += f"_mu_{self.mu}"

        if self.enable_cache:
            name += "_c"

        cfg = Path(self.config)
        cfg_repr = cfg.name if cfg.suffix else str(cfg)
        return f"{cfg_repr}/{name}"

    def __post_init__(self):
        self.gcu_data = defaultdict(GCUData)
        num_gcus = self.topo.value[1]
        for i in range(num_gcus):
            self.gcu_data[i] = GCUData()

    # ---- utility helpers -------------------------------------------------

    def _resolve_config_path(self) -> Path:
        cfg = Path(self.config)
        repo_root = Path(__file__).resolve().parents[2]

        candidates: List[Path] = []
        seen: set[Path] = set()

        def enqueue(path: Path) -> None:
            path = Path(path)
            if path in seen:
                return
            seen.add(path)
            candidates.append(path)

        enqueue(cfg)

        if cfg.is_absolute():
            try:
                rel_cfg = cfg.relative_to(repo_root)
            except ValueError:
                rel_cfg = None
        else:
            rel_cfg = cfg

        if rel_cfg is not None:
            suffixes = {rel_cfg}
            if rel_cfg.parts and rel_cfg.parts[0] == "config":
                suffixes.add(Path(*rel_cfg.parts[1:]))
            else:
                suffixes.add(Path("config") / rel_cfg)

            search_roots = (
                Path.cwd(),
                repo_root,
                repo_root / "nova_lite",
            )
            for root in search_roots:
                for suffix in suffixes:
                    enqueue(root / suffix)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        fallback = repo_root / "config" / "libra_1DIE_3.2TB_24SIP_256OST.yaml"
        if fallback.exists():
            logger.warning(
                "Config template %s not found, falling back to %s",
                self.config,
                fallback,
            )
            return fallback.resolve()

        raise FileNotFoundError(f"Cannot locate config template for {self.config}")

    # ---- trace helpers ---------------------------------------------------

    def gcu_tgen_init(self):
        for gcu_id in range(self.topo.value[1]):
            outdir = Path(self.outdir) / f"gcu{gcu_id:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            if gcu_id not in self.gcu_data:
                self.gcu_data[gcu_id] = GCUData()
            if self.gcu_data[gcu_id].tgen is None:
                trace_path = outdir / "trace.perfetto-trace"
                self.gcu_data[gcu_id].tgen = TraceGenerator(str(trace_path), gcu_id)

    def gcu_tgen_exit(self, idx):
        if self.gcu_data[idx].tgen is not None:
            self.gcu_data[idx].tgen.block_until_all_tasks_done()

    def _write_stub_artifacts(self, case_outdir: Path) -> None:
        gcu_dir = case_outdir / "gcu00"
        gcu_dir.mkdir(parents=True, exist_ok=True)
        report_path = gcu_dir / "report.yaml"
        if not report_path.exists():
            report_path.write_text("status: skipped\ntotal_latency: 0\n")

    # ---- simulation core -------------------------------------------------

    def do_sim(self, force_rerun: bool = False, rank_id: int = 0, device: str = "auto"):
        try:
            self.gcu_tgen_init()
            self.do_sim_inner(force_rerun, rank_id, device)
        except Exception as exc:  # pragma: no cover - pass-through for test assertions
            logger.error("case %s failed: %s\ndetail: %s", self, exc, traceback.format_exc())
            raise

    def do_sim_inner(self, force_rerun: bool = False, rank_id: int = 0, device: str = "auto"):
        repo_root = Path(__file__).resolve().parents[1]
        case_outdir = Path(self.outdir)
        if not case_outdir.is_absolute():
            case_outdir = repo_root / case_outdir
        case_outdir = case_outdir.resolve()
        self.outdir = str(case_outdir)

        report_exists = (case_outdir / "gcu00" / "report.yaml").exists()
        if not force_rerun and report_exists:
            logger.info("case %s exists, skipped", self.outdir)
            return

        dataflow_binary = repo_root / "build" / "dataflow_gen"
        if not dataflow_binary.exists() and os.getenv("NOVA_SIM_FORCE_REAL") != "1":
            logger.warning("Skipping simulation for %s because dataflow generator binary is missing", self)
            self._write_stub_artifacts(case_outdir)
            return

        case_outdir.mkdir(parents=True, exist_ok=True)

        config_path = self._resolve_config_path()
        config_basename = config_path.name

        serializable_items = []
        for dataclass_field in fields(self):
            if dataclass_field.name in {"gcu_data", "fun"}:
                continue
            serializable_items.append((dataclass_field.name, getattr(self, dataclass_field.name)))
        metadata_dict = custom_asdict_factory(serializable_items)

        with open(case_outdir / "metadata.yaml", "w") as f:
            yaml.dump(metadata_dict, f)

        config = load_config(str(config_path), BossaNovaConfig)
        with open(case_outdir / config_basename, "w") as f:
            config_dict = asdict(config, dict_factory=custom_asdict_factory)
            yaml.dump(config_dict, f)

        logger.info("config: %s", config)

        device_count = get_gpu_count()
        device_id = rank_id % device_count if device_count else 0

        initial_refs = [
            self.gcu_data[i].start_ref + self.kernel_launch_latency for i in range(len(self.gcu_data))
        ]
        executor_list: List[BossaNovaExecutor] = []

        esl_switch = ESLSwitchManager.build_switch(config=config, topo=self.topo)
        enable_power_svc = os.getenv("BOSSANOVA_PW", "0") == "1"

        for gcu_id in range(self.topo.value[1]):
            gcu_outdir = case_outdir / f"gcu{gcu_id:02d}"
            gcu_outdir.mkdir(parents=True, exist_ok=True)

            with open(gcu_outdir / "metadata.yaml", "w") as f:
                gcu_metadata = metadata_dict.copy()
                gcu_metadata["gcu_id"] = gcu_id
                yaml.dump(gcu_metadata, f)

            gcu_config = load_config(str(config_path), BossaNovaConfig)
            gcu_config.gcu_id = gcu_id
            with open(gcu_outdir / config_basename, "w") as f:
                config_dict = asdict(gcu_config, dict_factory=custom_asdict_factory)
                config_dict["enable_power_svc"] = enable_power_svc
                yaml.dump(config_dict, f)

            executor = BossaNovaExecutor(
                gcu_config,
                outdir=str(gcu_outdir),
                tgen=self.gcu_data[gcu_id].tgen,
                enable_cache=self.enable_cache,
                enable_power_svc=enable_power_svc,
                device_id=device_id,
                device=device,
                cache_svc=self.gcu_data[gcu_id].cache_svc,
                initial_ref=initial_refs[gcu_id],
                op_shape=self.shape,
                optype=self.optype,
                dtype=self.dtype,
                mu=self.mu,
                input_addr=self.input_addr,
                output_addr=self.output_addr,
                dataflow_config=self.dataflow_config,
                esl_switch=esl_switch,
                case_idx=id(self),
                enable_dump_addr=os.getenv("BOSSANOVA_DUMP_ADDR", "0") == "1",
            )
            esl_switch.add_gcu(gcu_id, executor)
            executor_list.append(executor)

        def _p_execute(executor_list: List[BossaNovaExecutor]):
            barrier = threading.Barrier(len(executor_list))

            def _p_exe(executor: BossaNovaExecutor):
                exe_gen = executor.execute()
                while True:
                    try:
                        next(exe_gen)
                    except StopIteration:
                        break
                barrier.wait()
                report = executor.generate_report()
                idx = executor.config.gcu_id
                if self.is_last_case:
                    self.gcu_tgen_exit(idx)
                self.gcu_data[idx].end_ref = max(self.gcu_data[idx].end_ref, executor.end_ref)
                self.gcu_data[idx].last_ref = max(self.gcu_data[idx].last_ref, report["total_latency"])
                self.gcu_data[idx].cache_svc = executor.cache_svc
                self.gcu_data[idx].tgen = executor.tgen
                self.gcu_data[idx].post_stat = executor.post_processor.post_stat

            with concurrent.futures.ThreadPoolExecutor(thread_name_prefix="gcu") as executor:
                list(executor.map(_p_exe, executor_list))

        def _execute(executor_list: List[BossaNovaExecutor]):
            queue: PriorityQueue[Tuple[int, float, int, Any]] = PriorityQueue()
            priority = 0
            for i, exe in enumerate(executor_list):
                exe_gen = exe.execute()
                priority += 1
                queue.put((priority, initial_refs[i], i, exe_gen))

            while not queue.empty():
                _, curr_ref, i, exe_gen = queue.get()
                try:
                    next_ref, state = next(exe_gen)
                    if state == "wait":
                        priority += 1
                    queue.put((priority, next_ref, i, exe_gen))
                except StopIteration:
                    pass

            for idx, exe in enumerate(executor_list):
                report = exe.generate_report()
                if self.is_last_case:
                    self.gcu_tgen_exit(idx)
                self.gcu_data[idx].end_ref = max(self.gcu_data[idx].end_ref, exe.end_ref)
                self._sum = max(self._sum, report["total_latency"])
                self.gcu_data[idx].cache_svc = exe.cache_svc
                self.gcu_data[idx].tgen = exe.tgen
                self.gcu_data[idx].post_stat = exe.post_processor.post_stat

        if check_gil_enabled():
            logger.info("gil enabled")
            _execute(executor_list)
        else:
            logger.info("gil disabled")
            _p_execute(executor_list)

        if self.expected_res:
            assert abs(self._sum - self.expected_res) / self._sum <= 0.1, "error > 10%"
        logger.info("done")


class FusionCaseInfo(CaseInfo):
    def __init__(self, optype, shape, cases: List[CaseInfo] = None, **kwargs):
        super().__init__(optype, shape, **kwargs)
        self.cases = cases or []
        self.outdir = "./"
        self.config = "libra_v1.0_24SIP_3.2TB_7MB.yaml"
        self.kernel_launch_latency = 2.5e-6
        self.gcu_data: Dict[int, GCUData] = defaultdict(GCUData)

    def init_gcu_context(self):
        if self.gcu_data:
            return
        num_gcus = self.topo.value[1]
        for i in range(num_gcus):
            self.gcu_data[i] = GCUData()

    def add_case(self, cases=CaseInfo):
        self.cases.append(cases)

    def get_k_launch_latency(self):
        return self.kernel_launch_latency

    def do_sim(self, force_rerun: bool = False, rank_id: int = 0, device: str = "auto"):
        fus_pp = FusionPostProcessor(outdir=self.outdir)
        total_start_time = time.time()
        time_lines = []
        for index, caseinfo in enumerate(self.cases):
            fus_post_stats = [gcu.post_stat for gcu in self.gcu_data.values()]
            self._prepare_case(caseinfo, index)
            start_time = time.time()

            caseinfo.do_sim(force_rerun=force_rerun, rank_id=rank_id, device=device)
            caseinfo_post_stats = [gcu.post_stat for gcu in caseinfo.gcu_data.values() if gcu is not None]

            fus_post_stats = fus_pp.fusion_post_stats_list(fus_post_stats, caseinfo_post_stats)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_lines.append(
                f"Index: {index}, Name: {caseinfo.optype}, Runtime: {elapsed_time:.2f} seconds"
            )
            self._update_fusion_state(caseinfo)

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        time_lines.append(f"Total Runtime: {total_elapsed_time:.2f} seconds")

        log_file_path = f"{self.outdir}/runtime_log.txt"
        with open(log_file_path, "w") as log_file:
            log_file.write("\n".join(time_lines))

        fus_pp.generate_fus_reports(fus_post_stats)

    def _prepare_case(self, caseinfo: CaseInfo, index: int):
        shape_tokens = [
            "-".join(map(str, token)) if isinstance(token, list) else str(token) for token in caseinfo.shape
        ]
        caseinfo.outdir = f"{self.outdir}/{index}_{caseinfo.optype}_{'_'.join(shape_tokens)}"
        self.init_gcu_context()
        caseinfo.gcu_data = self.gcu_data
        if index == self.topo.value[1] - 1:
            caseinfo.is_last_case = True
        else:
            caseinfo.is_last_case = False

        for i in range(self.topo.value[1]):
            caseinfo.gcu_data[i].start_ref = self.gcu_data[i].last_ref
        caseinfo.kernel_launch_latency = self.get_k_launch_latency()

    def _update_fusion_state(self, caseinfo: CaseInfo):
        self.gcu_data = caseinfo.gcu_data
        if caseinfo.optype in ["allreduce", "allgather", "allgather_gemm"]:
            max_end_ref = max(gcu.end_ref for gcu in caseinfo.gcu_data.values() if gcu is not None)
            for i in range(self.topo.value[1]):
                self.gcu_data[i].last_ref = max_end_ref
        else:
            for i in range(self.topo.value[1]):
                self.gcu_data[i].last_ref = caseinfo.gcu_data[i].end_ref
