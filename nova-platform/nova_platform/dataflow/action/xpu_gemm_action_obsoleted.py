import functools
import math
from typing import Generator, Tuple
from dataclasses import asdict, dataclass, field

from nova_platform.base_model import DType, DataflowActionMemoryAccess
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, BossaNovaContext
from nova_platform.dataflow.action.xpu_action import XPUAction
from nova_platform.utils.base_utils import hash_list


@dataclass
class InstructionInfo:
    dtype: DType
    mid_level_ins_num: int
    mid_level_ins_shape: list
    scalar_ins_num: int
    st_ins_num: int


@dataclass
class CoreCost(BaseCoreStat):
    dtype: DType = DType.FP16
    ld_smr_occupation: int = 0  # Bytes
    ld_iv_occupation: int = 0  # Bytes
    va_occupation: int = 0  # Bytes

    ld_smr_basic_cost: int = 0
    ld_iv_basic_cost: int = 0
    scalar_basic_cost: int = 0
    vmm_basic_cost: int = 0
    vst_basic_cost: int = 0
    ld_smr_latency: int = 0
    ld_iv_latency: int = 0
    main_body_length: int = 0
    data_preparation_cost: int = 0
    compute_cost: int = 0
    compute_vs_store_ratio: float = 0
    vst_cost: int = 0
    leading_latency_without_reuse: int = 0
    data_reuse_times: int = 0
    single_thread_main_body_cost_without_reuse: int = 0
    single_thread_compute_ratio_without_reuse: int = 0
    single_thread_main_body_cost_with_reuse: int = 0
    single_thread_compute_ratio_with_reuse: int = 0
    multi_thread_main_body_cost_without_reuse: int = 0
    multi_thread_compute_ratio_without_reuse: int = 0
    multi_thread_main_body_cost_with_reuse: int = 0
    multi_thread_compute_ratio_with_reuse: int = 0
    total_cost: float = 0
    instruction_info: InstructionInfo = 0


def gemm_kernel(m: int, n: int, k: int, dtype: DType = DType.FP16):
    computer_count = 0
    st_count = 0
    scalar_count = 0

    if dtype == DType.FP16:
        slice_m = 64
        slice_n = 64
        slice_k = 32
        eleByte = 2
    elif dtype == DType.INT8:
        slice_m = 64
        slice_n = 64
        slice_k = 64
        eleByte = 1
    else:
        raise RuntimeError(f"unsupport gemm_kernel dtype {dtype}")

    scalar_count += 23
    scalar_count += 2
    for loop_m in range(0, m, slice_m):
        scalar_count += 3
        scalar_count += 2
        scalar_count += 2
        for loop_n in range(0, n, slice_n):
            scalar_count += 3
            scalar_count += 8
            computer_count += 1
            scalar_count += 2
            for loop_k in range(0, k, slice_k):
                scalar_count += 3
                scalar_count += 4
                computer_count += 1
            scalar_count += 2
            st_count += 1
            scalar_count += 1
    ins_shape_m = min(m, slice_m)
    ins_shape_n = min(n, slice_n)
    ins_shape_k = min(k, slice_k)
    return InstructionInfo(dtype, computer_count, [ins_shape_m, ins_shape_n, ins_shape_k], scalar_count, st_count)


@dataclass
class XPUGemmAction(XPUAction):
    core_cost: CoreCost = field(default_factory=CoreCost)

    def get_valid_shape(self):
        valid_m = self.data[4] & (0xFFFF)
        valid_n = self.data[4] >> 16
        valid_k = self.data[5]
        return [valid_m, valid_n, valid_k]

    def get_memory_access(self) -> Generator[DataflowActionMemoryAccess, None, None]:
        for input in self.inputs:
            for tensor in input.tensor:
                yield from self._iter_tensor_addr(tensor.addr, tensor, 'r')
        for output in self.outputs:
            for tensor in output.tensor:
                yield from self._iter_tensor_addr(tensor.addr, tensor, 'w')

    def compute(self, context: BossaNovaContext) -> Generator[None, None, BaseCoreStat]:
        valid_shape = self.get_valid_shape()
        dtype = self.get_dtype()
        # dtype = DType.FP16
        gemm_shape = hash_list(valid_shape)
        self._compute(gemm_shape, dtype)
        self.core_cost.latency = self.core_cost.total_cost/1e9
        yield from ()
        return self.core_cost

    def _compute(self, gemm_shape, dtype) -> CoreCost:
        self.core_cost.dtype = dtype
        self.core_cost.instruction_info = gemm_kernel(
            gemm_shape[0], gemm_shape[1], gemm_shape[2], dtype=dtype)

        self._get_l0_occupation()
        self._get_basic_cost()
        self._get_input_latency()
        self._get_main_body_length()
        self._get_main_body_cost()
        self._get_leading_latency()
        self._get_single_thread_cost()
        self._get_multi_thread_cost()
        self._get_total_cost()
        self._basic_stat_info()

    def _get_l0_occupation(self):  # step 3
        instruction_info = self.core_cost.instruction_info
        mid_level_ins_shape = instruction_info.mid_level_ins_shape
        bpe = self.core_cost.dtype.get_bpe()
        m = 0
        n = 1
        k = 2
        self.core_cost.ld_smr_occupation = mid_level_ins_shape[m] * \
            mid_level_ins_shape[k] * bpe
        self.core_cost.ld_iv_occupation = mid_level_ins_shape[n] * \
            mid_level_ins_shape[k] * bpe
        self.core_cost.va_occupation = mid_level_ins_shape[m] * \
            mid_level_ins_shape[n] * bpe

    def _get_basic_cost(self):  # step 4
        m = 0
        n = 1
        k = 2
        instruction_info: InstructionInfo = self.core_cost.instruction_info
        mid_level_ins_shape = instruction_info.mid_level_ins_shape
        bpe = instruction_info.dtype.get_bpe()
        # throughput = asdict(self.config.compute.thread_2d_throughput)
        # mac_array = throughput[instruction_info.dtype.name.upper()]
        mac_array = self.config.compute.thread_2d_throughput[instruction_info.dtype]
        LOAD_SMR = self.config.bw.xpu.l0.SHARED
        self.core_cost.ld_smr_basic_cost = (
            mid_level_ins_shape[m]
            * mid_level_ins_shape[k]
            * bpe
            / (LOAD_SMR * self.config.freq.CORE)
        )  # 假设smr读取的是m,k
        LOAD_IV = self.config.bw.xpu.l0.SHARED
        self.core_cost.ld_iv_basic_cost = (
            mid_level_ins_shape[n]
            * mid_level_ins_shape[k]
            * bpe
            / (LOAD_IV * self.config.freq.CORE)
        )  # 假设iv读取的是n,k
        self.core_cost.scalar_basic_cost = (
            instruction_info.scalar_ins_num / instruction_info.mid_level_ins_num /
            self.config.freq.CORE
        )
        self.core_cost.vmm_basic_cost = (
            mid_level_ins_shape[m]
            * mid_level_ins_shape[n]
            * mid_level_ins_shape[k]
            / (mac_array * self.config.freq.CORE)
        )
        VST_BW = self.config.bw.xpu.l0.SHARED/self.config.bw.xpu.l0.SHARED_RW_RATIO
        self.core_cost.vst_basic_cost = (
            mid_level_ins_shape[m]
            * mid_level_ins_shape[n]
            * bpe
            / (VST_BW * self.config.freq.CORE)
        )

    def _get_input_latency(self):  # step 5
        # self.core_cost.ld_smr_latency = self.dsm_cost.dsm_local_latency
        # self.core_cost.ld_iv_latency = self.dsm_cost.dsm_local_latency
        self.core_cost.ld_smr_latency = self.config.latency.l0.SHARED[0] / \
            self.config.freq.CORE
        self.core_cost.ld_iv_latency = self.config.latency.l0.SHARED[0] / \
            self.config.freq.CORE

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = min(
            (self.config.memory.l0.IV_SIZE / self.core_cost.ld_iv_occupation),
            (self.config.memory.l0.SMR_SIZE / self.core_cost.ld_smr_occupation),
        )

    def _get_main_body_cost(self):  # step 7
        instruction_info = self.core_cost.instruction_info
        ld_smr_cost = self.core_cost.ld_smr_basic_cost * self.core_cost.main_body_length
        ld_iv_cost = self.core_cost.ld_iv_basic_cost * self.core_cost.main_body_length
        scalar_cost = self.core_cost.scalar_basic_cost * self.core_cost.main_body_length
        self.core_cost.data_preparation_cost = scalar_cost + \
            max(ld_smr_cost, ld_iv_cost)
        VMM = 10  # KG->L0 latency
        # VMM = self.config.latency.l0.SHARED[0]  # only core clock domain latency
        self.core_cost.compute_cost = self.core_cost.vmm_basic_cost * self.core_cost.main_body_length + (
            VMM / self.config.freq.CORE
        )
        self.core_cost.compute_vs_store_ratio = (
            instruction_info.mid_level_ins_num / instruction_info.st_ins_num
        )
        # only core clock domain latency
        VST = self.config.latency.l0.SHARED[0]
        self.core_cost.vst_cost = (
            self.core_cost.vst_basic_cost * self.core_cost.main_body_length /
            self.core_cost.compute_vs_store_ratio
            + (VST /
               self.config.freq.CORE)
        )

    def _get_leading_latency(self):  # step 8
        self.core_cost.leading_latency_without_reuse = max(
            0,
            (
                max(self.core_cost.ld_smr_latency, self.core_cost.ld_iv_latency)
                - (self.core_cost.compute_cost + self.core_cost.vst_cost)
                / (self.core_cost.main_body_length / self.core_cost.compute_vs_store_ratio)
            ),
        )
        self.core_cost.data_reuse_times = math.ceil(
            max(self.core_cost.ld_smr_latency, self.core_cost.ld_iv_latency)
            / (
                (self.core_cost.compute_cost + self.core_cost.vst_cost)
                / (self.core_cost.main_body_length / self.core_cost.compute_vs_store_ratio)
            )
        )

    def _get_single_thread_cost(self):  # step 9
        self.core_cost.single_thread_main_body_cost_without_reuse = (
            max(self.core_cost.data_preparation_cost, self.core_cost.compute_cost)
            + self.core_cost.vst_cost
            + self.core_cost.leading_latency_without_reuse
        )
        self.core_cost.single_thread_compute_ratio_without_reuse = (
            self.core_cost.compute_cost /
            self.core_cost.single_thread_main_body_cost_without_reuse
        )
        self.core_cost.single_thread_main_body_cost_with_reuse = (
            max(self.core_cost.data_preparation_cost,
                self.core_cost.compute_cost) + self.core_cost.vst_cost
        )
        self.core_cost.single_thread_compute_ratio_with_reuse = (
            self.core_cost.compute_cost / self.core_cost.single_thread_main_body_cost_with_reuse
        )

    def _get_multi_thread_cost(self):  # step 10
        self.core_cost.multi_thread_main_body_cost_without_reuse = (
            max(self.core_cost.data_preparation_cost, self.core_cost.compute_cost)
            if self.core_cost.leading_latency_without_reuse == 0
            else self.core_cost.single_thread_main_body_cost_without_reuse
        )
        self.core_cost.multi_thread_compute_ratio_without_reuse = (
            self.core_cost.compute_cost / self.core_cost.multi_thread_main_body_cost_without_reuse
        )
        self.core_cost.multi_thread_main_body_cost_with_reuse = max(
            self.core_cost.data_preparation_cost, self.core_cost.compute_cost
        )
        self.core_cost.multi_thread_compute_ratio_with_reuse = (
            self.core_cost.compute_cost / self.core_cost.multi_thread_main_body_cost_with_reuse
        )

    def _basic_stat_info(self):
        # TODO: need review
        info = self.core_cost.instruction_info
        m, n, k = info.mid_level_ins_shape  # m,n,k
        dtype = self.core_cost.dtype
        bpe = dtype.get_bpe()
        self.core_cost.tensor_macs[dtype] = m*n*k*info.mid_level_ins_num  # mac
        self.core_cost.vector_ops[dtype] = info.scalar_ins_num
        self.core_cost.ld_l0_shared = (
            m*k*bpe+n*k*bpe)*info.mid_level_ins_num
        self.core_cost.ld_l0_shared = m*n*bpe*info.st_ins_num

    def _get_total_cost(self):
        instruction_info = self.core_cost.instruction_info
        main_body_cost = self.core_cost.multi_thread_main_body_cost_with_reuse
        main_body_num = float(
            instruction_info.mid_level_ins_num) / self.core_cost.main_body_length
        self.core_cost.total_cost = main_body_cost * main_body_num
