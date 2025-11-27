from typing import Generator
from dataclasses import dataclass, field
from typing import Generator, List, Tuple

from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
    DataflowActionComputeStat,
)
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat
from nova_platform.dataflow.action.xpu_action import XPUAction


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
    scalar_basic_cost_per_cycle: int = 0
    vmm_basic_cost_per_cycle: int = 0
    vst_basic_cost: int = 0

    main_body_length: int = 0
    data_preparation_cost: int = 0
    compute_cost: int = 0
    compute_vs_store_ratio: float = 0
    vst_cost: int = 0
    leading_latency_without_reuse: int = 0
    data_reuse_times: int = 0

    instruction_info: InstructionInfo = 0
    gemm_shape: List[int] = field(default_factory=list)

    subthread_num: int = 4
    input_src: AddrDomain = AddrDomain.LOCAL
    output_dest: AddrDomain = AddrDomain.L3


def gemm_kernel(m: int, n: int, k: int, dtype: DType = DType.FP16):
    computer_count = 0
    st_count = 0
    scalar_count = 0

    if dtype == DType.FP16:
        slice_m = 64
        slice_n = 64
        slice_k = 32
        eleByte = 2
    elif dtype == DType.INT8 or dtype == DType.FP8:
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

    def get_memory_stat(self) -> Generator[DataflowActionMemoryStat, None, None]:
        gemm_shape = self.get_valid_shape()
        self.core_cost.gemm_shape = gemm_shape
        dtype = self.get_dtype()
        self.core_cost.dtype = dtype
        self.core_cost.instruction_info = gemm_kernel(
            gemm_shape[0], gemm_shape[1], gemm_shape[2], dtype=dtype)
        self._get_mem_src()
        self._get_l0_occupation()
        self._get_basic_cost()
        self._get_main_body_length()
        bpe = self.core_cost.instruction_info.dtype.get_bpe()
        m = 0
        n = 1
        k = 2
        dtype = self.core_cost.instruction_info.dtype
        mac_array = self.config.compute.thread_2d_throughput[dtype]

        lhs_tensor = self.inputs[0].tensor[0]
        lhs_access = DataflowActionMemoryAccess(
            lhs_tensor.addr, gemm_shape[m] * gemm_shape[n] * bpe, "r")
        lhs_gen = self._iter_access_gen([lhs_access])
        next(lhs_gen)

        rhs_tensor = self.inputs[1].tensor[0]
        rhs_access = DataflowActionMemoryAccess(
            rhs_tensor.addr, gemm_shape[n] * gemm_shape[k] * bpe, "r")
        rhs_gen = self._iter_access_gen([rhs_access])
        next(rhs_gen)

        if len(self.outputs) > 0:
            out_tensor = self.outputs[0].tensor[0]
            out_access = DataflowActionMemoryAccess(
                out_tensor.addr, gemm_shape[m] * gemm_shape[n] * bpe, "r")
            out_gen = self._iter_access_gen([out_access])
            next(out_gen)

        stat_ref = 0
        total_latency = 0
        mid_level_ins_shape = self.core_cost.instruction_info.mid_level_ins_shape
        for i in range(0, self.core_cost.instruction_info.mid_level_ins_num, self.core_cost.main_body_length):
            mid_ins_num = min(self.core_cost.main_body_length,
                              self.core_cost.instruction_info.mid_level_ins_num - i)
            vst_ins_num = (i + self.core_cost.main_body_length) // (self.core_cost.compute_vs_store_ratio) - i // (
                self.core_cost.compute_vs_store_ratio
            )
            lhs_size = mid_ins_num * \
                mid_level_ins_shape[m] * mid_level_ins_shape[k] * bpe
            lhs_read = DataflowActionMemoryStat(
                total_count=lhs_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=self.core_cost.input_src,
                rw="r",
                relative_ts=stat_ref,
                memory_access_list=lhs_gen.send(lhs_size),
                name="lhs",
            )
            yield lhs_read
            rhs_size = mid_ins_num * \
                mid_level_ins_shape[n] * mid_level_ins_shape[k] * bpe
            rhs_read = DataflowActionMemoryStat(
                total_count=rhs_size,
                master=DataflowActionType.XPU,
                src=AddrDomain.L0,
                dst=self.core_cost.input_src,
                rw="r",
                relative_ts=stat_ref,
                memory_access_list=rhs_gen.send(rhs_size),
                name="rhs",
            )
            yield rhs_read

            scalar_ops = self.core_cost.instruction_info.scalar_ins_num / \
                self.core_cost.instruction_info.mid_level_ins_num*mid_ins_num
            compute_stat1 = DataflowActionComputeStat(
                name="scalar",
                compute_scalar_cycle=scalar_ops,
                relative_ts=stat_ref + max(lhs_read.latency, rhs_read.latency),
            )
            yield compute_stat1
            scalar_cost = compute_stat1.latency

            compute_mac = self.core_cost.vmm_basic_cost_per_cycle * mid_ins_num

            VMM = 10  # KG->L0 latency
            compute_ref = (
                stat_ref
                + lhs_read.leading_latency
                # + (max(lhs_read.latency - lhs_read.leading_latency, rhs_read.latency - rhs_read.leading_latency))
                # / self.core_cost.subthread_num
            )
            compute_stat2 = DataflowActionComputeStat(
                name="mac",
                compute_2d_ops={dtype: compute_mac * 2},
                relative_ts=compute_ref,
            )
            yield compute_stat2

            compute_vmm = DataflowActionComputeStat(
                name="vmm",
                compute_nop_cycle=VMM,
                relative_ts=compute_ref + compute_stat2.latency,
            )
            yield compute_vmm
            compute_latency = compute_ref + compute_stat2.latency + compute_vmm.latency
            # out_ref = compute_ref + compute_latency / self.core_cost.subthread_num
            if self.core_cost.dtype == DType.FP8:
                scaling_ref = compute_ref
                compute_scaling = DataflowActionComputeStat(
                    name="scaling",
                    compute_1d_ops={DType.FP32: (
                        mid_level_ins_shape[m] * mid_level_ins_shape[n] * mid_ins_num)},
                    relative_ts=scaling_ref,
                )
                yield compute_scaling
                compute_latency = max(
                    compute_latency, scaling_ref + compute_scaling.latency)
            out_ref = compute_ref

            out_latency, out_leading_latency = 0, 0
            out_size = vst_ins_num * \
                mid_level_ins_shape[m] * mid_level_ins_shape[n] * bpe
            if len(self.outputs) > 0 and vst_ins_num > 0:
                out_wirte = DataflowActionMemoryStat(
                    total_count=out_size,
                    master=DataflowActionType.XPU,
                    src=AddrDomain.L0,
                    dst=self.core_cost.output_dest,
                    rw="w",
                    relative_ts=out_ref,
                    memory_access_list=out_gen.send(out_size),
                    name="out",
                )
                yield out_wirte
                out_latency, out_leading_latency = out_wirte.latency, out_wirte.leading_latency
            total_latency = max(
                total_latency,
                stat_ref + max(lhs_read.latency,
                               rhs_read.latency) + scalar_cost,
                compute_latency,
                out_ref + out_latency,
            )
            stat_ref = (
                max(
                    max(
                        stat_ref + max(lhs_read.latency,
                                       rhs_read.latency) + scalar_cost,
                        compute_latency,
                    ),
                    out_ref + out_leading_latency,
                )
                - lhs_read.leading_latency
            )  # update ref

        self.core_cost.latency = total_latency

    def _get_mem_src(self):
        bench_gemm_op_version = self.dataflow_config.get(
            "bench_gemm_op_version", 1)
        if bench_gemm_op_version == 1 or bench_gemm_op_version == 2:
            self.core_cost.input_src = AddrDomain.SHARED
            self.core_cost.output_dest = AddrDomain.SHARED
        elif bench_gemm_op_version == 3:
            self.core_cost.input_src = AddrDomain.LOCAL
            self.core_cost.output_dest = AddrDomain.L3
        else:
            raise RuntimeError(
                f"Unsupported bench_gemm_op_version {bench_gemm_op_version}")

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
        # bpe = instruction_info.dtype.get_bpe()
        # throughput = asdict(self.config.compute.thread_2d_throughput)
        # mac_array = throughput[instruction_info.dtype.name.upper()]
        # mac_array = self.config.compute.thread_2d_throughput[instruction_info.dtype]
        self.core_cost.scalar_basic_cost_per_cycle = (
            instruction_info.scalar_ins_num
            / instruction_info.mid_level_ins_num
            # / (self.config.freq.CORE_CLOCK_DOMAIN * 1e9)
        )
        self.core_cost.vmm_basic_cost_per_cycle = (
            mid_level_ins_shape[m]
            * mid_level_ins_shape[n]
            * mid_level_ins_shape[k]
            # / (mac_array * self.config.freq.CORE_CLOCK_DOMAIN * 1e9)
        )

        self.core_cost.compute_vs_store_ratio = instruction_info.mid_level_ins_num // instruction_info.st_ins_num

    def _get_main_body_length(self):  # step 6
        self.core_cost.main_body_length = min(
            (self.config.memory.l0.IV_SIZE // self.core_cost.ld_iv_occupation),
            (self.config.memory.l0.SMR_SIZE // self.core_cost.ld_smr_occupation),
        )

    def _basic_stat_info(self):
        # TODO: need review
        info = self.core_cost.instruction_info
        m, n, k = info.mid_level_ins_shape  # m,n,k
        dtype = self.core_cost.dtype
        bpe = dtype.get_bpe()
        self.core_cost.tensor_macs[dtype] = m * \
            n * k * info.mid_level_ins_num  # mac
        if dtype == DType.FP8:
            self.core_cost.vector_ops[DType.FP32] = info.scalar_ins_num + \
                m * n * info.mid_level_ins_num
        else:
            self.core_cost.vector_ops[dtype] = info.scalar_ins_num
        self.core_cost.ld_l0_shared = (
            m * k * bpe + n * k * bpe) * info.mid_level_ins_num
        self.core_cost.st_l0_shared = m * n * bpe * info.st_ins_num
