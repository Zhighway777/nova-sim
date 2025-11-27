from nova_platform.config import TOPO, BossaNovaConfig
from nova_platform.dataflow.action.diag_action import (
    DiagDataflow,
    DiagDataflowAction,
    DiagTensor,
    DiagTensorContainer,
    DataflowActionType,
    DiagTriggerID,
    TileInfo,
    BufCnt,
)
from nova_platform.dataflow.dataflow import Dataflow
from nova_platform.dataflow.action.xpu_local_gemm_action import XPULocalGemmAction, XPUSharedGemmAction, QuantType
from nova_platform.dataflow.action.xpu_activation_action import XPUSigmoidAction, XPUGeluAction, XPUReluAction, XPUSiluAction
from nova_platform.dataflow.action.xpu_elementwise_action import XPUAddAction, XPUMulAction
from nova_platform.dataflow.action.xpu_layernorm_action import XPULayernormAction
from nova_platform.dataflow.action.xpu_softmax_action import XPUSoftmaxAction
from nova_platform.dataflow.action.xpu_softmax_backward_action import XPUSoftmaxBackwardAction
from nova_platform.dataflow.action.xpu_gather_action import XPUGatherAction
from nova_platform.dataflow.action.xpu_scatter_action import XPUScatterAction
from nova_platform.dataflow.action.xpu_sdpa_action import XPUSdpaAction
from nova_platform.dataflow.action.xpu_allreduce_action import XPUAllReduceAction
from nova_platform.dataflow.action.xpu_allgather_action import XPUAllGatherAction
from nova_platform.dataflow.action.xpu_allgather_gemm_action import XPUAllGatherGemmAction
from nova_platform.dataflow.action.nop_action import XPUNopAction
from nova_platform.base_model import DType
from nova_platform.benchmark.op_base import Operand

from typing import Dict, Generator, Any, List
from dataclasses import dataclass, field
from enum import Enum

from functools import lru_cache, reduce
import operator


class MemoryManager:
    def __init__(self, total_bytes: int) -> None:
        self.mem_name = "gcu_l3"
        self.total_bytes = total_bytes
        self.base_addr = 0x50000000000
        self.byes_used = 0
        self.mem_list = {}

    def _align(self, bytes):
        return (bytes + 128 - 1) // 128 * 128

    def alloc_mem(self, bytes: int) -> int:
        bytes = self._align(bytes)
        assert bytes + self.byes_used <= self.total_bytes, f"Out of allocatable {self.mem_name} memory"
        addr = self.base_addr + self.byes_used
        self.byes_used += bytes
        self.mem_list[addr] = bytes
        return addr

    def alloc_diag_tensor(self, tensor: DiagTensor):
        elements = reduce(operator.mul, tensor.stride_dims, 1)
        bypes = elements * tensor.bpe
        tensor.addr = self.alloc_mem(bypes)
        return tensor


class  DataflowGenerator:
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        self.config = config
        self.dataflow_config = dataflow_config
        self.kwargs = kwargs
        self.op_type = dataflow_config.get("bench_op_type", "")
        self.dtype = dataflow_config.get("bench_basic_data_type", "fp16")
        self.bpe = DType(self.dtype).get_bpe()
        self.inputs = []
        self.outputs = []
        self.action_cls = DiagDataflowAction
        self.l3_mm = MemoryManager(self.config.memory.l3.TOTAL_SIZE)
        self.tensor_id = 0
        self.tile_info = TileInfo(
            cube_dim=[1, 1, 1],
            grid_dim=[1, 1, 1],
            block_dim=[1, 1, 1],
            tile_shape=[1, 1, 1],
            l2_buf_cnt=BufCnt(),
            l1_buf_cnt=BufCnt(),
        )

    def add_input_tensor(self, dim: List[int], bpe: float = 0, level: int = 3):
        self.inputs.append(DiagTensorContainer(self.tensor_id, [self._add_tensor(dim, bpe, level)]))
        self.tensor_id += 1

    def add_output_tensor(self, dim: List[int], bpe: float = 0, level: int = 3):
        self.outputs.append(DiagTensorContainer(self.tensor_id, [self._add_tensor(dim, bpe, level)]))
        self.tensor_id += 1

    def _add_tensor(self, dim: List[int], bpe: float = 0, level: int = 3):
        assert isinstance(dim, (tuple, list)) and all(isinstance(i, int) and i != 0 for i in dim)
        bpe = self.bpe if bpe == 0 else bpe
        tensor = DiagTensor(0, dims=list(dim), offsets=[0 for _ in dim], stride_dims=[i for i in dim], bpe=bpe)
        self.l3_mm.alloc_diag_tensor(tensor)
        return tensor

    def _generate(self):
        raise NotImplementedError()

    def generate_dataflow(self):
        self._generate()
        action_list = []
        sic_num = self.config.inst_num.NUM_OF_CLUSTER * self.config.inst_num.NUM_OF_DIE
        sip_num = self.config.inst_num.NUM_OF_CORE_PER_CLUSTER
        action_id = 0
        xpu_code = f"{self.op_type}_{self.dtype}"
        for sic_idx in range(sic_num):
            for sip_idx in range(sip_num):
                engine_id = sip_idx + sic_idx * sip_num #the global id of the sip
                data = [sic_idx, sip_idx]
                action_list.append(
                    self.action_cls(
                        code=xpu_code,
                        config=self.config,
                        action_id=action_id,
                        action_type=DataflowActionType.XPU,
                        engine_id=engine_id,
                        engine_sub_id=0,
                        inputs=self.inputs,
                        outputs=self.outputs,
                        dataflow_config=self.dataflow_config,
                        child_action_ids=[],
                        parent_action_ids=[],
                        depth=0,
                        setup_parent_action_id=-1,
                        setup_child_action_id=-1,
                        exe_sem_id=0,
                        setup_sem_id=-1,
                        trigger_id=DiagTriggerID(0, []),
                        input_hints=[],
                        die_id=0,
                        tile_info=self.tile_info,
                        data=data,
                        **self.kwargs,
                    )
                )
                action_id += 1
        return DiagDataflow(
            dataflow_name=f"{self.op_type}_{self.dtype}",
            dataflow_id=0,
            odte_total_bytes=0,
            cdte_total_bytes=0,
            sdte_total_bytes=0,
            action_list=action_list,
        )


class GemmSharedDataflowGenerator(DataflowGenerator):
    #初始化 调用父类初始化
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUSharedGemmAction
    #生成gemm数据流
    def _generate(self):
        #从config中获得shape
        gemm_shape = self.dataflow_config.get("bench_gemm_shape_b_m_k_n", [])
        assert len(gemm_shape) == 4, "wrong input of bench_gemm_shape_b_m_k_n"
        b, m, k, n = gemm_shape
        #获取量化类型
        quant_type = QuantType(self.dataflow_config.get("bench_gemm_quant_type", "No_Quant"))
        if quant_type == QuantType.Wf4g_Af8t:
            group_size = self.dataflow_config.get("bench_gemm_quant_group_size", 32)
            group_num = (k + group_size - 1) // group_size
            lhs_bpe, rhs_bpe, scaling_bpe, res_bpe, bias_bpe = 1, 0.5, 2, 2, 4
        elif quant_type == QuantType.Wf8t_Af8t:
            lhs_bpe, rhs_bpe, res_bpe, bias_bpe = 1, 1, 2, 4
        elif quant_type == QuantType.No_Quant:
            lhs_bpe, rhs_bpe, res_bpe, bias_bpe = [self.bpe] * 4
        else:
            raise RuntimeError(f"Unsupported quant type: {quant_type}")

        # 添加输入输出张量
        # TODO: reversed dims in shape to be compatible with diag tool, fix later
        self.add_input_tensor([k, m, b], lhs_bpe)  # [b, m, k]
        self.add_input_tensor([n, k, b], rhs_bpe)  # [b, k, n]
        if quant_type == QuantType.Wf4g_Af8t:
            self.add_input_tensor([n, group_num, b], scaling_bpe)  # [b, group_num, n]
        self.add_output_tensor([n, m, b], res_bpe)  # [b, m, n]

        # 设定tile info【TODO：理解cube grid block tile的作用】
        self.tile_info = TileInfo(
            cube_dim=[b, m, n],
            grid_dim=[b, m, n],
            block_dim=[1, 1, 1],
            tile_shape=[1, 1, 1],
            l2_buf_cnt=BufCnt(),
            l1_buf_cnt=BufCnt(),
        )


class GemmLocalDataflowGenerator(GemmSharedDataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPULocalGemmAction


class ElementwiseDataflowGenerator(DataflowGenerator):
    _BINARY_OPS = {"add", "mul"}
    _ACTION_MAP = {
        "add": XPUAddAction,
        "mul": XPUMulAction,
        "gelu": XPUGeluAction,
        "relu": XPUReluAction,
        "silu": XPUSiluAction,
        "sigmoid": XPUSigmoidAction,
    }

    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = self._ACTION_MAP[self.op_type]

    def _generate(self):
        shape = self.dataflow_config.get("bench_elementwise_shape")
        assert shape, "bench_elementwise_shape must be provided for elementwise ops"
        operand_count = 2 if self.op_type in self._BINARY_OPS else 1
        for _ in range(operand_count):
            self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class LayerNormDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPULayernormAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_layernorm_shape")
        assert shape, "bench_layernorm_shape must be provided for layernorm"
        self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class SoftmaxDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        if self.op_type == "softmaxbackward":
            self.action_cls = XPUSoftmaxBackwardAction
        else:
            self.action_cls = XPUSoftmaxAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_softmax_shape_b_c")
        assert shape, "bench_softmax_shape_b_c must be provided for softmax"
        self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class GatherDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUGatherAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_gather_shape")
        assert shape, "bench_gather_shape must be provided for gather"
        self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class ScatterDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUScatterAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_scatter_shape")
        assert shape, "bench_scatter_shape must be provided for scatter"
        self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class TransposeDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUNopAction

    def _generate(self):
        tensor_shape = self.dataflow_config.get("bench_transpose_tensor_b_h_w_c")
        assert tensor_shape, "bench_transpose_tensor_b_h_w_c must be provided for transpose"
        self.add_input_tensor(tensor_shape)
        self.add_output_tensor(tensor_shape)


class SdpaDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUSdpaAction

    def _generate(self):
        q_shape = self.dataflow_config.get("bench_sdpa_q_shape")
        kv_shape = self.dataflow_config.get("bench_sdpa_kv_shape")
        assert q_shape and kv_shape, "sdpa requires q/kv shapes"
        self.add_input_tensor(q_shape)
        self.add_input_tensor(kv_shape)
        self.add_output_tensor(q_shape)


class AllReduceDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUAllReduceAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_all_reduce_shape")
        assert shape, "bench_all_reduce_shape must be provided for allreduce"
        self.add_input_tensor(shape)
        self.add_output_tensor(shape)


class AllGatherDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUAllGatherAction

    def _generate(self):
        out_shape = self.dataflow_config.get("bench_allgather_out_shape")
        in_shape = self.dataflow_config.get("bench_allgather_in_shape")
        assert in_shape and out_shape, "allgather requires in/out shapes"
        self.add_input_tensor(in_shape)
        self.add_output_tensor(out_shape)


class AllGatherGemmDataflowGenerator(DataflowGenerator):
    def __init__(self, config: BossaNovaConfig, dataflow_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(config, dataflow_config, **kwargs)
        self.action_cls = XPUAllGatherGemmAction

    def _generate(self):
        shape = self.dataflow_config.get("bench_allgather_gemm_shape_b_m_k_n")
        assert shape and len(shape) == 4, "bench_allgather_gemm_shape_b_m_k_n must be [B,M,K,N]"
        b, m, k, n = shape
        self.add_input_tensor([k, m, b])
        self.add_input_tensor([n, k, b])
        self.add_output_tensor([n, m, b])


DATAFLOW_GENERATOR_MAPPING = {
    "gemm": GemmSharedDataflowGenerator,
    "gemm.shared": GemmSharedDataflowGenerator,
    "gemm.local": GemmLocalDataflowGenerator,
    "add": ElementwiseDataflowGenerator,
    "mul": ElementwiseDataflowGenerator,
    "gelu": ElementwiseDataflowGenerator,
    "relu": ElementwiseDataflowGenerator,
    "silu": ElementwiseDataflowGenerator,
    "sigmoid": ElementwiseDataflowGenerator,
    "layernorm": LayerNormDataflowGenerator,
    "softmax": SoftmaxDataflowGenerator,
    "softmaxbackward": SoftmaxDataflowGenerator,
    "gather": GatherDataflowGenerator,
    "scatter": ScatterDataflowGenerator,
    "transpose": TransposeDataflowGenerator,
    "sdpa": SdpaDataflowGenerator,
    "allreduce": AllReduceDataflowGenerator,
    "allgather": AllGatherDataflowGenerator,
    "allgather_gemm": AllGatherGemmDataflowGenerator,
}


def generate_dataflow(config: BossaNovaConfig, dataflow_config: Dict[str, Any], topo: TOPO, case_id: int):
    bench_op_type = dataflow_config.get("bench_op_type", "")
    generator_cls = DATAFLOW_GENERATOR_MAPPING[bench_op_type]
    generator = generator_cls(config, dataflow_config, topo=topo, case_id=case_id)
    return generator.generate_dataflow()
