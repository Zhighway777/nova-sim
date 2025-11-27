import optrace_benchmark as optrace
from tests.test_base import CaseInfo
from nova_platform.base_model import DType
from nova_platform.config import TOPO
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from pathlib import Path
import re


class MemoryAllocator:
    def __init__(self):
        self.addr_ref = 5 * 2**40 # 5TB
        self.addr_table = {}
    
    #input operand_id_list, operand_shape_list and result_id_list, output_shape_list and dtype_list
    #output: input address list and output address list
    def allocate(self, ssa_ids, shapes, dtype):
        if not isinstance(ssa_ids, list):
            ssa_ids = [ssa_ids]
            shapes = [shapes]
        
        addr_list = []
        for ssa_id, shape in zip(ssa_ids, shapes):
            if (ssa_id in self.addr_table):
                addr_list.append(self.addr_table[ssa_id])
            else:
                offset = self.get_shape_offset(shape, dtype)
                address = self.addr_ref
                self.addr_ref += offset
                self.addr_table[ssa_id] = address
                addr_list.append(address)
        return addr_list

    def get_shape_offset(self, shape, dtype:DType) -> int:
        bpe = dtype.get_bpe()
        total_elem = 1
        for dim in shape:
            total_elem *= dim
        offset = total_elem * bpe
        return offset
        
class TensorInfo:
    def __init__(self, tensor_id, shape, data_type):
        self.tensor_id = tensor_id
        self.shape = shape
        self.data_type = data_type

    @classmethod
    def from_tensor(cls, tensor):
        tensor_id = tensor.getTensorID()
        shape = tensor.getDims()
        dtype_str = tensor.getDataType().name

        # TODO: Add more data types as needed
        dtype_map={
            'F8':DType.FP8,
            'F16':DType.FP16,
            'F32':DType.FP32,
        }
        dtype=dtype_map[dtype_str]
        
        return cls(tensor_id, shape, dtype)

class ScalarInfo:
    def __init__(self, value):
        self.value = value
    @classmethod
    def from_scalar(cls, scalar):
        value = scalar.getValue()
        return cls(value)
    


@dataclass
class OperationInfo:
    result_ids: List[int]
    result_shapes: List[int]
    op_ids: List[int]
    op_shapes: List[List[int]]
    #不支持混合精度量化
    dtype: DType
    caseinfo: CaseInfo

    def __str__(self):
        caseinfo_str = (
            f"OperationInfo(\n"
            f"  op_name: {self.caseinfo.optype},\n"
            "\n"
            f"  dtype: {self.dtype},\n"
            f"  op_ids: {self.op_ids},\n"
            f"  op_shapes: {self.op_shapes},\n"
            "\n"
            f"  result_ids: {self.result_ids},\n"
            f"  result_shapes: {self.result_shapes},\n"
            "\n"
            f"  case_shape: {self.caseinfo.shape},\n"
            f"  cache_enable: {self.caseinfo.enable_cache},\n"
            "\n"
            f"  topo: {self.caseinfo.topo},\n"
            f")"
        )
        return caseinfo_str
        
    # generate dim_map for transpose
    @staticmethod
    def get_dim_map(dim_list:List[int]):
        dim_map = [0, 1, 2, 3]
        dim_map[3-dim_list[0]], dim_map[3-dim_list[1]] =  dim_map[3-dim_list[1]], dim_map[3-dim_list[0]]
        return dim_map

    @classmethod
    def from_instruction(cls, instr: optrace.instruct) -> "OperationInfo":
        optrace_name = instr.getOpname()
        operands = instr.getOperands()

        # Parse operand details using TensorInfo
        tensor_info = [TensorInfo.from_tensor(op) for op in operands if op.getValueType().name == "TENSOR"]
        scalar_info = [op.getDataI32() for op in operands if op.getValueType().name == "SCALAR"]
        

        # Reflect to caseinfo optype and shape
        caseinfo_shape, optype = cls._caseinfo_shape(optrace_name, [op.shape for op in tensor_info], scalar_info)

        # Get result info using TensorInfo
        results = instr.getResults()
        result_info = [TensorInfo.from_tensor(r) for r in results]

        # Create CaseInfo instance
        caseinfo = CaseInfo(
            optype=optype, 
            shape=caseinfo_shape, 
            expected_res=None,
            fun=None,
            tag=["Fusion_DecodeLayer"], 
            dtype=result_info[0].data_type,  
        )
        
        return cls(
            result_ids=[r.tensor_id for r in result_info],
            result_shapes=[r.shape for r in result_info],
            op_ids=[op.tensor_id for op in tensor_info],
            op_shapes=[op.shape for op in tensor_info],

            caseinfo=caseinfo,
            dtype=result_info[0].data_type
        )
    
    @staticmethod
    def _caseinfo_shape(optrace_name: str, op_shape: List[List[int]], op_dim: List[int]) -> List[int]:
        """Determine the shape for the caseinfo based on the operation name."""
        #TODO: Add more cases as needed
        #caseinfo name is not same as optrace_name
        if optrace_name in ["dot"]:
            optype = "gemm"
            caseinfo_shape = op_shape[0][:]+ op_shape[2]
        elif optrace_name in ["mbedding", "embedding"]:
            optype = "gather"
            caseinfo_shape = op_shape[0][:]+ op_shape[1][:]
        elif optrace_name in ["flash_attention_fusion"]:
            optype = "sdpa"
            caseinfo_shape = [op_shape[0][:]]+ [op_shape[1][:]]+[True]
        elif optrace_name in ["layer_norm"]:
            optype = "layernorm"
            caseinfo_shape = op_shape[0][1:]
        else:#caseinfo name is same as optrace_name
            if optrace_name in ["add", "gelu"]:
                caseinfo_shape = [1, op_shape[0][1], 1, op_shape[0][2]]
                optype = optrace_name
            elif optrace_name in ["transpose"]:
                dim_map = OperationInfo.get_dim_map(op_dim)
                caseinfo_shape = [op_shape[0][:]] + [dim_map[:]]
                optype = optrace_name
            elif optrace_name in ["allreduce"]:
                optype = "allreduce"
                caseinfo_shape = op_shape[0][:]
            elif optrace_name in ["allgather"]:
                optype = "allgather"
                #TODO：need to review ‘32’ 
                caseinfo_shape = [[op_shape[0][0], op_shape[0][1]*32, op_shape[0][2], op_shape[0][3]]] + [op_shape[0][:]]
            elif optrace_name in ["allgather_gemm"]:
                optype = "allgather_gemm"
                caseinfo_shape = op_shape[0][:]+ op_shape[2]
        return caseinfo_shape, optype


'''
input: optrace_path 
output: [CaseInfo with (input_addr_list, output_addr_list) in it]
'''
def get_caseinfo_list(op_trace_path: str, topo=TOPO.STANDALONE, enable_cache=True) -> Tuple[List[CaseInfo], List[OperationInfo]]:
    opt = optrace.optrace(op_trace_path)
    mod = opt.getModule()
    instruct_list = mod.getInstructs()

    alloc = MemoryAllocator()
    caseinfo_list = []
    op_info_list = []
    for instr_dump in instruct_list:
        case_in_trace = OperationInfo.from_instruction(instr_dump)
        case_in_trace.caseinfo.topo = topo
        case_in_trace.caseinfo.enable_cache = enable_cache
        if case_in_trace.caseinfo.optype == "gather":
           case_in_trace.caseinfo.mu = 1e-9
        case_in_trace.caseinfo.input_addr = alloc.allocate(case_in_trace.op_ids, case_in_trace.op_shapes, case_in_trace.dtype)

        case_in_trace.caseinfo.output_addr = alloc.allocate(case_in_trace.result_ids, case_in_trace.result_shapes, case_in_trace.dtype)
        caseinfo_list.append(case_in_trace.caseinfo)
        op_info_list.append(case_in_trace)
    return caseinfo_list, op_info_list


#print to parse optrace
'''
Could check the allocation of memory and the CaseInfo object
'''
def display_fusion_info(outdir, op_info_list: List[OperationInfo]):
    with open(f"{outdir}/optrace_file_output.txt", "w") as f:
        for op_info in op_info_list:
            try:
                f.write(f"{op_info}\n")
                f.write(f"mu: {op_info.caseinfo.mu}\n")
                f.write(f"Input Address: {op_info.caseinfo.input_addr}\n")            
                f.write(f"Output Address: {op_info.caseinfo.output_addr}\n")   

                f.write(f"Input Address Hex: {[hex(addr) for addr in op_info.caseinfo.input_addr]}\n")
                f.write(f"Output Address Hex: {[hex(addr) for addr in op_info.caseinfo.output_addr]}\n")       
                f.write("-" * 40 + "\n") 
            except Exception as e:
                f.write(f"Error processing case: {op_info}\n")
                f.write(f"Exception: {e}\n")
                f.write("-" * 40 + "\n")
