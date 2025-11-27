import optrace_benchmark as optrace
from tests.test_base import CaseInfo
from nova_platform.base_model import DType
from dataclasses import dataclass
from typing import List,  Tuple
import re

result = instr.getResultItem(0)
def parse_string(string: str) -> Tuple[int, List[int], DType]:
        match = re.match(r"%(\d+):<([\dx]+)xf(\d+)>", string)
        if match:
            ssa_id = int(match.group(1))
            shape_str = match.group(2)
            shape = [int(dim) for dim in shape_str.split('x')]
            dtype_str = match.group(3)
            if dtype_str == "16":
                dtype = DType.FP16
            elif dtype_str == "32":
                dtype = DType.FP32
            return ssa_id, shape, dtype
        else:
            raise ValueError(f"Invalid result string format: {string}")
        
res = parse_string("%22:<1x12x3072x128xf16>")