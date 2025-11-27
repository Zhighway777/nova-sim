from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, List, Tuple, Optional, Union, Dict, Any

import re


class DataType(str, Enum):
    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"
    f8e4m3 = "f8e4m3"
    f8e5m2 = "f8e5m2"
    u32 = "u32"
    i32 = "i32"
    u16 = "u16"
    i16 = "i16"
    u8 = "u8"
    i8 = "i8"
    b8 = "b8"
    str = "str"


@dataclass
class Scalar:
    value_name: str = ""
    value: Any = None
    dtype: DataType = ""


@dataclass
class Tensor:
    id: str
    dtype: DataType
    shape: List[int] = field(default_factory=list)
    stride: List[int] = field(default_factory=list)
    offset: int = 0


@dataclass
class Struct:
    value_name: str
    struct_name: str
    values: List[Union[Tensor, Scalar]]


@dataclass
class OpTraceRecord:
    process_id: Union[int, None]
    rank_id: Union[int, None]
    stream_id: Union[int, None]
    domain_name: str = ""
    version_number: str = ""
    op_name: str = ""
    operands: List[Union[Struct, Tensor, Scalar]] = field(default_factory=list)
    results: List[Union[Struct, Tensor, Scalar]] = field(default_factory=list)


def parse_op_trace(traces: str) -> Generator[OpTraceRecord, None, None]:
    record_pattern = r"(?P<op_ids>\w+:\w+:\w+)\s+(?P<results>.+)\s+=\s*(?P<operator>.+)\((?P<args>.+)\)"

    def parse_value(arg):
        if "list{" in arg:
            args = split_args(arg[5:-1])
            values = []
            for v in args:
                if value := parse_value(v):
                    values.append(value)
            return Struct(value_name="", struct_name="list", values=values)
        if match := re.search(r"(%\d+):<(\w+)>\s*(\{[\d,]+\})?\s*(\+\d+)?\s*", arg):  # tensor
            shape_n_dtype = match.group(2).split("x")
            stride = match.group(3)
            stride = [int(i) for i in stride[1:-1].split(",")] if stride else []
            offset = match.group(4)
            offset = int(offset[1:]) if offset else 0

            t = Tensor(
                id=match.group(1),
                dtype=shape_n_dtype[-1],
                shape=[int(i) for i in shape_n_dtype[:-1]],
                stride=stride,
                offset=offset,
            )
            if len(t.shape) == 0:
                t.shape = [1]
            return t
        if match := re.search(r"([\w.]+)\s*=\s*([\w.-]+):([\w.]+)", arg):  # scalar with name
            s = Scalar(value_name=match.group(1), value=match.group(2), dtype=match.group(3))
            return s
        if match := re.search(r"([\w.-]+):([\w.]+)", arg):  # scalar without name
            s = Scalar(value_name="", value=match.group(1), dtype=match.group(2))
            return s
        return None

    def split_args(args):
        stack = []
        splits = []
        start_index = 0
        end_index = len(args)
        for i, char in enumerate(args):
            if char == "{":
                stack.append(i)
            elif char == "}":
                if stack:
                    stack.pop()
            elif char == ",":
                if not stack:
                    splits.append(args[start_index:i].strip())
                    start_index = i + 1
        splits.append(args[start_index:end_index].strip())
        return splits

    def parse_operands(args):
        operands = []
        args = split_args(args)
        for arg in args:
            if operand := parse_value(arg):
                operands.append(operand)
        return operands

    trace_lines = traces.split("\n")
    for trace_str in trace_lines:
        # print(trace_str)

        match = re.search(record_pattern, trace_str)
        if not match:
            continue
        op_ids = match.group("op_ids").split(":")
        op_ids = [int(i) if i.isdigit() else i for i in op_ids]
        results = match.group("results")
        operator = match.group("operator")
        args = match.group("args")
        operator = operator.split(".")
        operands = parse_operands(args)
        results = parse_operands(results)
        record = OpTraceRecord(
            process_id=op_ids[0],
            rank_id=op_ids[1],
            stream_id=op_ids[2],
            domain_name=operator[0],
            version_number=operator[1],
            op_name=operator[2],
            operands=operands,
            results=results,
        )
        yield record
