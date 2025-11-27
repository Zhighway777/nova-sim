from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class _ValueType:
    name: str


@dataclass
class _DataType:
    name: str


_TENSOR = _ValueType("TENSOR")
_SCALAR = _ValueType("SCALAR")


class _Operand:
    def __init__(self, tensor_id: int, dims: List[int], dtype: str):
        self._tensor_id = tensor_id
        self._dims = dims
        self._dtype = _DataType(dtype)

    def getValueType(self):
        return _TENSOR

    def getTensorID(self):
        return self._tensor_id

    def getDims(self):
        return list(self._dims)

    def getDataType(self):
        return self._dtype


class _ScalarOperand:
    def __init__(self, value: int):
        self._value = value

    def getValueType(self):
        return _SCALAR

    def getDataI32(self):
        return self._value


class _Result(_Operand):
    pass


class instruct:
    def __init__(self, opname: str, operands: List[object], results: List[_Result]):
        self._opname = opname
        self._operands = operands
        self._results = results

    def getOpname(self):
        return self._opname

    def getOperands(self):
        return list(self._operands)

    def getResults(self):
        return list(self._results)


class _Module:
    def __init__(self, instructs: List[instruct]):
        self._instructs = instructs

    def getInstructs(self):
        return list(self._instructs)


class optrace:
    """
    Bare-bones parser for the text-based optrace format used in the tests.
    The real optrace_benchmark package contains a full protobuf parser; we only
    need enough to drive the unit tests with the simplified traces under
    ``tests/resources``.
    """

    def __init__(self, trace_path: str):
        trace = Path(trace_path)
        if not trace.exists():
            raise FileNotFoundError(trace_path)
        self._module = _Module(_parse_trace(trace.read_text().splitlines()))

    def getModule(self):
        return self._module


def _parse_trace(lines: List[str]) -> List[instruct]:
    instructions: List[instruct] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        lhs = lhs.split("%", 1)[-1]  # drop timeline prefix
        result_tokens = [tok.strip() for tok in lhs.split(",") if tok.strip()]
        results = [_parse_operand(tok, as_result=True) for tok in result_tokens]

        op_spec, args = rhs.strip().split("(", 1)
        op_name = op_spec.strip().split(".")[-1]
        args = args.rsplit(")", 1)[0]
        operands = []
        for token in _split_args(args):
            token = token.strip()
            if not token:
                continue
            if token.startswith("%"):
                operands.append(_parse_operand(token, as_result=False))
            else:
                try:
                    operands.append(_ScalarOperand(int(token)))
                except ValueError:
                    continue
        instructions.append(instruct(op_name, operands, results))
    return instructions


def _split_args(arg_str: str) -> List[str]:
    chunks = []
    depth = 0
    current = []
    for ch in arg_str:
        if ch == "," and depth == 0:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


def _parse_operand(token: str, as_result: bool) -> object:
    # token format: %5:<1x384x12288xf16>
    token = token.strip()
    if token.startswith("%"):
        token = token[1:]
    tensor_id, _, remainder = token.partition(":")
    tensor_id = int(tensor_id)
    dims, dtype = _parse_shape(remainder)
    operand_cls = _Result if as_result else _Operand
    return operand_cls(tensor_id, dims, dtype)


def _parse_shape(payload: str):
    payload = payload.strip()
    if payload.startswith("<") and payload.endswith(">"):
        payload = payload[1:-1]
    parts = [p for p in payload.split("x") if p]
    if not parts:
        return [], "F16"
    dtype = parts[-1].upper()
    dims = []
    for part in parts[:-1]:
        try:
            dims.append(int(part))
        except ValueError:
            dims.append(1)
    return dims, dtype


__all__ = ["optrace", "instruct"]
