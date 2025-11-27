import pytest

from bossa_nova.benchmark.op_trace_obsolete import parse_op_trace

import logging

logger = logging.getLogger(__name__)


@pytest.mark.skip
def test_parse_op_trace_string(outdir):
    op_trace_str = """58387:0:0 %286:<2048x1x12x384xf16> = torch.2_1_0.aten::cat(list{%274:<2048x1x12x128xf16>{128,3145728,262144,1}, %270:<2048x1x12x128xf16>{1,3145728,262144,2048}, %311:<2048x1x12x128xf16>{128,3145728,262144,1}}, 3:i32)
58388:1:0 %252:<1x12x2048x2048xf16> = torch.2_1_0.aten::_softmax(%251:<1x12x2048x2048xf16>, -1:i32, False:pred)
"""
    op_trace_list = list(parse_op_trace(op_trace_str))
    op_trace = op_trace_list[0]

    assert op_trace.process_id == 58387
    assert op_trace.rank_id == 0
    assert op_trace.stream_id == 0
    assert op_trace.domain_name == "torch"
    assert op_trace.version_number == "2_1_0"
    assert op_trace.op_name == "aten::cat"

    assert len(op_trace.operands) == 2
    assert op_trace.operands[0].struct_name == "list"
    assert op_trace.operands[0].values[0].id == "%274"
    assert op_trace.operands[0].values[0].shape == [2048, 1, 12, 128]
    assert op_trace.operands[0].values[0].dtype == "f16"
    assert op_trace.operands[0].values[0].stride == [128, 3145728, 262144, 1]

    assert op_trace.operands[0].values[1].id == "%270"
    assert op_trace.operands[0].values[1].shape == [2048, 1, 12, 128]
    assert op_trace.operands[0].values[1].dtype == "f16"
    assert op_trace.operands[0].values[1].stride == [1, 3145728, 262144, 2048]

    assert op_trace.operands[0].values[2].id == "%311"
    assert op_trace.operands[0].values[2].shape == [2048, 1, 12, 128]
    assert op_trace.operands[0].values[2].dtype == "f16"
    assert op_trace.operands[0].values[2].stride == [128, 3145728, 262144, 1]

    assert op_trace.operands[1].value_name == ""
    assert op_trace.operands[1].value == "3"
    assert op_trace.operands[1].dtype == "i32"

    assert len(op_trace.results) == 1
    assert op_trace.results[0].id == "%286"
    assert op_trace.results[0].shape == [2048, 1, 12, 384]
    assert op_trace.results[0].dtype == "f16"

    op_trace = op_trace_list[1]
    assert op_trace.process_id == 58388
    assert op_trace.rank_id == 1
    assert op_trace.stream_id == 0
    assert op_trace.domain_name == "torch"
    assert op_trace.version_number == "2_1_0"
    assert op_trace.op_name == "aten::_softmax"

    assert len(op_trace.operands) == 3
    assert op_trace.operands[0].id == "%251"
    assert op_trace.operands[0].shape == [1, 12, 2048, 2048]
    assert op_trace.operands[0].dtype == "f16"
    assert op_trace.operands[0].stride == []
    assert op_trace.operands[0].offset == 0

    assert op_trace.operands[1].value_name == ""
    assert op_trace.operands[1].value == "-1"
    assert op_trace.operands[1].dtype == "i32"

    assert op_trace.operands[2].value_name == ""
    assert op_trace.operands[2].value == "False"
    assert op_trace.operands[2].dtype == "pred"

    assert len(op_trace.results) == 1
    assert op_trace.results[0].id == "%252"
    assert op_trace.results[0].shape == [1, 12, 2048, 2048]
    assert op_trace.results[0].dtype == "f16"
    assert op_trace.operands[0].stride == []
    assert op_trace.operands[0].offset == 0


@pytest.mark.skip
def test_parse_op_trace_file(outdir):
    op_trace_path = "tests/resources/train_gpt3_175b_1layer_tp8.op_trace.txt"
    with open(op_trace_path) as f:
        input_str = f.read()
    op_traces = parse_op_trace(input_str)
    for trace in op_traces:
        print(trace)
