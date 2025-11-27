import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Tuple
from nova_platform.config import TOPO, BossaNovaConfig
from nova_platform.cost_service.compute.base_compute_model import BaseCoreStat, DataflowAction
from nova_platform.dataflow.dataflow import Dataflow
from nova_platform.base_model import DataflowActionMemoryAccess, DataflowActionType


@dataclass
class DiagTensorId:
    id_key: int
    id_val: int


@dataclass
class DiagTensor:
    addr: int
    offsets: List[int]
    dims: List[int]
    stride_dims: List[int]
    bpe: int


@dataclass
class DiagTensorContainer:
    id: DiagTensorId
    tensor: List[DiagTensor]


@dataclass
class DiagTriggerID:
    id: int
    values: List[int]


@dataclass
class BufCnt:
    lhs: int = 1
    rhs: int = 1
    res: int = 1
    traverse_dim_order: str = ""


@dataclass
class TileInfo:
    cube_dim: List[int] = field(default_factory=list)
    grid_dim: List[int] = field(default_factory=list)
    block_dim: List[int] = field(default_factory=list)
    tile_shape: List[int] = field(default_factory=list)
    l2_buf_cnt: BufCnt = None  # lhs, rhs, res,
    l1_buf_cnt: BufCnt = None


@dataclass
class DiagDataflowAction(DataflowAction):
    config: BossaNovaConfig
    action_id: int
    topo: TOPO
    action_type: DataflowActionType
    engine_id: int
    engine_sub_id: int
    inputs: List[DiagTensorContainer]
    outputs: List[DiagTensorContainer]
    child_action_ids: List[int]
    # child_actions: List["DiagAction"]
    parent_action_ids: List[int]
    # parent_actions: List["DiagAction"]
    depth: int
    setup_parent_action_id: int
    setup_child_action_id: int
    exe_sem_id: int
    setup_sem_id: int
    trigger_id: DiagTriggerID
    input_hints: List[int]
    tile_info: TileInfo
    dataflow_config: dict[str, str]
    die_id: int
    case_id: int

    def get_die_id(self):
        return (self.engine_id//self.config.inst_num.NUM_OF_CORE_PER_CLUSTER) % self.config.inst_num.NUM_OF_DIE
        # return self.get_cluster_id() % 2

    def get_action_id(self):
        return self.action_id

    def get_engine_id(self):
        return self.engine_id

    def get_cluster_id(self):
        return self.engine_id//self.config.inst_num.NUM_OF_CORE_PER_CLUSTER//self.config.inst_num.NUM_OF_DIE

    def get_engine_sub_id(self):
        return self.engine_sub_id

    def get_port_id(self):
        return 0

    def get_child_ids(self):
        return self.child_action_ids

    def get_parent_ids(self):
        return self.parent_action_ids

    def get_action_type(self) -> DataflowActionType:
        return self.action_type

    def get_core_stat(self):
        return self.core_cost

    def _iter_tensor_addr(self, base_addr, tensor: DiagTensor, rw) -> Generator[DataflowActionMemoryAccess, None, None]:
        # return addr, size
        bpe = tensor.bpe
        stride0, stride1, stride2, _ = tensor.stride_dims
        stride0 *= bpe
        shape0, shape1, shape2, shape3 = tensor.dims
        shape0 *= bpe
        offset0, offset1, offset2, offset3 = tensor.offsets
        offset0 *= bpe
        for k in range(offset3, offset3+shape3):
            for j in range(offset2, offset2+shape2):
                for i in range(offset1, offset1+shape1):
                    addr = base_addr + \
                        (
                            i * stride0
                            + j * stride1 * stride0
                            + k * stride2 * stride1 * stride0
                            + offset0
                        )
                    yield DataflowActionMemoryAccess(addr, shape0, rw)

    def _iter_access_gen(self, mem_acc_list: List[DataflowActionMemoryAccess]):
        counter = 0
        batch: List[DataflowActionMemoryAccess] = []
        fetch_size = yield batch
        # history = [] # for debug
        # history_raw = [] # for debug
        while True:
            for acc in mem_acc_list:
                # history_raw.append(acc)
                batch.append(acc)
                counter += acc.size
                while True:
                    last_acc = batch.pop()

                    if counter >= fetch_size:
                        last_size_right = counter-fetch_size
                        last_size_left = last_acc.size - last_size_right
                        batch.append(
                            DataflowActionMemoryAccess(
                                base_addr=last_acc.base_addr,
                                size=last_size_left,
                                rw=last_acc.rw
                            )
                        )

                        new_base = last_acc.base_addr + last_size_left
                        new_size = last_acc.size - last_size_left
                        counter = fetch_size
                        # history.append(batch)
                        fetch_size = yield batch
                        if new_size > 0:
                            batch = [
                                DataflowActionMemoryAccess(
                                    base_addr=new_base,
                                    size=new_size,
                                    rw=last_acc.rw
                                )
                            ]
                            counter = new_size
                        else:
                            batch = []
                            counter = 0
                            break
                    else:
                        batch.append(last_acc)
                        break
            if batch:
                # history.append(batch)
                yield batch


@dataclass
class DiagDataflow(Dataflow):
    dataflow_name: str
    dataflow_id: int
    odte_total_bytes: int
    cdte_total_bytes: int
    sdte_total_bytes: int
    action_list: List[DiagDataflowAction]

    def __post_init__(self):
        self._build_dag(self.action_list)


input_output_pattern = r"(\{[\d+.\d+:\{\{\d+,\d+,\d+,\d+\},\{\d+,\d+,\d+,\d+\},\{0-9a-fA-F\}\}]+\}\},\}|\{\})"
pattern = [
    r'([0-9a-fA-F+,]+) : (cdte|odte|sdte|xpu)\{(\d+),([^,]+),',
    input_output_pattern, ',', input_output_pattern,
    r',([\{[0-9a-fA-F+,]+\}|\{\}])',
    r',([\{[0-9a-fA-F+,]+\}|\{\}])',
    r',depth:(\{[0-9a-fA-F]+\})',
    r',setup:(\{-?[0-9a-fA-F]+,-?[0-9a-fA-F]+\})',
    r',sem:(\{-?\d+,-?\d+\})',
    r',trigger:(\{-?\d+->[-?\d+,]+\}|\{->\})',
    r',hint:(\{-?\d+,-?\d+\,}|\{\})',
    r',(vc\d+|ctx\d+)',
    r',(\{-?\d+,-?\d+,-?\d+,-?\d+,\},\{-?\d+,-?\d+,-?\d+,-?\d+,\},\{-?\d+,-?\d+,-?\d+,-?\d+,\},\{-?\d+,-?\d+,-?\d+,-?\d+,\},|\{[\d+,]+,\})',
]
pattern = "".join(pattern)


def _parse_input_output(input_string: str) -> List[DiagTensorContainer]:
    pattern = r'(\d+\.\d+):\{\{.*?\}\}'

    # 查找所有匹配项
    matches = re.findall(pattern, input_string)

    # 初始化结果字典
    result_dict = {}

    # 处理每个匹配项
    actions = []
    for match in matches:
        #   # dec param # hex PA
        key_pattern = r'(\d+).(\d+):\{\{(\d+,\d+,\d+,\d+)\},\{(\d+,\d+,\d+,\d+)\},\{([0-9a-fA-F]+)\}'
        key_match = re.search(key_pattern, input_string)
        if key_match:
            actions.append(
                DiagTensorContainer(
                    DiagTensorId(
                        int(key_match.group(1)), int(key_match.group(2))),
                    [
                        DiagTensor(
                            int(key_match.group(5), base=16),  # hex
                            [int(offset)
                             for offset in key_match.group(3).split(',')],  # dec
                            [int(dim)
                             for dim in key_match.group(4).split(',')],  # dec
                            [int(dim)
                             for dim in key_match.group(4).split(',')],  # TODO: stride dim, need upstream to provide!!! fix later,
                            2,  # TODO: bpe
                        )
                    ]
                )
            )
    return actions


def parse_diag_dataflow(path: str) -> DiagDataflow:
    with open(path) as f:
        input_str = f.read()
    lines = input_str.split('\n')
    line0 = re.match(
        r'# dataflow:\{name\{(\w+)\},id\{(\w+)\}\}', lines[0]).groups()
    line1 = re.match(
        r'# odte:\{(\w+)\},cdte:\{(\w+)\},sdte:\{(\w+)\}', lines[1]).groups()
    dataflow_name, dataflow_id = line0[0], int(line0[1])  # dec
    (
        odte_total_bytes,
        cdte_total_bytes,
        sdte_total_bytes
    ) = [int(i) for i in line1]  # dec

    actions = []
    for i, line in enumerate(lines[2:]):
        if line.strip() == '':
            continue
        match = re.search(pattern, line)
        if match:
            (
                action_id,  # hex
                action_type,  # str
                engine_id,  # hex
                dte_op_or_xpu_code,  # str
                inputs,
                outputs,
                child_action_ids,  # hex
                parent_action_ids,  # hex
                depth,  # hex
                setup_action_id_pair,  # hex
                sem_id_pair,  # dec
                trigger_id,  # dec
                input_hints,  # hex
                vc_ctx_id,
                vc_ctx_data
            ) = match.groups()

            action_id = int(action_id, base=16)
            action_type = DataflowActionType(action_type.upper())
            engine_id = int(engine_id, base=16)

            inputs = _parse_input_output(inputs)
            outputs = _parse_input_output(outputs)

            child_action_ids = re.findall(r'-?[0-9a-fA-F]+', child_action_ids)
            child_action_ids = [int(i, base=16)
                                for i in child_action_ids]  # hex
            parent_action_ids = re.findall(
                r'-?[0-9a-fA-F]+', parent_action_ids)
            parent_action_ids = [int(i, base=16)
                                 for i in parent_action_ids]  # hex
            depth = int(re.findall(
                r'-?[0-9a-fA-F]+', depth)[0], base=16)  # hex

            # '{-1,-1}'
            setup_parent_action_id, setup_child_action_id = re.findall(
                r'-?[0-9a-fA-F]+', setup_action_id_pair)
            setup_parent_action_id, setup_child_action_id = int(
                setup_parent_action_id, base=16), int(setup_child_action_id, base=16)  # hex

            # '{0,-1}'
            exe_sem_id, setup_sem_id = re.findall(
                r'-?[0-9a-fA-F]+', sem_id_pair)
            exe_sem_id, setup_sem_id = int(
                exe_sem_id), int(setup_sem_id)  # dec

            # '{0->0,1,2,3,4,}' => [0,0,1,2,3,4]
            trigger_id = re.findall(
                r'-?[0-9a-fA-F]+', trigger_id)
            if len(trigger_id) > 0:
                trigger_id = DiagTriggerID(
                    int(trigger_id[0]),  # dec
                    [int(i) for i in trigger_id[1:]]  # dec
                )
            else:
                trigger_id = None

            # {80000000,80000001,}
            input_hints = re.findall(r'[0-9a-fA-F]+', input_hints)
            input_hints = [int(i, base=16) for i in input_hints]  # hex

            # vc0/xpu0
            engine_sub_id = re.findall(r'[vc|ctx](\d+)', vc_ctx_id)
            if len(engine_sub_id) > 0:
                engine_sub_id = int(engine_sub_id[0])  # dec
            else:
                engine_sub_id = None

            if action_type in [DataflowActionType.CDTE, DataflowActionType.SDTE, DataflowActionType.ODTE]:
                dte_data = re.findall(r'[0-9a-fA-F]+', vc_ctx_data)
                # dec
                param0 = [int(d) for d in dte_data[:4]]
                param1 = [int(d) for d in dte_data[4:8]]
                param2 = [int(d) for d in dte_data[8:12]]
                param3 = [int(d) for d in dte_data[12:]]

                action = DiagDataflowAction(
                    action_id=action_id,
                    action_type=action_type,
                    engine_id=engine_id,
                    engine_sub_id=engine_sub_id,
                    inputs=inputs,
                    outputs=outputs,
                    child_action_ids=child_action_ids,
                    parent_action_ids=parent_action_ids,
                    depth=depth,
                    setup_parent_action_id=setup_parent_action_id,
                    setup_child_action_id=setup_child_action_id,
                    exe_sem_id=exe_sem_id,
                    setup_sem_id=setup_sem_id,
                    trigger_id=trigger_id,
                    input_hints=input_hints,
                    op=dte_op_or_xpu_code,
                    param0=param0, param1=param1, param2=param2, param3=param3
                )
            else:
                xpu_data = re.findall(r'[0-9a-fA-F]+', vc_ctx_data)
                xpu_data = [int(d, base=16) for d in xpu_data]  # hex
                action = DiagDataflowAction(
                    action_id=action_id,
                    action_type=action_type,
                    engine_id=engine_id,
                    engine_sub_id=engine_sub_id,
                    inputs=inputs,
                    outputs=outputs,
                    child_action_ids=child_action_ids,
                    parent_action_ids=parent_action_ids,
                    depth=depth,
                    setup_parent_action_id=setup_parent_action_id,
                    setup_child_action_id=setup_child_action_id,
                    exe_sem_id=exe_sem_id,
                    setup_sem_id=setup_sem_id,
                    trigger_id=trigger_id,
                    input_hints=input_hints,
                    code=dte_op_or_xpu_code,
                    data=xpu_data
                )
            actions.append(action)
        else:
            raise Exception(i, line)

    dataflow = DiagDataflow(
        dataflow_name=dataflow_name,
        dataflow_id=dataflow_id,
        odte_total_bytes=odte_total_bytes,
        cdte_total_bytes=cdte_total_bytes,
        sdte_total_bytes=sdte_total_bytes,
        action_list=actions
    )
    return dataflow


def parse_shape_tile_info(path):
    def to_list(s: str):
        elements = [i.strip() for i in s[1:-1].split(",")]
        l = [int(i) if i.isdigit() else i for i in elements]
        return l

    p = Path(path)
    if not p.exists():
        raise Exception(f"{path} not exists!")
    with open(path) as f:
        input_str = f.read()
    lines = input_str.split("\n")
    if "# SYSTEM" in lines[0]:
        line1 = lines[1].split("-")
    else:
        line1 = lines[0].split("-")
    l2_buf_cnt = to_list(line1[4]) if len(line1) > 4 else None
    l1_buf_cnt = to_list(line1[5]) if len(line1) > 5 else None
    info = TileInfo(
        cube_dim=to_list(line1[0]),
        grid_dim=to_list(line1[1]),
        block_dim=to_list(line1[2]),
        tile_shape=to_list(line1[3]),
        l2_buf_cnt=l2_buf_cnt,
        l1_buf_cnt=l1_buf_cnt,
    )
    print(line1)
    return info
