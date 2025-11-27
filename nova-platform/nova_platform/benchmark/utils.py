from typing import Generator, List, Tuple, Dict
from nova_platform.base_model import (
    DType,
    DataflowActionMemoryAccess,
    DataflowActionMemoryStat,
    DataflowActionType,
    AddrDomain,
    DataflowActionComputeStat,
)


def _iter_access_gen(mem_acc_list: List[DataflowActionMemoryAccess]):
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
                    last_size_right = counter - fetch_size
                    last_size_left = last_acc.size - last_size_right
                    batch.append(
                        DataflowActionMemoryAccess(base_addr=last_acc.base_addr, size=last_size_left, rw=last_acc.rw)
                    )

                    new_base = last_acc.base_addr + last_size_left
                    new_size = last_acc.size - last_size_left
                    counter = fetch_size
                    # history.append(batch)
                    fetch_size = yield batch
                    if new_size > 0:
                        batch = [DataflowActionMemoryAccess(base_addr=new_base, size=new_size, rw=last_acc.rw)]
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
