from typing import Dict, Tuple

from nova_platform.config import BossaNovaConfig
from nova_platform.benchmark.op_base import Workload
from nova_platform.benchmark.batch_gemm import BatchGemmBase, BatchGemmGridShape


class BatchGemmTPU(BatchGemmBase):
    """
    极简 TPU 占位 tiler：固定单一 tile，不做搜索，仅用于打通多后端接口。
    """

    def __init__(self, config: BossaNovaConfig, workload: Workload) -> None:
        super().__init__(config, workload)
        self.shape_list = []
        self._best_shape = None

    def split(self):
        shape = BatchGemmGridShape(
            grid_dims=[1, 1, 1, 1],
            block_dims=[1, 1, 1, 1],
            thread_dims=[1, 1, 1, 1],
            subthread_dims=[1, 1, 1, 1],
            lhs_buf_cnt_l2=1,
            rhs_buf_cnt_l2=1,
            res_buf_cnt_l2=1,
            lhs_buf_cnt_l1=1,
            rhs_buf_cnt_l1=1,
            res_buf_cnt_l1=1,
            calc_ceil_K_l2=self.in_k,
            calc_ceil_K_l1=self.in_k,
        )
        self.shape_list = [(0, shape)]

    def _calc_sol_cost(self, batch_gemm_shape: BatchGemmGridShape):
        assumed_throughput = 8e11  # ops/s，占位
        return self.calc_flops_total() / assumed_throughput

    def _gen_workloads(self, batch_gemm_shape: BatchGemmGridShape, first_sip_only=False):
        return {0: {0: [Workload(self.workload.inputs, self.workload.outputs, dtype=self.workload.dtype, attr=self.workload.attr)]}}

    def _sort_shape_candidates(self, shape_list):
        scored = []
        for idx, shape in shape_list:
            shape.sol_cost = self._calc_sol_cost(shape)
            scored.append((idx, shape))
        return scored

    def impl(self):
        if not self.shape_list:
            self.split()
        self.shape_list = self._sort_shape_candidates(self.shape_list)
        self._best_shape = self.get_best_shape()
        self._tiled_workloads = self._gen_workloads(self._best_shape)

    def get_best_shape(self) -> BatchGemmGridShape:
        return self.shape_list[0][1]


def tile_tpu_gemm_workload(config: BossaNovaConfig, chip_workload: Workload) -> Tuple[Dict, BatchGemmGridShape]:
    gemm = BatchGemmTPU(config, chip_workload)
    gemm.split()
    gemm.impl()
    return gemm.get_tiled_workloads(), gemm.get_best_shape()
