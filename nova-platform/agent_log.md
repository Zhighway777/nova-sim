## 记录规则
- 每次提交/修改后记录改动内容（涉及文件、功能、原因）。
- 如有运行验证（测试、脚本、命令），记录命令和结果（通过/失败及关键信息）。
- 记录每次变动的日期与时间。


## 变更日志
- [2025-12-02T17:09:13+08:00] 简化 TPU 配置 `config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml`，将架构收缩为 1x1，移除 XPU 相关带宽，仅保留 mc/sic_io/noc 最小必需字段；运行 `NOVA_SIM_FORCE_REAL=1 PYTHONPATH=.:./nova-platform pytest -v -p nova_platform.pytest_nova_lite --run-nova-lite --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml --nova-lite-shape 32,40,128,40 --nova-lite-dtype fp16 --nova-lite-bench-version 5 --log-file out/nova_lite/test.log --log-cli-level=INFO --log-file-level=INFO` 通过（TraceGenerator 因权限回退为线程模式）。


## 旧版本log记录
- 更新了 `nova_platform/simulator/case.py`，改进配置模板的查找范围，覆盖工作目录、仓库根目录和 `nova_lite/config`。
- 调整 `nova_lite/test_pipeline.py`，断言产物位于 `SimulationPipeline.output_root`。
- 通过 `pytest nova_lite/test_pipeline.py -q` 以及使用 libra 配置模板的 nova-lite pytest 入口完成修复验证。
- 重构 `nova_platform/benchmark/batch_gemm.py`，引入抽象的 `BatchGemmBase`（后端无关接口）和 XPU 专用的 `XpuBatchGemmBase`，将带宽/DSM 切分状态移出基类；确保 `impl` 存储 `_tiled_workloads`。
- 更新 XPU tiler 继承新的 XPU 基类（`batch_gemm_local.py`、`batch_gemm_shared.py`），功能无变化。
- 添加 GPU/TPU 占位 tiler（`batch_gemm_gpu.py`、`batch_gemm_tpu.py`），固定单 tile 工作负载和简易 FLOPs 成本，并提供 `tile_gpu_gemm_workload`/`tile_tpu_gemm_workload`。
- 对修改后的 benchmark 文件运行 `python -m py_compile` 做语法校验。
- 打通最小 TPU 后端路径：在 `executor/dataflow_gen.py` 增加 TPU GEMM 生成器和 action 映射；创建 TPU 配置克隆 `config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml`（arch=tpu）。
- 给 nova-lite 输出追加架构后缀：`SimulationPipeline._build_case_dir` 现在附加 `_arch`（如 `_tpu`），用轻量级架构检测避免 YAML `!include` 解析问题。
- 实现 TPU tiler/成本/action：`batch_gemm_tpu.py` 采用数组维度切分并考虑 SRAM/HBM 成本；`tpu_gemm_action.py`（TpuGemmAction）发出 ComputeStat/latency（内存统计简化以避免无效路径）；清理重复生成器定义。
- TPU 配置更新：新增 `compute.tpu`（ARRAY_M/N/K_TILE），将 `memory.l1.SIZE_PER_CORE` 提升到 16 MiB；移除 `bw.xpu` 块。给 `BossaNovaConfig` 增加 `arch` 字段以便后端选择遵循配置架构。
- 烟雾测试（TPU 路径，nova-lite）：执行 `NOVA_SIM_FORCE_REAL=1 PYTHONPATH=.:./nova-platform pytest -v -p nova_platform.pytest_nova_lite --run-nova-lite --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml --nova-lite-shape 32,40,128,40 --nova-lite-dtype fp16 --nova-lite-bench-version 5 --log-file out/nova_lite/test.log --log-cli-level=INFO --log-file-level=INFO`；pytest 通过，因权限限制 TraceGenerator 仍回退线程模式。输出目录：`out/nova_lite/gemm_v5_fp16_32-40-128-40_tpu/...`。
- [2025-12-02T17:09:13+08:00] 简化 TPU 配置 `config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml`，将架构收缩为 1x1，移除 XPU 相关带宽，仅保留 mc/sic_io/noc 最小必需字段；运行 `NOVA_SIM_FORCE_REAL=1 PYTHONPATH=.:./nova-platform pytest -v -p nova_platform.pytest_nova_lite --run-nova-lite --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml --nova-lite-shape 32,40,128,40 --nova-lite-dtype fp16 --nova-lite-bench-version 5 --log-file out/nova_lite/test.log --log-cli-level=INFO --log-file-level=INFO` 通过（TraceGenerator 因权限回退为线程模式）。
- [2025-12-02T19:04:19+08:00] 进一步精简 TPU 配置：移除无关带宽段，仅保留 sic_io/noc/mc、映射内存层次为 VREGS(l0)/VMEM(l1)/HBM(l3)；同命令重跑 pytest 通过（仍有 TraceGenerator 权限警告但不影响结果），产物路径同上。
- [2025-12-02T20:12:37+08:00] 为 TPU GEMM 添加三级搬运仿真：`batch_gemm_tpu.py` 补充 tile bytes 统计；`tpu_gemm_action.py` 生成 HBM→VMEM、VMEM→VREGS、VREGS→HBM 的 `DataflowActionMemoryStat` 及计算阶段；配置中补齐 l0/local/sic_io/noc/llc/mc 带宽、xpu/cdte 频域，避免路径缺失。运行同命令 pytest 通过（TraceGenerator 仍回退线程模式），可在 trace 中看到搬运片段，report 更新为 `total_latency: 14245ns`。
- [2025-12-02T20:41:54+08:00] 移除 TPU 中的 SIC/SIP 假设：`TPUGemmDataflowGenerator` 重写 dataflow 生成，仅按阵列数（默认单阵列）创建动作；`TpuGemmAction.get_kernel_idx` 兼容单元素 data；TPU 配置保持 VREGS/VMEM/HBM 三层。重跑同命令 pytest 通过（TraceGenerator 仍回退线程模式），trace/report 正常。
- [2025-12-02T20:50:xx+08:00] Trace 标签适配 TPU：`DataflowAction.get_trace_label` 默认使用动作类型名；`TpuGemmAction` 覆盖为 `sa` 前缀；executor 创建轨道时使用 trace_label，compute/memory 轨道随之更新。重跑同命令 pytest 通过（TraceGenerator 仍回退线程模式），SA 轨迹出现在 trace 中。
