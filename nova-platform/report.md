实验目标：本实验旨在复现并量化 nova-lite 驱动的 FP16 GEMM（形状 32×40×128×40，bench 版本 5）在 Libra 单 DIE 拓扑下的延迟与带宽表现，梳理从命令行入口到数据流仿真的全过程，并给出可用于体系结构评估的定量指标。

实验设置：实验通过 ytest -p nova_platform.pytest_nova_lite --run-nova-lite --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST.yaml --nova-lite-shape 32,40,128,40 --nova-lite-dtype fp16 --nova-lite-bench-version 5 --nova-lite-tags nova-lite-cli --log-file out/nova_lite/test.log --log-cli-level=INFO --log-file-level=INFO 触发，输出目录为 out/nova_lite/gemm_v5_fp16_32-40-128-40，其中包含 report.yaml、metadata.yaml 以及 Perfetto 轨迹。配置文件描述的 Libra 拓扑拥有 4×6 个 SIP、核心频率 1.5 GHz、互联与存储频率 1.6 GHz、单 DIE 144 GiB L3、每 SIC 7 MiB L2 以及每核心 896 KiB L1，SIC↔NoC 与 MC↔L3 链路分别提供 4×128 B/cycle 与 16×128 B/cycle 的理论带宽。

实验流程：pytest 插件解析 CLI 参数后构造 SimulationPipeline，Pipeline 在 nova_lite/pipeline.py 中解析配置模板、建立按形状命名的输出目录并生成 CaseInfo。CaseInfo.do_sim 将运行元数据与配置写入磁盘，随后针对每个 GCU 创建 BossaNovaExecutor、初始化 Perfetto TraceGenerator，并注入 bench_gemm_op_version 与 bench_gemm_shape。Executor 调用 generate_dataflow 时将 bench 版本 5 映射为 gemm.shared，借助 nova_platform/executor/dataflow_gen.py 中的 Python 数据流生成器对输入输出张量分片并为每个 SIP 生成 XPU 行为序列。执行阶段，ComputeCostService 对每个行为记账，累计延迟、访存次数与带宽占用，同时记录最长路径信息；执行结束后，PostProcessor 计算关键路径与带宽利用率，整合各 Cost Service 的统计数据并输出 report.yaml。由于未设置 BOSSANOVA_PW，功耗相关字段保持为空。

实验结果：report.yaml 表明总延迟为 1.082 µs，纯计算 DAG 的 action end time 为 0.810 µs；最长 SIP 的三个阶段分别为 0.583 µs、0.192 µs 与 0.307 µs。ComputeCostService 统计的 tensor MAC 数为 6,553,600，对应 6.06 TMAC/s（约 12.1 TFLOP/s）的有效吞吐；workload balance 为 0.667，说明约 2/3 的 SIP 达到峰值 MAC。内存侧，L3 读写利用率为 0.285，SIC IO 读和写利用率分别为 0.185 与 0.029，双向平均 0.107；总体 L0 shared 访问量为 640 KiB 读和 100 KiB 写，L3 总流量 740 KiB，换算得到平均 L3 带宽约 700 GB/s，SIC IO 读/写带宽分别为 606 GB/s 与 94.7 GB/s。ESL 与 D2D 传输在该单 DIE 场景下未被触发。

结果分析：总延迟显著高于 action end time，说明 L3 排空阶段主导尾部延迟，且写链路利用率极低，提示更大的 M/N 可能改善计算与访存的重叠；workload balance 仅 0.667，表明当前 tile 划分未能充分填满 24 个 SIP，可考虑重新划分网格或调整批尺寸；core_util 为零是由于未启用 power service，如需核心占用或能耗指标须设置 BOSSANOVA_PW=1；STANDALONE 拓扑下 ESL/D2D 未被激活，若要评估多 DIE 互联需更换配置。

复现性：设置 PYTHONPATH=/home/zhw/nova-sim/nova-platform 后执行上述 ytest 命令即可重现；若需要重新计算并忽略缓存，可附加 --nova-lite-force-rerun。关键结果位于 out/nova_lite/gemm_v5_fp16_32-40-128-40/gcu00/report.yaml，Perfetto 轨迹可在同级目录验证时间线。

进一步工作：建议在更大形状上进行 sweep 以观察 SIP 利用率随负载变化的趋势；启用 BOSSANOVA_PW 以收集能耗与核心利用信息；比较 bench 版本 5（shared）与版本 6（local）在 Libra 上的带宽压力和计算效率，以形成对不同微结构的全面评估。

数据流生成算法（可直接用于论文描述）：
步骤 1：根据 bench 版本确定算子映射，定义函数 f(v) = gemm.local（v∈{4,6}）、gemm.shared（v=5）、gemm（其他），本实验取 f(5)=gemm.shared，并记录 bench_basic_data_type = dtype。
步骤 2：计算结构参数，令 sic_num = NUM_OF_CLUSTER × NUM_OF_DIE，sip_num = NUM_OF_CORE_PER_CLUSTER，动作总数 |A| = sic_num × sip_num。
步骤 3：构造张量：对输入 A∈ℝ^{B×M×K}、B∈ℝ^{B×K×N}、可选量化表以及输出 C∈ℝ^{B×M×N}，按字节数 bpe 计算 size = ∏_{d∈dims} d × bpe，内存地址遵循 addr = base + 128 ⌈size/128⌉。
步骤 4：遍历 (sic_idx, sip_idx)，令 engine_id = sip_idx + sic_idx × sip_num，生成动作 a_{sic_idx,sip_idx} = (action_type=XPU, code=op_type_dtype, die_id=0, engine_id, inputs, outputs, tile_info) 并赋予唯一 action_id。
步骤 5：设置 tile_info，使 cube_dim = grid_dim = [B,M,N]、block_dim = tile_shape = [1,1,1]，为后续调度提供统一坐标。
步骤 6：返回 DiagDataflow = (A, inputs, outputs, tile_info)，供 ComputeCostService 在执行阶段计算延迟和带宽。

成本模型计算算法：
步骤 1（输入）：给定数据流 DiagDataflow、架构配置 Config、初始上下文 Context（含初始参考时间 ref₀、带宽/功耗状态、TraceGenerator）。
步骤 2（依赖解析）：对每个动作 a_i，计算其父节点集合 P_i，通过 DAG 拓扑关系确保动作按依赖顺序进入执行队列。
步骤 3（开始时间）：定义 start(a_i) = max_{a_p ∈ P_i} (finish(a_p))，若 P_i 为空则 start(a_i) = ref₀。实际实现通过 compute_svc.get_ref 递归求解并缓存。
步骤 4（ComputeCostService 调用）：对于动作 a_i，将 (a_i, Context, start(a_i)) 送入 ComputeCostService.process，得到 cost_book(a_i)，其中包含 latency_i、memory_access_list、tensor_macs_i、ld/st 统计等。
步骤 5（结束时间与追踪）：设 finish(a_i) = start(a_i) + latency_i，并将该区间写入 TraceGenerator；同时更新 Context.cost_dict[a_i] = cost_book(a_i) 以便 PostProcessor 后续读取。
步骤 6（带宽记账）：根据 cost_book(a_i).memory_access_list，将每次访存映射到对应的带宽资源 bw_resource，调用 bw_resource.timeline.add_frame(start, finish, allocated_bw)。L0/L1/L3/SIC/MC 等级别的利用率即为 Σ(帧持续时间 × 分配带宽)/ (total_latency × 理论峰值)。
步骤 7（关键路径与总延迟）：执行完全部动作后，PostProcessor 通过遍历 DAG 根节点，递归求解 f(a_i) = latency_i + max_{a_c ∈ child(i)} f(a_c)，最大值即 action_end_time；L3 时间线的最后一个 frame end 减去 ref₀ 得到 memory_end_time，total_latency = max(action_end_time, memory_end_time)。
步骤 8（吞吐量与其他指标）：吞吐量 = Σ tensor_macs / total_latency；workload balance = (max per-SIP MAC) / (Σ tensor_macs / #SIP)；SIC/L3 利用率如步骤 6；若启用功耗服务，还需对电流帧做直方图聚合得到 EDC 报告。
步骤 9（输出）：返回 report = {total_latency, action_end_time, core_util, 各级带宽利用率, service_report_dict, longest_sip_stat, ...}，写入 report.yaml 供后续分析。
