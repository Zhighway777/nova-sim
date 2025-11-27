## Change Log

- Updated `nova_platform/simulator/case.py` to improve config template discovery, covering working directory, repo root, and `nova_lite/config`.
- Adjusted `nova_lite/test_pipeline.py` to assert artifacts live under `SimulationPipeline.output_root`.
- Verified fixes via `pytest nova_lite/test_pipeline.py -q` and the nova-lite pytest entrypoint using the libra config template.
- Refactored `nova_platform/benchmark/batch_gemm.py` to introduce abstract `BatchGemmBase` (backend-agnostic interfaces) and XPU-specific `XpuBatchGemmBase`, moving bandwidth/DSM tiling state out of the base; ensured `impl` stores `_tiled_workloads`.
- Updated XPU tilers to inherit the new XPU base (`batch_gemm_local.py`, `batch_gemm_shared.py`) without functional changes.
- Added GPU/TPU placeholder tilers (`batch_gemm_gpu.py`, `batch_gemm_tpu.py`) with fixed single-tile workload generation and simple FLOPs-based cost, plus `tile_gpu_gemm_workload`/`tile_tpu_gemm_workload` helpers.
- Ran `python -m py_compile` on the modified benchmark files to sanity-check syntax.
- Wired minimal TPU backend path: added `TPUGemmAction` placeholder, `TPUGemmDataflowGenerator`, and mapped TPU backend in `executor/dataflow_gen.py`; created TPU config clone `config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml` (arch=tpu).
- Added arch suffix to nova-lite outputs: `SimulationPipeline._build_case_dir` now appends `_arch` (e.g., `_tpu`), using a lightweight arch detector to avoid YAML `!include` parsing issues.
- Smoke test (TPU path, nova-lite): `NOVA_SIM_FORCE_REAL=1 PYTHONPATH=.:./nova-platform pytest -v -p nova_platform.pytest_nova_lite --run-nova-lite --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST_tpu.yaml --nova-lite-shape 32,40,128,40 --nova-lite-dtype fp16 --nova-lite-bench-version 5 --log-file out/nova_lite/test.log --log-cli-level=INFO --log-file-level=INFO`; pytest passed, with multiprocessing listener PermissionError fallback to thread-based TraceGenerator. Outputs now under `out/nova_lite/gemm_v5_fp16_32-40-128-40_tpu/...`.
