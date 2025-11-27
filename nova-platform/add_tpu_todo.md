# TODO: Enable TPU Architecture in nova-lite Platform

## 1. Define TPU Architecture Configs
- Extend `nova_platform/config.py` dataclasses (e.g., `InstNumConfig`, `FreqConfig`, `MemoryConfig`, `BWConfig`) so they can capture TPU-specific hierarchy (core counts, memory sizes, bandwidth tiers).
- Add TPU YAML templates under `config/` (and mirror in `nova_lite/config/` if needed) that populate the new fields. Verify `nova_platform.utils.config_utils.load_config` can deserialize them without extra tweaks.
- Update `CaseInfo._resolve_config_path` expectation if TPU configs live in a new directory structure.

## 2. Dataflow Generation
- Introduce TPU-aware generator(s) in `nova_platform/executor/dataflow_gen.py`:
  - Create `TpuGemmDataflowGenerator` (and other ops as needed) deriving from `DataflowGenerator`.
  - Encode TPU tile/grid/block semantics, memory allocation limits, and `TileInfo` layout.
  - Register new generators in `DATAFLOW_GENERATOR_MAPPING` with distinct `bench_op_type` keys (e.g., `tpu_gemm`).
- Ensure `SimulationPipeline`/CLI can select TPU op types by writing the proper `bench_op_type` into `dataflow_config`.

## 3. TPU Action Classes
- Add TPU-specific `DiagDataflowAction` subclasses (e.g., `TPUGemmAction`) if kernel layout or statistics differ.
- Update `action_map` in `nova_platform/executor/nova_platform_executor.py` so the new `bench_op_type` resolves to the TPU action class.
- Implement TPU tiling + latency plumbing (analogous to `tile_local_gemm_workload` / `dsm_local_gemm_kernel`) and wire them inside `get_memory_stat`.

## 4. Executor & Interconnect
- If TPU interconnect deviates from ESL, add a new `TOPO` entry and corresponding `BaseESLSwitch` implementation (`nova_platform/executor/nova_platform_switch.py`).
- Review `get_data_flow_gen_cmd`: supplement CLI args with TPU-only parameters or adjust existing ones to avoid unused XPU assumptions.
- Confirm `BossaNovaExecutor.generate_dataflow` understands the new `bench_op_type` and passes TPU-specific metadata (e.g., SRAM sizes, tensor addresses).

## 5. Cost Services & Reporting
- Verify cache/compute/power services can consume TPU config fields. Add new bandwidth resources or core stats if the pipeline exposes different counters.
- Extend `CoreCost` / `InstructionInfo` (or TPU equivalents) so post-processing captures TPU instruction/grid info.
- Make sure `PostProcessor` can handle the additional stats without breaking longest-path or utilization calculations.

## 6. CLI / Pipeline Integration
- Document TPU usage in README and pytest plugin help text (describe required `--nova-lite-config` or new `--nova-lite-arch` flag).
- Add regression tests (e.g., `pytest nova_lite/test_pipeline.py -k tpu`) that instantiate `SimulationPipeline` with TPU configs to guard against regressions.

## 7. Validation Pass
- Run nova-lite end-to-end with TPU config to confirm:
  - Dataflow generation produces valid DAGs / metadata.
  - Executors emit TPU-specific reports and perfetto traces.
  - Cached artifacts properly segregate TPU runs (consider tagging output directories with `tpu`).
