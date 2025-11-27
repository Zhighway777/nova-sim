## Change Log

- Updated `nova_platform/simulator/case.py` to improve config template discovery, covering working directory, repo root, and `nova_lite/config`.
- Adjusted `nova_lite/test_pipeline.py` to assert artifacts live under `SimulationPipeline.output_root`.
- Verified fixes via `pytest nova_lite/test_pipeline.py -q` and the nova-lite pytest entrypoint using the libra config template.
