# Nova Platform Bundle

This directory contains a self-contained copy of the nova-lite simulation pipeline
along with the minimal backend modules required to execute GEMM workloads without
referencing the original `bossa_nova` package. Key contents:

- `nova_lite/`: façade APIs for running simulations.
- `nova_platform/`: refactored backend modules (base models, executor, dataflow,
  cost services, perfetto protobuf helpers, etc.).
- `config/`: topology templates and runtime YAML configs expected by the pipeline.

To exercise the bundle in isolation, add this directory to `PYTHONPATH` and import
`nova_lite`:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'nova-platform')
from nova_lite import SimulationPipeline
pipeline = SimulationPipeline('config/libra_1DIE_3.2TB_24SIP_256OST.yaml')
print(pipeline)
PY
```

Artifacts produced by runs are written to `nova-platform/out/nova_lite` by default.

## Pytest CLI Support

The bundle ships with a lightweight pytest plugin so you can drive nova-lite
directly from the command line. Example:

```bash
PYTHONPATH=nova-platform pytest -p nova_platform.pytest_nova_lite \
    --run-nova-lite \
    --nova-lite-config config/libra_1DIE_3.2TB_24SIP_256OST.yaml \
    --nova-lite-shape 1,512,256,256 \
    --nova-lite-dtype fp16 \
    --nova-lite-bench-version 5
```

Override `--nova-lite-output` to choose a different artifact directory or
`--nova-lite-force-rerun` to ignore cached reports.

## Dependencies

Install the minimal runtime dependencies with:

```bash
pip install -r nova-platform/requirements.txt
```

GPU cache modelling additionally depends on `pycuda`; install it separately if
you need the CUDA-backed cache cost service.
