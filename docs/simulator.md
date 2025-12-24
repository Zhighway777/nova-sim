# Simulator Module

## Overview

The simulator module is the core orchestration layer of the Nova Platform that manages the execution of computational workloads across multiple GCU (Graphics Compute Unit) configurations. It provides a framework for simulating, executing, and analyzing performance characteristics of various operations in a distributed computing environment.

The module handles:
- **Case Management**: Defines and manages individual simulation cases with configurable parameters
- **Multi-GCU Coordination**: Orchestrates parallel execution across multiple GCUs with synchronization
- **Trace Generation**: Creates performance traces for detailed analysis
- **Fusion Support**: Enables fusion of multiple cases for composite workload simulation
- **Configuration Management**: Resolves and validates hardware and simulation configurations

## Architecture

### High-Level System Overview

```mermaid
graph TB
    subgraph "Simulator Module"
        CaseInfo[CaseInfo<br/>Single Case Definition]
        FusionCaseInfo[FusionCaseInfo<br/>Composite Case Manager]
        
        subgraph "Execution Flow"
            do_sim[do_sim<br/>Entry Point]
            do_sim_inner[do_sim_inner<br/>Core Simulation]
            _execute[_execute<br/>Sequential]
            _p_execute[_p_execute<br/>Parallel]
        end
        
        subgraph "GCU Management"
            gcu_tgen_init[gcu_tgen_init<br/>Trace Initialization]
            gcu_tgen_exit[gcu_tgen_exit<br/>Trace Finalization]
            GCUData[GCUData<br/>Per-GCU State]
        end
        
        subgraph "Artifact Generation"
            _write_stub_artifacts[_write_stub_artifacts<br/>Fallback Artifacts]
            metadata[metadata.yaml]
            config[config.yaml]
            report[report.yaml]
            trace[trace.perfetto-trace]
        end
    end
    
    subgraph "External Dependencies"
        BossaNovaExecutor[BossaNovaExecutor<br/>executor module]
        ESLSwitchManager[ESLSwitchManager<br/>executor module]
        FusionPostProcessor[FusionPostProcessor<br/>data_visual module]
        TraceGenerator[TraceGenerator<br/>perfetto_protobuf module]
        ConfigLoader[ConfigLoader<br/>utils module]
        GCUData[GCUData<br/>utils module]
    end
    
    CaseInfo --> do_sim
    FusionCaseInfo --> do_sim
    do_sim --> do_sim_inner
    do_sim_inner --> gcu_tgen_init
    do_sim_inner --> _write_stub_artifacts
    do_sim_inner --> _execute
    do_sim_inner --> _p_execute
    _execute --> BossaNovaExecutor
    _p_execute --> BossaNovaExecutor
    BossaNovaExecutor --> ESLSwitchManager
    BossaNovaExecutor --> FusionPostProcessor
    BossaNovaExecutor --> TraceGenerator
    gcu_tgen_init --> TraceGenerator
    gcu_tgen_exit --> TraceGenerator
    
    style CaseInfo fill:#e1f5ff
    style FusionCaseInfo fill:#e1f5ff
    style BossaNovaExecutor fill:#fff2e1
    style ESLSwitchManager fill:#fff2e1
    style TraceGenerator fill:#fff2e1
```

### Component Relationships

```mermaid
classDiagram
    class CaseInfo {
        +str optype
        +List[int] shape
        +float expected_res
        +Callable fun
        +str model_name
        +List[str] tag
        +DType dtype
        +str config
        +bool enable_cache
        +str outdir
        +dict dataflow_config
        +int mu
        +List input_addr
        +List output_addr
        +float start_ref
        +float end_ref
        +float kernel_launch_latency
        +float _sum
        +TOPO topo
        +Dict[int, GCUData] gcu_data
        +bool is_last_case
        +do_sim()
        +do_sim_inner()
        +gcu_tgen_init()
        +gcu_tgen_exit()
        +_write_stub_artifacts()
        +_resolve_config_path()
        +__repr__()
        +__post_init__()
    }
    
    class FusionCaseInfo {
        +List[CaseInfo] cases
        +do_sim()
        +_prepare_case()
        +_update_fusion_state()
        +init_gcu_context()
        +add_case()
        +get_k_launch_latency()
    }
    
    class GCUData {
        +TraceGenerator tgen
        +float start_ref
        +float end_ref
        +float last_ref
        +CacheService cache_svc
        +PostStat post_stat
    }
    
    CaseInfo <|-- FusionCaseInfo : extends
    CaseInfo o-- GCUData : manages
    FusionCaseInfo o-- CaseInfo : contains
```

## Core Components

### CaseInfo

The `CaseInfo` class represents a single simulation case with all necessary parameters for execution.

**Key Responsibilities:**
- Defines operation type, shape, and configuration
- Manages GCU-specific data and trace generators
- Orchestrates simulation execution across GCUs
- Generates metadata and configuration artifacts

**Execution Flow:**
```mermaid
flowchart TD
    Start[do_sim] --> InitTrace[gcu_tgen_init]
    InitTrace --> CheckForce[Force Rerun?]
    CheckForce -->|No| CheckExists[Report Exists?]
    CheckForce -->|Yes| Continue[Continue]
    CheckExists -->|Yes| Skip[Skip Simulation]
    CheckExists -->|No| Continue
    Continue --> CheckBinary[Dataflow Binary Exists?]
    CheckBinary -->|No| WriteStub[_write_stub_artifacts]
    CheckBinary -->|Yes| Prepare[Prepare Execution]
    Prepare --> ResolveConfig[_resolve_config_path]
    ResolveConfig --> WriteMetadata[Write metadata.yaml]
    WriteMetadata --> WriteConfig[Write config.yaml]
    WriteConfig --> CheckGIL[Check GIL Enabled?]
    CheckGIL -->|Yes| Sequential[_execute]
    CheckGIL -->|No| Parallel[_p_execute]
    Sequential --> GenerateReport[Generate Report]
    Parallel --> GenerateReport
    GenerateReport --> ExitTrace[gcu_tgen_exit]
    ExitTrace --> End[End]
    
    style Skip fill:#d4edda
    style WriteStub fill:#f8d7da
```

**Configuration Resolution:**
```mermaid
flowchart TD
    InputConfig[Input Config Path] --> IsAbsolute{Is Absolute?}
    IsAbsolute -->|Yes| CheckExists1[Check Exists]
    IsAbsolute -->|No| BuildCandidates[Build Candidates]
    
    BuildCandidates --> AddCwd[Add CWD Path]
    BuildCandidates --> AddRepo[Add Repo Root Path]
    BuildCandidates --> AddNovaLite[Add nova_lite Path]
    
    AddCwd --> Search1[Search in CWD]
    AddRepo --> Search2[Search in Repo]
    AddNovaLite --> Search3[Search in nova_lite]
    
    Search1 --> Found{Found?}
    Search2 --> Found
    Search3 --> Found
    
    Found -->|Yes| ReturnPath[Return Path]
    Found -->|No| Fallback[Use Fallback Config]
    
    CheckExists1 -->|Yes| ReturnPath
    CheckExists1 -->|No| Fallback
    
    style ReturnPath fill:#d4edda
    style Fallback fill:#fff3cd
```

### FusionCaseInfo

The `FusionCaseInfo` class extends `CaseInfo` to support composite workloads consisting of multiple sequential cases.

**Key Features:**
- Manages a collection of `CaseInfo` objects
- Maintains shared GCU state across cases
- Performs post-processing fusion
- Tracks cumulative timing and performance metrics

**Fusion Flow:**
```mermaid
flowchart TD
    Start[do_sim] --> InitFusion[Initialize Fusion PostProcessor]
    InitFusion --> Loop[For Each Case]
    Loop --> PrepareCase[_prepare_case]
    PrepareCase --> UpdateStartRef[Update Start References]
    UpdateStartRef --> ExecCase[caseinfo.do_sim]
    ExecCase --> FusionStats[Fusion Post Stats]
    FusionStats --> UpdateState[_update_fusion_state]
    UpdateState --> UpdateLastRef[Update Last References]
    UpdateLastRef --> Loop
    Loop -->|All Cases| GenerateFusionReport[Generate Fusion Report]
    GenerateFusionReport --> WriteRuntimeLog[Write Runtime Log]
    WriteRuntimeLog --> End[End]
    
    style PrepareCase fill:#e1f5ff
    style ExecCase fill:#fff2e1
    style UpdateState fill:#e1f5ff
```

## Execution Modes

### Sequential Execution (`_execute`)

Used when Python GIL (Global Interpreter Lock) is enabled.

**Process:**
1. Creates a priority queue for task scheduling
2. Executes GCU tasks in priority order
3. Manages inter-GCU dependencies via reference timestamps
4. Collects reports from all GCUs

**Priority Queue Logic:**
- Tasks are prioritized by execution order
- Reference timestamps ensure proper synchronization
- Wait states trigger priority adjustments

### Parallel Execution (`_p_execute`)

Used when GIL is disabled, enabling true parallelism.

**Process:**
1. Creates a thread barrier for synchronization
2. Spawns thread pool for concurrent GCU execution
3. Each thread runs a GCU executor independently
4. Barrier ensures all GCUs complete before finalization
5. Collects reports and updates shared state

**Advantages:**
- Faster execution for multi-GCU configurations
- Better resource utilization
- Reduced total simulation time

## Integration Points

### External Module Dependencies

The simulator module integrates with several key components:

1. **executor module** (`nova-platform.nova_platform.executor`)
   - `BossaNovaExecutor`: Core execution engine for each GCU
   - `ESLSwitchManager`: Manages ESL (Electronic System Level) switches for inter-GCU communication
   - `BarrierManager`: Handles synchronization primitives

2. **data_visual module** (`nova-platform.nova_platform.data_visual`)
   - `FusionPostProcessor`: Aggregates and fuses performance statistics from multiple cases

3. **perfetto_protobuf module** (`nova-platform.nova_platform.perfetto_protobuf`)
   - `TraceGenerator`: Creates Perfetto-compatible performance traces

4. **config module** (`nova-platform.nova_platform.config`)
   - `BossaNovaConfig`: Hardware and simulation configuration
   - `TOPO`: Topology enumeration (STANDALONE, 2DIE, etc.)

5. **utils module** (`nova-platform.nova_platform.utils`)
   - `GCUData`: Per-GCU state container
   - `ConfigLoader`: Configuration file parsing
   - `BaseEnum`: Enumerations support

### Data Flow Between Components

```mermaid
sequenceDiagram
    participant CaseInfo
    participant ConfigLoader
    participant ESLSwitchManager
    participant BossaNovaExecutor
    participant TraceGenerator
    participant FusionPostProcessor
    
    CaseInfo->>ConfigLoader: Load BossaNovaConfig
    ConfigLoader-->>CaseInfo: Configuration Object
    
    CaseInfo->>ESLSwitchManager: Build Switch (per topo)
    ESLSwitchManager-->>CaseInfo: ESL Switch Object
    
    loop For Each GCU
        CaseInfo->>TraceGenerator: Initialize (gcu_tgen_init)
        TraceGenerator-->>CaseInfo: Trace Generator
        
        CaseInfo->>BossaNovaExecutor: Create with Config, TGen, Switch
        BossaNovaExecutor-->>CaseInfo: Executor
        
        CaseInfo->>ESLSwitchManager: Add GCU to Switch
        ESLSwitchManager-->>CaseInfo: Acknowledged
        
        CaseInfo->>BossaNovaExecutor: Execute (do_sim_inner)
        BossaNovaExecutor->>TraceGenerator: Generate Traces
        BossaNovaExecutor->>FusionPostProcessor: Post Process
        BossaNovaExecutor-->>CaseInfo: Report
        
        CaseInfo->>TraceGenerator: Finalize (gcu_tgen_exit)
    end
    
    CaseInfo->>FusionPostProcessor: Fusion Stats (if FusionCaseInfo)
    FusionPostProcessor-->>CaseInfo: Fused Report
```

## Configuration Management

### Configuration Resolution Strategy

The simulator uses a multi-tier configuration resolution strategy:

1. **Explicit Path**: If provided path is absolute, use directly
2. **Relative Search**: Search in common locations:
   - Current working directory
   - Repository root
   - Repository/config directory
   - nova_lite directory
3. **Fallback**: Use default configuration if not found

### Supported Topologies

```python
class TOPO(Enum):
    STANDALONE = (1, 1)   # 1 GCU, 1 device
    TWODIE = (2, 2)       # 2 GCUs, 2 devices
    # Additional topologies...
```

## Artifact Generation

### Output Structure

```
<outdir>/
├── metadata.yaml          # Case metadata
├── config.yaml            # Hardware configuration
├── gcu00/
│   ├── metadata.yaml      # GCU-specific metadata
│   ├── config.yaml        # GCU-specific config
│   ├── report.yaml        # Performance report
│   └── trace.perfetto-trace  # Performance trace
├── gcu01/
│   └── ...
└── runtime_log.txt        # Fusion runtime log (if fusion)
```

### Metadata Format

```yaml
optype: "gemm"
shape: [1024, 1024, 1024]
dtype: "FP16"
config: "libra_1DIE_3.2TB_24SIP_256OST.yaml"
enable_cache: false
topo: "STANDALONE"
# ... additional fields
```

### Report Format

```yaml
status: success
total_latency: 1.234e-3
# ... additional metrics
```

## Usage Patterns

### Single Case Simulation

```python
from nova_platform.simulator.case import CaseInfo
from nova_platform.base_model import DType
from nova_platform.config import TOPO

case = CaseInfo(
    optype="gemm",
    shape=[1024, 1024, 1024],
    dtype=DType.FP16,
    config="libra_1DIE.yaml",
    topo=TOPO.STANDALONE,
    outdir="./output/single_case"
)

case.do_sim()
```

### Fusion Case Simulation

```python
from nova_platform.simulator.case import CaseInfo, FusionCaseInfo

# Create multiple cases
case1 = CaseInfo(optype="gemm", shape=[512, 512, 512])
case2 = CaseInfo(optype="add", shape=[512, 512])
case3 = CaseInfo(optype="softmax", shape=[512, 512])

# Create fusion case
fusion = FusionCaseInfo(
    optype="layer_norm",
    shape=[512, 512],
    cases=[case1, case2, case3]
)

fusion.do_sim()
```

### Parallel Execution

```python
import os
# Disable GIL for parallel execution
os.environ["PYTHON_GIL"] = "0"

case = CaseInfo(
    optype="allreduce",
    shape=[1024],
    topo=TOPO.TWODIE,
    outdir="./output/parallel"
)

case.do_sim()
```

## Error Handling

### Common Failure Scenarios

1. **Missing Dataflow Binary**
   - Detection: Binary not found at `build/dataflow_gen`
   - Fallback: Generates stub artifacts with zero latency
   - Environment Override: `NOVA_SIM_FORCE_REAL=1` to force real execution

2. **Config Not Found**
   - Resolution: Multi-location search with fallback
   - Warning: Logged with fallback path

3. **Simulation Failure**
   - Logging: Full traceback captured
   - Exception: Propagated to caller
   - Partial Results: May be available in output directory

### Stub Artifacts

When dataflow generator is unavailable:

```yaml
# report.yaml
status: skipped
total_latency: 0
```

This allows downstream processing to continue without failure.

## Performance Considerations

### GIL Impact

- **GIL Enabled**: Sequential execution via priority queue
- **GIL Disabled**: Parallel execution via thread pool
- **Recommendation**: Disable GIL for multi-GCU configurations

### Caching

The `enable_cache` flag controls whether cache simulation results are reused:
- `True`: Cache service maintains state across cases
- `False`: Fresh simulation for each case

### Trace Generation Overhead

Trace generation adds significant overhead:
- Disable for batch processing if traces not needed
- Use `gcu_tgen_init()` and `gcu_tgen_exit()` strategically

## Best Practices

1. **Configuration Management**
   - Store configs in `config/` directory
   - Use relative paths for portability
   - Validate configs before simulation

2. **Output Organization**
   - Use descriptive `outdir` paths
   - Include topology and operation in path
   - Clean up old results before batch runs

3. **Parallel Execution**
   - Verify GIL is disabled
   - Ensure sufficient thread resources
   - Monitor for thread contention

4. **Fusion Cases**
   - Order cases logically (compute → communication)
   - Monitor cumulative timing
   - Use fusion for composite workloads

5. **Error Recovery**
   - Check for existing reports before rerunning
   - Use `force_rerun=True` for clean runs
   - Inspect stub artifacts for missing binaries

## Testing Considerations

### Unit Testing

- Mock `BossaNovaExecutor` to isolate simulator logic
- Test configuration resolution paths
- Verify metadata generation

### Integration Testing

- Test with actual dataflow generator binary
- Validate multi-GCU synchronization
- Verify trace generation and fusion

### Performance Testing

- Measure overhead of trace generation
- Compare sequential vs parallel execution
- Validate cache service impact

## Future Enhancements

### Planned Improvements

1. **Dynamic Topology**: Support for runtime topology changes
2. **Enhanced Fusion**: More sophisticated case fusion strategies
3. **Progressive Tracing**: Conditional trace generation based on thresholds
4. **Resource Monitoring**: Real-time resource utilization tracking
5. **Checkpoint/Resume**: Ability to pause and resume long simulations

### Extensibility Points

- Custom `CaseInfo` subclasses for domain-specific cases
- Pluggable post-processors for custom analysis
- Alternative execution strategies via strategy pattern
- Custom ESL switch implementations

## Related Documentation

- [Executor Module](executor.md) - Execution engine details
- [Cost Service Module](cost_service.md) - Performance cost models
- [Data Visual Module](data_visual.md) - Visualization and post-processing
- [Config Module](config.md) - Configuration schemas
- [Nova Lite Module](nova_lite.md) - High-level simulation pipeline

## References

- Perfetto Trace Format: https://perfetto.dev/docs/
- BossaNova Architecture: [Internal Architecture Document]
- GCU Topology Specifications: [Hardware Specification]
