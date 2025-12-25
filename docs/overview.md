# Nova-Sim Repository Overview

## Purpose

The `nova-sim` repository is a comprehensive simulation and performance modeling framework for the Nova Platform, a heterogeneous hardware accelerator architecture. It provides an end-to-end pipeline for simulating computational workloads (primarily GEMM operations), modeling memory hierarchies, calculating performance costs (compute, memory, power), and generating detailed performance traces for analysis.

The core purpose is to enable performance prediction and bottleneck analysis for AI/ML workloads on the Nova hardware architecture before physical silicon is available. This allows for architectural exploration, workload optimization, and hardware/software co-design.

## End-to-End Architecture

The repository follows a layered, modular architecture that transforms high-level operation specifications into detailed performance reports and visual traces.

### High-Level Data Flow

```mermaid
flowchart TD
    subgraph "Input Layer"
        User["User / Application"]
        Config["Hardware Config<br/>(YAML)"]
        Workload["Workload Spec<br/>(Shape, Type, Topology)"]
    end

    subgraph "Simulation Pipeline"
        NL[Nova Lite<br/>Pipeline Orchestrator]
        Sim["Simulator<br/>CaseInfo / FusionCaseInfo"]
        Executor[Executor<br/>BossaNovaExecutor]
        DG[Dataflow Generator<br/>Operation Specific]
        DF[Dataflow Core<br/>DAG Execution]
    end

    subgraph "Cost Modeling"
        CS[Cost Service<br/>Compute, Cache, Power]
        CacheModel[Cache Model<br/>L1/LLC/L3 Simulation]
        DataTransport[Data Transport<br/>BW & Latency]
    end

    subgraph "Output & Visualization"
        Visual[Data Visual<br/>PostProcessor]
        Trace[Perfetto Protobuf<br/>Trace Generator]
        Report["Performance Report<br/>(YAML, Metrics)"]
    end

    %% Data Flow
    User --> NL
    Config --> NL
    Workload --> NL
    
    NL --> Sim
    Sim --> Executor
    
    Executor --> DG
    DG --> DF
    DF --> Executor
    
    Executor --> CS
    CS --> CacheModel
    CS --> DataTransport
    
    CS --> Visual
    Visual --> Trace
    Visual --> Report
    
    %% Styling
    style NL fill:#e1f5ff
    style Sim fill:#fff2e1
    style Executor fill:#fff2e1
    style CS fill:#ffe0b2
    style CacheModel fill:#ffe0b2
    style Visual fill:#e8f5e9
    style Trace fill:#f3e5f5
    
```

### Detailed Component Architecture

```mermaid
graph TB
    subgraph "Frontend & Orchestration"
        NovaLite[nova_lite<br/>SimulationPipeline]
        Simulator[simulator<br/>CaseInfo / FusionCaseInfo]
        Config[config<br/>BossaNovaConfig]
        Utils[utils<br/>ConfigLoader, GCUData]
    end

    subgraph "Execution Engine"
        Executor[executor<br/>BossaNovaExecutor]
        DataflowGen[executor.dataflow_gen<br/>DataflowGenerator]
        DataflowCore[dataflow_core<br/>Dataflow]
        Sync[executor.synchronization<br/>Barrier/Event]
        ESL[executor.switch<br/>ESL Switch Manager]
    end

    subgraph "Action & Operation Layer"
        Actions[dataflow_actions<br/>XPUGemmAction, etc.]
        Benchmark[benchmark<br/>BatchGemmBase]
        BaseModels[base_models<br/>DType, DataflowActionType]
    end

    subgraph "Cost & Performance Modeling"
        CostService[cost_service<br/>ComputeCostService]
        CacheCost[cost_service.cache<br/>CacheCostService]
        PowerCost[cost_service.power<br/>PowerCostService]
        DataTransport[cost_service.compute<br/>DataTransportService]
        CacheModel[cache_model<br/>LLC/L1/L3 Simulation]
    end

    subgraph "Visualization & Output"
        DataVisual[data_visual<br/>PostProcessor]
        Perfetto[perfetto_protobuf<br/>TraceGenerator]
        OpTrace[optrace_benchmark<br/>Trace Parser]
    end

    %% Integration Flow
    NovaLite --> Simulator
    Simulator --> Config
    Simulator --> Executor
    Simulator --> Utils
    
    Executor --> DataflowGen
    Executor --> DataflowCore
    Executor --> Sync
    Executor --> ESL
    Executor --> CostService
    
    DataflowGen --> Actions
    DataflowGen --> Benchmark
    DataflowCore --> Actions
    
    Actions --> BaseModels
    
    CostService --> CacheCost
    CostService --> PowerCost
    CostService --> DataTransport
    CostService --> CacheModel
    
    Executor --> DataVisual
    DataVisual --> Perfetto
    
    Benchmark --> OpTrace

    %% Styling
    style NovaLite fill:#e1f5ff
    style Simulator fill:#fff2e1
    style Executor fill:#fff2e1
    style Actions fill:#ffe0b2
    style CostService fill:#ffe0b2
    style DataVisual fill:#e8f5e9
    style Perfetto fill:#f3e5f5

```

## Core Modules Documentation

### 1. Cache Model (`cache_model`)

**Purpose**: Provides a comprehensive simulation framework for modeling multi-level cache hierarchies (L1, LLC, L3) with configurable parameters, LRU replacement policies, and detailed statistics tracking.

**Key Components**:
- `AbstractGCU`: Top-level interface for cache simulation
- `AbstractMemoryManager`: Base interface for memory managers
- `LLCManager`, `L1CManager`, `L3Manager`: Cache level implementations
- `_LRUSet`: LRU replacement policy implementation
- `HardwareConfig`, `Request`, `Access`: Data models

**Architecture**:
```mermaid
classDiagram
    class AbstractGCU {
        +process(request, timestamp)
        +post_process(timestamp)
    }
    class AbstractMemoryManager {
        +process(request)
        +next_level: AbstractMemoryManager
    }
    class _BaseCacheManager {
        +_stat: Counter
        +_hist: Counter
    }
    class L1CManager {
        +_cache_selector
        +_sets
    }
    class LLCManager {
        +_addr_convert
        +_sets
    }
    class _LRUSet {
        +_lines: OrderedDict
        +access(tag) bool
    }
    
    AbstractGCU --> AbstractMemoryManager
    _BaseCacheManager --|> AbstractMemoryManager
    L1CManager --|> _BaseCacheManager
    LLCManager --|> _BaseCacheManager
    _BaseCacheManager --> _LRUSet
```

**Data Flow**:
```mermaid
flowchart TD
    A[External Request] --> B[AbstractGCU.process()]
    B --> C[L1CManager.process()]
    C --> D{L1 Hit?}
    D -->|Yes| E[Record Hit]
    D -->|No| F[LLCManager.process()]
    F --> G{LLC Hit?}
    G -->|Yes| H[Record LLC Hit]
    G -->|No| I[L3Manager.process()]
    I --> J[Record L3 Access]
    J --> K[Propagate back]
    H --> K
    E --> K
    K --> L[Update Statistics]
    L --> M[Post-process]
    M --> N[Generate Reports]
```

**Documentation**: See `cache_model` module docs for detailed information.

---

### 2. Nova Lite (`nova_lite`)

**Purpose**: Provides a streamlined, Python-only simulation pipeline for GEMM operations, abstracting away the complexity of the full Nova platform infrastructure.

**Key Components**:
- `SimulationPipeline`: Main orchestrator for running simulations
- `SimulationResult`: Encapsulates all simulation artifacts

**Architecture**:
```mermaid
classDiagram
    class SimulationPipeline {
        +run_gemm(shape, dtype, bench_version, quant_type, enable_cache, topo, force_rerun, tags)
        -_build_case_dir()
        -_collect_results()
    }
    class SimulationResult {
        +report: Dict
        +report_path: Path
        +trace_path: Path
        +output_dir: Path
    }
    SimulationPipeline --> SimulationResult
```

**Integration**:
```mermaid
flowchart TD
    NovaLite[SimulationPipeline] --> BaseModels[base_models]
    NovaLite --> Simulator[simulator/CaseInfo]
    NovaLite --> Config[config]
    
    Simulator --> Benchmark[benchmark]
    Simulator --> Executor[executor]
    Simulator --> CostService[cost_service]
```

**Documentation**: See `nova_lite` module docs for detailed information.

---

### 3. Base Models (`base_models`)

**Purpose**: Foundational data model layer defining core enumerations, data structures, and abstract base classes for computational operations, memory domains, and statistics.

**Key Components**:
- `DType`: Data type definitions (FP16, FP32, INT8, etc.)
- `AddrDomain`: Memory address space classification
- `DataflowActionType`: Operation categorization (XPU, DTE, ESL)
- `DataflowOpType`: Computational primitives (GEMM, ADD, SOFTMAX, etc.)
- `BaseActionStat`, `DataflowActionMemoryStat`, `DataflowActionComputeStat`: Statistical models
- `PostStat`, `EDCStat`: Post-simulation statistics
- `BaseFrame`: Temporal framework
- `BaseESLSwitch`: Hardware abstraction

**Architecture**:
```mermaid
classDiagram
    class DType {
        +FP16, FP32, INT8, etc.
        +get_bpe()
    }
    class AddrDomain {
        +L0, L1C, LOCAL, SHARED, LLC, L3, ESL
        +get_addr_domain(addr)
    }
    class DataflowActionMemoryStat {
        +src, dst: AddrDomain
        +cache_stat
        +memory_access_list
    }
    class DataflowActionComputeStat {
        +compute_1d_ops
        +compute_2d_ops
        +__iadd__()
    }
    class PostStat {
        +total_latency
        +core_util
        +l3_rw_bw_util
        +edc: EDCStat
    }
    
    BaseActionStat <|-- DataflowActionMemoryStat
    BaseActionStat <|-- DataflowActionComputeStat
    DataflowActionMemoryStat o-- DataflowActionMemoryAccess
    PostStat o-- EDCStat
```

**Documentation**: See `base_models` module docs for detailed information.

---

### 4. Benchmark (`benchmark`)

**Purpose**: Framework for benchmarking and tiling operations across different hardware backends (XPU, GPU, TPU) and memory architectures.

**Key Components**:
- `BatchGemmBase`: Abstract GEMM tiler
- `XpuBatchGemmBase`: XPU-specific base
- `BatchGemmLocal`, `BatchGemmShared`: XPU memory architectures
- `BatchGemmTpu`: TPU backend
- `BatchGemmGPU`: GPU backend
- `Workload`, `OpBase`: Workload representation
- `GridShape`: Tiling dimensions

**Architecture**:
```mermaid
graph TD
    A[Workload<br/>OpBase] --> B[BatchGemmBase]
    B --> C[XpuBatchGemmBase]
    B --> D[BatchGemmGPU]
    B --> E[BatchGemmTpu]
    C --> F[BatchGemmLocal]
    C --> G[BatchGemmShared]
    
    H[ECCL Primitives] --> I[SimpleProtoPrimitives]
    H --> J[EcclPrimitives]
```

**Documentation**: See `benchmark` module docs for detailed information.

---

### 5. Config (`config`)

**Purpose**: Central configuration management system defining comprehensive dataclass structures for hardware specifications, memory hierarchies, compute capabilities, and power characteristics.

**Key Components**:
- `BossaNovaConfig`: Top-level system configuration
- `InstNumConfig`: Hardware topology
- `FreqConfig`: Clock domains
- `ComputeConfig`: Processing capabilities
- `MemoryConfig`: Memory hierarchy (L0, L1, L2, L3, LLC)
- `BWConfig`: Bandwidth specifications
- `PowerLibConfig`: Power models
- `TOPO`: System topologies (STANDALONE, FULLMESH8, etc.)

**Architecture**:
```mermaid
graph TB
    BossaNovaConfig --> InstNumConfig
    BossaNovaConfig --> FreqConfig
    BossaNovaConfig --> ComputeConfig
    BossaNovaConfig --> MemoryConfig
    BossaNovaConfig --> BWConfig
    BossaNovaConfig --> PowerLibConfig
    
    MemoryConfig --> LLC_Config
    MemoryConfig --> L1C_Config
    MemoryConfig --> MemoryL3Config
    
    BWConfig --> BWFile
    BWFile --> BWEle
    
    PowerLibConfig --> PowerSIPLibConfig
    PowerLibConfig --> PowerSOCLibConfig
    PowerLibConfig --> PowerMemLibConfig
```

**Documentation**: See `config` module docs for detailed information.

---

### 6. Cost Service (`cost_service`)

**Purpose**: Comprehensive cost modeling for hardware accelerators, calculating computational costs, memory access costs, and power consumption for dataflow operations.

**Key Components**:
- `ComputeCostService`: Main orchestration engine
- `CacheCostService`: Cache hierarchy simulation
- `PowerCostService`: Power consumption modeling
- `DataTransportService`: Bandwidth and latency calculation
- `BossaNovaContext`: Global execution context
- `CostBook`: Per-action cost accumulation

**Architecture**:
```mermaid
graph TD
    A[DataflowAction] --> B[ComputeCostService]
    B --> C[CacheCostService]
    B --> D[PowerCostService]
    B --> E[DataTransportService]
    C --> F[Cache Model Architecture]
    D --> G[Power Model Domains]
    E --> H[BWResourceContext]
    
    F --> F1[Eltanin]
    F --> F2[Libra]
    
    G --> G1[SIP Domain]
    G --> G2[L1 Domain]
    G --> G3[SOC Domain]
    G --> G4[Mem Domain]
    
    B --> I[CostBook]
    I --> I1[Cache Stats]
    I --> I2[Power Stats]
    I --> I3[Core Stats]
    I --> I4[Latency]
```

**Documentation**: See `cost_service` module docs for detailed information.

---

### 7. Data Visual (`data_visual`)

**Purpose**: Post-processing simulation results and generating comprehensive reports and visualizations in YAML and Perfetto trace formats.

**Key Components**:
- `PostProcessor`: Single-GCU post-processing
- `FusionPostProcessor`: Multi-GCU result fusion
- `BossaNovaTraceProcessor`: Trace analysis with SQL queries
- `AbstractPostProcessor`: Base interface

**Architecture**:
```mermaid
graph TB
    AbstractPostProcessor --> PostProcessor
    AbstractPostProcessor --> FusionPostProcessor
    
    PostProcessor --> ReportYAML[YAML Report]
    PostProcessor --> PerfettoTrace[Perfetto Trace]
    FusionPostProcessor --> FusionReport[Fusion Report]
    
    BossaNovaTraceProcessor --> SQLQuery[SQL Queries]
    
    Input[BossaNovaContext, Dataflow] --> PostProcessor
    PostProcessor --> Output[Reports & Traces]
```

**Documentation**: See `data_visual` module docs for detailed information.

---

### 8. Dataflow Actions (`dataflow_actions`)

**Purpose**: Defines and implements computational operations for dataflow graphs with detailed performance modeling capabilities.

**Key Components**:
- `DiagDataflowAction`: Base class for all actions
- `XPUGemmAction`: Matrix multiplication
- `XPUAllReduceAction`, `XPUAllGatherAction`: Collective operations
- `XPUActivationAction`: Activation functions (ReLU, Sigmoid, GELU, SiLU)
- `XPULayernormAction`, `XPUSoftmaxAction`: Normalization operations
- `XPUSdpaAction`: Scaled Dot-Product Attention
- `CDTESliceAction`, `CDTEDesliceAction`: Data transformation engine

**Architecture**:
```mermaid
graph TB
    DDA[DiagDataflowAction] --> XPUA[XPUAction]
    DDA --> DTEA[DTEAction]
    
    XPUA --> XPUGemm
    XPUA --> XPUAllReduce
    XPUA --> XPUAllGather
    XPUA --> XPUActivation
    XPUA --> XPULayernorm
    XPUA --> XPUSoftmax
    XPUA --> XPUSdpa
    
    XPUActivation --> XPURelu
    XPUActivation --> XPUSigmoid
    XPUActivation --> XPUSilu
    XPUActivation --> XPUGelu
    
    DTEA --> DTESlice
    DTEA --> DTEDeslice
    DTEA --> DTEReshape
```

**Documentation**: See `dataflow_actions` module docs for detailed information.

---

### 9. Dataflow Core (`dataflow_core`)

**Purpose**: Central orchestration engine managing execution of computational workflows through graph-based dependency management.

**Key Components**:
- `Dataflow`: Main orchestrator class
- `DataflowAction`: Abstract base for operations
- `BARRIER`: Synchronization primitive
- `BossaNovaEvent`: Event-based synchronization

**Architecture**:
```mermaid
graph TD
    A[Action List] --> B[_build_dag]
    B --> C[NetworkX DiGraph]
    B --> D[Action Map]
    B --> E[Root Nodes]
    B --> F[Priority Queue]
    
    C --> G[execute_dataflow]
    D --> G
    E --> G
    F --> G
    
    G --> H[Generator Output]
    H --> I[Executor]
    I --> J[Cost Services]
    J --> K[Trace Generation]
```

**Documentation**: See `dataflow_core` module docs for detailed information.

---

### 10. Executor (`executor`)

**Purpose**: Core orchestration engine managing complete lifecycle of computational workloads from dataflow generation to execution and performance analysis.

**Key Components**:
- `BossaNovaExecutor`: Main orchestrator
- `DataflowGenerator`: Operation-specific generators
- `MemoryManager`: L3 memory allocation
- `ESLSwitchManager`: Inter-GCU communication
- `EventManager`, `BarrierManager`: Synchronization

**Architecture**:
```mermaid
graph TB
    NE[BossaNovaExecutor] --> DG[Dataflow Generator]
    DG --> DF[Dataflow]
    NE --> CS[Compute Service]
    NE --> PS[Power Service]
    NE --> CTS[Cache Service]
    CS --> TG[Trace Generator]
    NE --> EM[Event Manager]
    NE --> BM[Barrier Manager]
    EM --> EX[Execution Engine]
    BM --> EX
    ESLS[ESL Switch Manager] --> NE
```

**Documentation**: See `executor` module docs for detailed information.

---

### 11. Perfetto Protobuf (`perfetto_protobuf`)

**Purpose**: Generates Perfetto-compatible performance traces for timeline visualization and analysis.

**Key Components**:
- `TraceGenerator`: High-level trace generation
- `_BaseTraceGenerator`: Core protobuf generation
- `NormalTrack`, `CounterTrack`, `GroupTrack`: Track types
- `trace()`, `instant()`, `count()`: Profile APIs

**Architecture**:
```mermaid
graph TB
    App[Application] --> ProfileAPI[_profile.py]
    ProfileAPI --> TGen[_tgen.py]
    TGen --> Core[_core.py]
    Core --> Protobuf[perfetto_trace_pb2]
    Protobuf --> TraceFile[Trace File]
    
    TGen --> BaseTrack
    BaseTrack --> NormalTrack
    BaseTrack --> CounterTrack
    BaseTrack --> GroupTrack
    BaseTrack --> Group
```

**Documentation**: See `perfetto_protobuf` module docs for detailed information.

---

### 12. Simulator (`simulator`)

**Purpose**: Core orchestration layer managing execution of computational workloads across multiple GCU configurations with trace generation and fusion support.

**Key Components**:
- `CaseInfo`: Single case definition
- `FusionCaseInfo`: Composite case manager
- `GCUData`: Per-GCU state

**Architecture**:
```mermaid
graph TB
    CaseInfo --> do_sim
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
```

**Documentation**: See `simulator` module docs for detailed information.

---

### 13. Utils (`utils`)

**Purpose**: Foundational utilities including configuration loading, memory allocation, and data structures.

**Key Components**:
- `SingletonMeta`: Singleton pattern metaclass
- `BaseDataclass`: Enhanced dataclass
- `ConfigLoader`: YAML configuration loader
- `MemoryAllocator`: Virtual memory management
- `GCUData`: GCU state container

**Architecture**:
```mermaid
graph TD
    base_utils[base_utils.py] --> SingletonMeta
    base_utils --> BaseDataclass
    base_utils --> hash_list
    
    config_utils[config_utils.py] --> ConfigLoader
    config_utils --> Deserializable
    config_utils --> BaseEnum
    
    fusion_utils[fusion_utils.py] --> MemoryAllocator
    fusion_utils --> TensorInfo
    fusion_utils --> ScalarInfo
    fusion_utils --> OperationInfo
    
    gcu_utils[gcu_utils.py] --> GCUData
```

**Documentation**: See `utils` module docs for detailed information.

---

### 14. OpTrace Benchmark (`optrace_benchmark`)

**Purpose**: Lightweight parser for text-based operation trace formats used in benchmarking.

**Key Components**:
- `optrace`: Main parser
- `instruct`: Operation instruction
- `_Operand`, `_ScalarOperand`: Input representations
- `_Result`: Output representation
- `_Module`: Collection of instructions

**Architecture**:
```mermaid
classDiagram
    class optrace {
        -_module: _Module
        +getModule() _Module
    }
    class _Module {
        -_instructs: List[instruct]
    }
    class instruct {
        -_opname: str
        -_operands: List[object]
        -_results: List[_Result]
    }
    class _Operand {
        -_tensor_id: int
        -_dims: List[int]
        -_dtype: _DataType
    }
    class _ScalarOperand {
        -_value: int
    }
    class _Result {
    }
    
    optrace --> _Module
    _Module --> instruct
    instruct --> _Operand
    instruct --> _Result
    _Operand <|-- _Result
    _ScalarOperand <|-- _Operand
```

**Documentation**: See `optrace_benchmark` module docs for detailed information.

---

### 15. Perfetto Trace Processor (`perfetto_trace_processor`)

**Purpose**: Provides interface for querying Perfetto trace files using SQL.

**Key Components**:
- `TraceProcessor`: Main processor class
- `TraceProcessorConfig`: Configuration

**Documentation**: Minimal documentation provided.

## Key Features

1. **Multi-Level Cache Simulation**: Accurate modeling of L1, LLC, and L3 caches with LRU replacement
2. **Multi-Backend Support**: XPU, TPU, and GPU operation backends
3. **Distributed Computing**: Multi-GCU, multi-die, multi-cluster configurations
4. **Comprehensive Cost Modeling**: Compute, memory, power, and bandwidth costs
5. **Trace Generation**: Perfetto-compatible traces for timeline visualization
6. **Fusion Support**: Composite workloads from multiple operations
7. **Flexible Configuration**: Hardware specifications via YAML configs
8. **Performance Prediction**: End-to-end latency and utilization metrics

## Integration Flow

The repository provides a complete simulation pipeline:

1. **Configuration**: Define hardware specs via `BossaNovaConfig`
2. **Workload**: Specify operation (GEMM, etc.), shape, data type, topology
3. **Simulation**: Run via `NovaLite` or `Simulator`
4. **Dataflow Generation**: Convert to hardware-specific operations
5. **Execution**: Orchestrate through `Executor` with cost services
6. **Cost Calculation**: Compute, cache, power, and bandwidth modeling
7. **Post-Processing**: Generate reports and traces via `DataVisual`
8. **Analysis**: Visualize traces and analyze performance metrics

This architecture enables comprehensive performance analysis and optimization for Nova Platform workloads.