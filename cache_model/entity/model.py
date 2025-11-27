from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class No_MSHR(RuntimeError):
    """Raised when the simulated cache runs out of MSHR slots."""


@dataclass
class L1C_Config:
    CACHE_LINE_SIZE: int
    CACHE_WAYS: int
    CACHE_SIZE: int
    MEM_LATENCY: int
    NON_MEM_LATENCY: int
    NUM_MSHR: int
    NUM_OF_SIP: int
    CACHE_SIZE_PER_SIP: int


@dataclass
class LLC_Config:
    CACHE_LINE_SIZE: int
    CACHE_WAYS: int
    CACHE_SIZE: int
    MEM_LATENCY: int
    NON_MEM_LATENCY: int
    NUM_MSHR: int
    NUM_OF_PARTITIONS: int = 1
    NUM_OF_SLICES_PER_PARTITION: int = 1


@dataclass
class L3_Config:
    START_ADDR: int
    SIZE_PER_HBM: int
    NUM_OF_HBM: int


@dataclass
class Memory:
    LLC: LLC_Config
    L3: Optional[L3_Config] = None
    L1C: Optional[L1C_Config] = None


@dataclass
class HardwareConfig:
    MEMORY: Memory

    def __getattr__(self, item: str):
        """Expose LLC fields directly for the legacy SimpleStrategy helper."""
        llc = getattr(self.MEMORY, "LLC", None)
        if llc and hasattr(llc, item):
            return getattr(llc, item)
        raise AttributeError(item)


@dataclass
class ModelContext:
    hardware: HardwareConfig
    timestamp: int = 0


@dataclass
class Access:
    direction: int
    address: int
    line_addr: int
    width: int
    end_address: int
    thread: Any = None


@dataclass
class Request:
    direction: int
    address: int
    line_addr: int
    thread: Any = None


__all__ = [
    "Access",
    "HardwareConfig",
    "L1C_Config",
    "L3_Config",
    "LLC_Config",
    "Memory",
    "ModelContext",
    "No_MSHR",
    "Request",
]
