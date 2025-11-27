from __future__ import annotations

from typing import Iterable, Iterator, Sequence


class TraceProcessorConfig:
    def __init__(self, **kwargs):
        self.options = kwargs


class TraceProcessor:
    """
    Very small stub that mimics the public API used by the tests.

    The real perfetto TraceProcessor streams SQL queries from trace files.
    For the purposes of the unit tests we only need query() to be iterable, so
    the stub simply returns an empty iterator.
    """

    def __init__(self, trace: str | None = None, config: TraceProcessorConfig | None = None):
        self.trace = trace
        self.config = config

    def query(self, sql: str) -> Iterator[dict]:
        return iter(())

    def close(self):
        pass


__all__ = ["TraceProcessor", "TraceProcessorConfig"]
