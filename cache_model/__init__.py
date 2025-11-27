"""
Lightweight cache model primitives used by the nova-sim tests.

The original project depends on an internal ``cache_model`` package.  To keep
the tests runnable in this repository we provide a greatly simplified
implementation that offers the tiny subset of behaviours nova-lite relies on
when exercising cache cost services.
"""

__all__ = [
    "arch",
    "entity",
    "memory",
    "type_util",
]
