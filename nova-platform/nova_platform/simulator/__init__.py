"""
Lightweight simulation interfaces for the Python-first workflow.

This package exposes high-level case abstractions that were previously
embedded in the test suite, making them reusable by downstream projects.
"""

from .case import CaseInfo, FusionCaseInfo

__all__ = ["CaseInfo", "FusionCaseInfo"]

