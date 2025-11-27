from dataclasses import dataclass


@dataclass
class DistanceCount:
    distance: int
    count: int


__all__ = ["DistanceCount"]
