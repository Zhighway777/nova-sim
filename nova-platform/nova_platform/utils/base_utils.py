from dataclasses import asdict
from collections.abc import Iterable
import sys


class BaseDataclass:
    def __str__(self):
        return str(asdict(self))


class hash_list(list):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Iterable):
            args = args[0]
        super().__init__(args)

    def __hash__(self):
        return hash(tuple(self))


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def check_gil_enabled():
    if not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled():
        return True
    else:
        return False
