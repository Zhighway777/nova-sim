import os
import yaml
from dataclasses import _MISSING_TYPE, Field
from enum import Enum
from typing import Any, Dict, List, TypeVar, _GenericAlias

import logging
logger = logging.getLogger(__name__)


class ConfigLoader(yaml.SafeLoader):
    # https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(ConfigLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, "r") as f:
            ret = yaml.load(f, ConfigLoader)
        assert ret is not None, "included file is empty? file: %s" % filename
        return ret

    def eval(self, node):
        expr = self.construct_scalar(node)
        return eval(expr)


ConfigLoader.add_constructor("!include", ConfigLoader.include)
ConfigLoader.add_constructor("!eval", ConfigLoader.eval)


class Deserializable:
    def deserialize(o: any):
        raise NotImplementedError()


T = TypeVar("T")


def dict_to_dataclass(d: Dict[str, Any], cls: T, custom_loader_map={}, *, restrict_mode=True) -> T:
    if cls in custom_loader_map:
        cls = custom_loader_map[cls](d)

    if isinstance(cls, type) and issubclass(cls, Deserializable):
        return cls.deserialize(d)

    if not hasattr(cls, "__dataclass_fields__"):
        if isinstance(cls, _GenericAlias) and cls.__origin__ is list:
            cls: List
            inner_cls = cls.__args__[0]
            return [dict_to_dataclass(x, inner_cls, custom_loader_map, restrict_mode=restrict_mode) for x in d]
        elif isinstance(cls, _GenericAlias) and cls.__origin__ is tuple:
            return tuple([dict_to_dataclass(x,  cls.__args__[i], custom_loader_map, restrict_mode=restrict_mode) for i, x in enumerate(d)])
        elif isinstance(cls, _GenericAlias) and cls.__origin__ is dict:
            key_type = cls.__args__[0]
            val_type = cls.__args__[1]
            res = dict({
                key_type(k): dict_to_dataclass(v, val_type, custom_loader_map, restrict_mode=restrict_mode)
                for k, v in d.items()
            })
            return res
        return cls(d)

    fields = cls.__dataclass_fields__
    kwargs = {}
    for field_name, field_type in fields.items():
        # TODO: need review
        if isinstance(field_type, Field):
            field_value = d.get(field_name)
            if field_value is not None:
                if field_type.type is None or type(field_value) == field_type.type:
                    kwargs[field_name] = field_value
                else:
                    kwargs[field_name] = dict_to_dataclass(
                        field_value, field_type.type, custom_loader_map, restrict_mode=restrict_mode)
            else:
                if not isinstance(field_type.default_factory, _MISSING_TYPE):
                    kwargs[field_name] = field_type.default_factory()
                elif not isinstance(field_type.default, _MISSING_TYPE):
                    kwargs[field_name] = field_type.default
                else:
                    if restrict_mode:
                        raise Exception(
                            f"required {field_name} is not provided")
                    else:
                        kwargs[field_name] = None
                        # logger.warn(f"required {field_name} is not provided")

        else:
            kwargs[field_name] = field_type.type(field_value)
    return cls(**kwargs)


def load_config(config_path: str, cls: T) -> T:
    with open(config_path) as f:
        data = yaml.load(f, ConfigLoader)
        config = dict_to_dataclass(data, cls)
    return config


class BaseEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.name.lower() == value:
                return member
        return None

    def __repr__(self):
        return self.name
