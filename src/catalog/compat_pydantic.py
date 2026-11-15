"""
Compatibility shim providing Pydantic v2 core interface without external dependencies.
If pydantic is installed, wraps real pydantic classes; otherwise provides pure-Python equivalents.
Handles stringified annotations under Python 3.14 `from __future__ import annotations`.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

T = TypeVar("T", bound="BaseModel")

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
    BaseModel = _PydanticBaseModel
except ImportError:
    PYDANTIC_AVAILABLE = False

    class _FieldInfo:
        def __init__(self, default=..., default_factory=None, description=None, ge=None, le=None, min_length=None, **kwargs):
            self.default = default
            self.default_factory = default_factory
            self.description = description
            self.ge = ge
            self.le = le
            self.min_length = min_length
            self.extra = kwargs

    def Field(default=..., *, default_factory=None, description=None, ge=None, le=None, min_length=None, **kwargs):
        return _FieldInfo(default=default, default_factory=default_factory, description=description, ge=ge, le=le, min_length=min_length, **kwargs)

    def field_validator(*field_names, mode="after"):
        def decorator(fn):
            fn.__field_validator_fields__ = field_names
            fn.__field_validator_mode__ = mode
            return fn
        return decorator

    def model_validator(*, mode="after"):
        def decorator(fn):
            fn.__model_validator_mode__ = mode
            return fn
        return decorator

    class BaseModel:
        """Lightweight pure-python Pydantic v2 compatible BaseModel."""
        def __init__(self, **kwargs: Any) -> None:
            cls = self.__class__
            type_hints = {}
            for base in reversed(cls.__mro__):
                if hasattr(base, '__annotations__'):
                    type_hints.update(base.__annotations__)

            # Set defaults or passed values
            for attr_name in type_hints:
                field_def = getattr(cls, attr_name, ...)
                if attr_name in kwargs:
                    val = kwargs[attr_name]
                elif isinstance(field_def, _FieldInfo):
                    if field_def.default_factory is not None:
                        val = field_def.default_factory()
                    elif field_def.default is not ...:
                        val = copy.deepcopy(field_def.default)
                    else:
                        val = None
                elif field_def is not ...:
                    val = copy.deepcopy(field_def)
                else:
                    val = None

                setattr(self, attr_name, val)

            # Store any extra kwargs
            for k, v in kwargs.items():
                if k not in type_hints:
                    setattr(self, k, v)

            # Run field validators
            for name, member in cls.__dict__.items():
                if hasattr(member, "__field_validator_fields__"):
                    fn = member.__func__ if isinstance(member, classmethod) else member
                    for f in member.__field_validator_fields__:
                        curr_val = getattr(self, f, None)
                        try:
                            new_val = fn(cls if isinstance(member, classmethod) else self, curr_val)
                            setattr(self, f, new_val)
                        except TypeError:
                            new_val = fn(curr_val)
                            setattr(self, f, new_val)

            # Run model validators
            for name, member in cls.__dict__.items():
                if hasattr(member, "__model_validator_mode__"):
                    fn = getattr(self, name)
                    fn()

        def model_dump(self, *, mode: str = "python", **kwargs: Any) -> Dict[str, Any]:
            res: Dict[str, Any] = {}
            for k, v in self.__dict__.items():
                if k.startswith("_"):
                    continue
                res[k] = self._serialize_value(v, mode=mode)
            return res

        def dict(self, **kwargs: Any) -> Dict[str, Any]:
            return self.model_dump(**kwargs)

        def model_dump_json(self, *, indent: Optional[int] = None, **kwargs: Any) -> str:
            return json.dumps(self.model_dump(mode="json"), indent=indent, default=str)

        def json(self, **kwargs: Any) -> str:
            return self.model_dump_json(**kwargs)

        @classmethod
        def model_validate(cls: Type[T], obj: Any) -> T:
            if isinstance(obj, cls):
                return obj
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict, got {type(obj)}")
            
            cleaned: Dict[str, Any] = {}
            hints = {}
            for base in reversed(cls.__mro__):
                if hasattr(base, '__annotations__'):
                    hints.update(base.__annotations__)

            # Get namespace of caller module for resolving type hints
            mod = sys.modules.get(cls.__module__)
            ns = mod.__dict__ if mod else {}

            for k, v in obj.items():
                target_type = hints.get(k)
                cleaned[k] = cls._coerce_value(v, target_type, ns)
            return cls(**cleaned)

        @classmethod
        def parse_obj(cls: Type[T], obj: Any) -> T:
            return cls.model_validate(obj)

        @classmethod
        def _resolve_type(cls, hint: Any, ns: dict) -> Any:
            if isinstance(hint, str):
                hint = hint.strip()
                # strip optional | None
                hint_clean = re.sub(r"\s*\|\s*None", "", hint).strip()
                if hint_clean in ns:
                    return ns[hint_clean]
                if hint_clean in ("int", "str", "float", "bool", "dict"):
                    return __builtins__.get(hint_clean, hint)
                if hint_clean == "date":
                    return date
                if hint_clean == "datetime":
                    return datetime
                return hint_clean
            return hint

        @classmethod
        def _coerce_value(cls, val: Any, hint: Any, ns: dict) -> Any:
            if val is None:
                return None
            if hint is None:
                return val

            # Check if list type
            is_list = False
            sub_hint = None

            if isinstance(hint, str):
                m = re.match(r"(?:list|List)\[(.*)\]", hint.strip())
                if m:
                    is_list = True
                    sub_hint = m.group(1).strip()
            elif getattr(hint, "__origin__", None) in (list, List):
                is_list = True
                args = getattr(hint, "__args__", None)
                sub_hint = args[0] if args else None

            if is_list:
                if isinstance(val, list):
                    resolved_sub = cls._resolve_type(sub_hint, ns)
                    return [cls._coerce_value(item, resolved_sub, ns) for item in val]
                return val

            resolved = cls._resolve_type(hint, ns)

            # Date / Datetime
            if resolved in (date, "date") and isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.split("T")[0]).date()
                except Exception:
                    return val
            if resolved in (datetime, "datetime") and isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    return val

            # Nested BaseModel
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                if isinstance(val, dict):
                    return resolved.model_validate(val)
                return val

            # Enum
            if isinstance(resolved, type) and issubclass(resolved, Enum):
                if hasattr(resolved, "from_string"):
                    return resolved.from_string(val)
                try:
                    return resolved(val)
                except Exception:
                    return val

            return val

        def _serialize_value(self, val: Any, mode: str = "python") -> Any:
            if isinstance(val, BaseModel):
                return val.model_dump(mode=mode)
            if isinstance(val, Enum):
                return val.value
            if isinstance(val, (date, datetime)):
                return val.isoformat() if mode == "json" else val
            if isinstance(val, list):
                return [self._serialize_value(x, mode=mode) for x in val]
            if isinstance(val, dict):
                return {k: self._serialize_value(v, mode=mode) for k, v in val.items()}
            return val

        def __repr__(self) -> str:
            attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
            return f"{self.__class__.__name__}({attrs})"
