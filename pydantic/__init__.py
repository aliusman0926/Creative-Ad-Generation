"""Minimal stub of the pydantic API used for offline testing.

This is **not** a full replacement for pydantic but implements enough surface
area for lightweight validation within this repository when the dependency is
not available in the execution environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Union, get_args, get_origin


class ValidationError(Exception):
    """Simplified validation error mirroring pydantic's interface."""

    def __init__(self, errors: Iterable[Exception] | None = None) -> None:
        self.raw_errors = list(errors or [])
        super().__init__("Validation error")

    @classmethod
    def from_exception_data(cls, title: str, line_errors: List[Any]) -> "ValidationError":
        return cls(line_errors)


@dataclass
class Field:
    default: Any
    description: str | None = None


ValidatorFunc = Callable[[Any, Any], Any]


def validator(*field_names: str) -> Callable[[ValidatorFunc], ValidatorFunc]:
    def decorator(func: ValidatorFunc) -> ValidatorFunc:
        setattr(func, "__pydantic_validators__", field_names)
        return func

    return decorator


class BaseModel:
    """Lightweight BaseModel with minimal type coercion and validation hooks."""

    _validators: Dict[str, List[ValidatorFunc]]

    def __init_subclass__(cls) -> None:
        cls._validators = {}
        for attr_value in cls.__dict__.values():
            fields = getattr(attr_value, "__pydantic_validators__", None)
            if fields:
                for field in fields:
                    cls._validators.setdefault(field, []).append(attr_value)

    def __init__(self, **data: Any) -> None:
        errors: List[Exception] = []
        for field, field_type in self.__annotations__.items():
            value = data.get(field, getattr(self, field, None))
            value = self._coerce_value(value, field_type)

            for validator_fn in self._validators.get(field, []):
                try:
                    value = validator_fn(self.__class__, value)
                except Exception as exc:  # pragma: no cover - thin wrapper
                    errors.append(exc)
            setattr(self, field, value)

        if errors:
            raise ValidationError(errors)

    def dict(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__annotations__}

    @staticmethod
    def _coerce_value(value: Any, field_type: Any) -> Any:
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            allowed_types = tuple(arg for arg in args if arg is not type(None))
            if value is None:
                return None
            field_type = allowed_types[0]

        try:
            if field_type is float:
                return float(value)
            if field_type is int:
                return int(value)
            if field_type is str:
                return str(value)
        except Exception:
            return value

        return value


class BaseSettings(BaseModel):
    class Config:
        env_file: str = ""
        env_file_encoding: str = "utf-8"
        case_sensitive: bool = False
