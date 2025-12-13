"""Pytest configuration and compatibility shims."""

from typing import ForwardRef

import pydantic.typing as pyd_typing


# pydantic<1.10.14 does not handle Python 3.12's keyword-only ``recursive_guard``
# argument when calling ``ForwardRef._evaluate``. In such environments FastAPI
# imports can raise ``TypeError: ForwardRef._evaluate() missing 1 required
# keyword-only argument: 'recursive_guard'`` during model initialization.
# This shim retries with a keyword argument to maintain compatibility without
# modifying application code.
_original_evaluate_forwardref = pyd_typing.evaluate_forwardref


def _patched_evaluate_forwardref(type_, globalns, localns=None):
    try:
        return _original_evaluate_forwardref(type_, globalns, localns)
    except TypeError as exc:  # pragma: no cover - compatibility path
        if "recursive_guard" in str(exc) and isinstance(type_, ForwardRef):
            return type_._evaluate(globalns, localns, recursive_guard=set())
        raise


pyd_typing.evaluate_forwardref = _patched_evaluate_forwardref
