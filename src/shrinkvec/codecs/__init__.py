"""Codecs by name: ``float32``, ``halfvec``, ``int8``, ``binary``, ``tq-mse-<b>``, ``tq-prod-<b>``, ``rabitq-<b>``."""

from __future__ import annotations

import re

from .base import Binary, Codec, Float16, Float32, Int8
from .rabitq import RaBitQ
from .turboquant import TurboQuant

__all__ = ["Codec", "make_codec", "Binary", "Float16", "Float32", "Int8", "RaBitQ", "TurboQuant"]

_FIXED = {"float32": Float32, "halfvec": Float16, "int8": Int8, "binary": Binary}


def make_codec(name: str) -> Codec:
    if name in _FIXED:
        return _FIXED[name]()
    if m := re.fullmatch(r"tq-(mse|prod)-(\d)", name):
        return TurboQuant(int(m.group(2)), m.group(1))
    if m := re.fullmatch(r"rabitq-(\d)", name):
        return RaBitQ(int(m.group(1)))
    raise ValueError(
        f"unknown codec '{name}'. Use float32, halfvec, int8, binary, tq-mse-<1-8>, tq-prod-<2-8> or rabitq-<1-8>"
    )
