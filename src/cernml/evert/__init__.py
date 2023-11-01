"""Main docs here."""

from . import asynch, synch
from ._types import (
    MethodOrderError,
    Objective,
    OptResult,
    SolveFunc,
)
from .synch import CancelledError, Eversion, evert

__all__ = [
    "CancelledError",
    "Eversion",
    "MethodOrderError",
    "Objective",
    "OptResult",
    "SolveFunc",
    "asynch",
    "evert",
    "synch",
]
