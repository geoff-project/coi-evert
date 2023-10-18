"""Some simple type definitions."""

import typing as t

__all__ = [
    "Loss",
    "Objective",
    "OptResult",
    "Params",
    "SolveFunc",
]

Params = t.TypeVar("Params")
Loss = t.TypeVar("Loss")
OptResult = t.TypeVar("OptResult")

Objective = t.Callable[[Params], Loss]
SolveFunc = t.Callable[[Objective[Params, Loss], Params], OptResult]
