"""Some simple type definitions."""

import typing as t

__all__ = [
    "Loss",
    "MethodOrderError",
    "Objective",
    "OptFinished",
    "OptResult",
    "Params",
    "SolveFunc",
]

Params = t.TypeVar("Params")
Loss = t.TypeVar("Loss")
OptResult = t.TypeVar("OptResult")

Objective = t.Callable[[Params], Loss]
SolveFunc = t.Callable[[Objective[Params, Loss], Params], OptResult]


class OptFinished(Exception, t.Generic[OptResult]):
    """Raised by `Eversion.ask()` if the optimization has finished."""

    __slots__ = ("result",)

    def __init__(self, result: OptResult) -> None:
        super().__init__(result)
        self.result = result


class MethodOrderError(RuntimeError):
    """Raised when `Eversion` methods are called in the wrong order."""

    def __init__(self, *, called: str, expected: str, context: str) -> None:
        super().__init__(
            f"called {called}() when {expected}() should be called when {context}"
        )
