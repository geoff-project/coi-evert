"""Some simple type definitions."""

from __future__ import annotations

import typing as t
import weakref

if t.TYPE_CHECKING:
    from .asynch import Eversion as AsyncEversion

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

    __slots__ = ("result", "_origin")

    def __init__(self, result: OptResult, origin: AsyncEversion) -> None:
        super().__init__(result)
        self.result = result
        self._origin: t.Callable[[], t.Optional[AsyncEversion]] = weakref.ref(origin)

    @property
    def origin(self) -> t.Optional[AsyncEversion]:
        """The eversion object that raised this exception, or None.

        This property only exists for internal purposes. There is
        usually no need to use it. Note that it internally uses
        :mod:`weakref`, so the eversion object might no longer be alive
        by the point you access this property.
        """
        return self._origin()


class MethodOrderError(RuntimeError):
    """Raised when `Eversion` methods are called in the wrong order."""

    def __init__(self, *, called: str, expected: str, context: str) -> None:
        super().__init__(
            f"called {called}() when {expected}() should be called when {context}"
        )
