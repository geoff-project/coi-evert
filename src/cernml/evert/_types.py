# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

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

T = t.TypeVar("T")


class OptFinished(Exception, t.Generic[Params, Loss, OptResult]):
    """Raised by the eversion if the optimization has finished.

    Both `~cernml.evert.synch.Eversion.ask()` and
    `~cernml.evert.synch.Eversion.tell()` can raise this exception. In
    practice, the former is vastly more likely due to how the two
    threads synchronize.

    Attributes:
        result (`cernml.evert._types.OptResult`): The return value of
            `SolveFunc`. If you have no use for it where the exception
            is caught, you can always call
            `~cernml.evert.synch.Eversion.join()` at a later point to
            receive it again.
    """

    __slots__ = ("result", "_origin")

    def __init__(
        self, result: OptResult, origin: AsyncEversion[Params, Loss, OptResult]
    ) -> None:
        super().__init__(result)
        self.result = result
        self._origin: t.Callable[
            [], t.Optional[AsyncEversion[Params, Loss, OptResult]]
        ] = weakref.ref(origin)

    @property
    def origin(self) -> t.Optional[AsyncEversion[Params, Loss, OptResult]]:
        """The eversion object that raised this exception, or None.

        This property only exists for internal purposes. There is
        usually no need to use it. Note that it internally uses
        :mod:`weakref`, so the eversion object might no longer be alive
        by the point you access this property.
        """
        return self._origin()


class MethodOrderError(RuntimeError):
    """Raised when eversion methods are called in the wrong order.

    Users are expected to call `~cernml.evert.synch.Eversion.ask()`
    first and then alternate between it and
    `~cernml.evert.synch.Eversion.tell()`. This exception is raised if
    the user calls ``ask()`` when ``tell()`` was expected, or vice
    versa.

    Whether the exception is raised on the main thread or the background
    thread depends on the precise timing between the two. In particular,
    this means that there is no guarantee that `SolveFunc` will or will
    not observe this exception.
    """

    def __init__(self, *, called: str, expected: str, context: str) -> None:
        super().__init__(
            f"called {called}() when {expected}() should be called when {context}"
        )
