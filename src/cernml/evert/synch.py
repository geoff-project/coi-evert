"""Synchronous, blocking version of the Eversion API.

Many optimization routines are implemented as a function called *solve*
that takes a callback function *objective* and an initial argument *x0*.
The *solve* function then runs a loop, potentially for a long time, and
calls *objective* repeatedly until it has calculated a result.
Because *solve* seizes the *control flow*, its caller is stuck waiting
for however long it takes and cannot do anything else.

Example:

    >>> def solve(objective, x0):
    ...     x = x0
    ...     for i in range(1, 4):
    ...         loss = objective(x)
    ...         x = f"args {i}"
    ...         print("solve received:", loss)
    ...     print("solve finished")
    ...     return "final result"
    ...
    >>> with evert(solve, "args 0") as ev:
    ...     i = 0
    ...     while True:
    ...         i += 1
    ...         try:
    ...             print("main received:", ev.ask())
    ...         except OptFinished as exc:
    ...             print("main received:", exc.result)
    ...             break
    ...         ev.tell(f"result {i}")
    main received: args 0
    solve received: result 1
    main received: args 1
    solve received: result 2
    main received: args 2
    solve received: result 3
    solve finished
    main received: final result

.. note::
    This API uses :doc:`asynch` behind the scenes. If you run any
    asynchronous code yourself, keep in mind that entering the eversion
    context via :keyword:`with` will necessarily manipulate the global
    event loop. If you don't run any asynchronous code, there is nothing
    you need to worry about.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import typing as t
from types import TracebackType

from ._runner import Runner
from ._types import Loss, MethodOrderError, OptFinished, OptResult, Params, SolveFunc
from .asynch import Eversion as _AsyncEversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

__all__ = [
    "CancelledError",
    "Eversion",
    "MethodOrderError",
    "OptFinished",
    "SolveFunc",
    "evert",
]


CancelledError = asyncio.CancelledError


@t.overload
def evert(solve: SolveFunc[Params, Loss, OptResult], x0: Params) -> Eversion:
    ...


@t.overload
def evert(
    solve: SolveFunc[Params, Loss, OptResult],
    x0: Params,
    *,
    debug: t.Optional[bool] = None,
    loop_factory: t.Optional[t.Callable[[], asyncio.AbstractEventLoop]] = None,
) -> Eversion:
    ...


def evert(
    solve: SolveFunc[Params, Loss, OptResult],
    x0: Params,
    *,
    debug: t.Optional[bool] = None,
    loop_factory: t.Optional[t.Callable[[], asyncio.AbstractEventLoop]] = None,
) -> Eversion:
    """Turn a long-running function inside out.

    This assumes that *solve* is a function that receives a second
    function which it will call many times before eventually returning
    a result. This includes most numerical optimization algorithms.

    The returned `Eversion` object has methods `~Eversion.ask()` and
    `~Eversion.tell()` that manually progress the state machine inside
    *solve*, meaning that you can run it without yielding the control
    flow.

    Args:
        solve: the function to turn inside out.
        x0: the second parameter to *solve* beside the callback
            function. Most numerical optimization algorithms will pass
            this directly to the first call to the objective.
        debug: if True, run the internal event loop in debug mode. If
            False, disable debug mode. If not passed, respect the global
            debug settings.
        loop_factory: if passed, use this to create an event loop that
            drives communication with the everted function's thread.
    """
    return Eversion(solve, x0, debug=debug, loop_factory=loop_factory)


class Eversion(t.Generic[Params, Loss, OptResult]):
    """Long-running function turned inside out. Instantiate via `evert()`."""

    def __init__(
        self,
        solve: SolveFunc[Params, Loss, OptResult],
        x0: Params,
        *,
        debug: t.Optional[bool] = None,
        loop_factory: t.Optional[t.Callable[[], asyncio.AbstractEventLoop]] = None,
    ) -> None:
        self._logger = logging.getLogger(__name__ + ".Eversion")
        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._closed = False
        self._inner = _AsyncEversion(solve, x0)
        self._inner.set_logger(self._logger)

    def __enter__(self) -> Self:
        self._runner.__enter__()
        self._runner.run(self._inner.__aenter__())
        return self

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> bool:
        try:
            return self._runner.run(self._inner.__aexit__(exc_type, exc_val, exc_tb))
        except asyncio.CancelledError:
            # The worker didn't terminate, e.g. because we lost
            # interest. This is expected behavior.
            return False
        finally:
            self._runner.__exit__(exc_type, exc_val, exc_tb)
            self._closed = True

    def ask(self) -> Params:
        """Progress the everted function by asking for the next callback.

        This should be called first and after every call to `tell()`. If
        the function has not finished yet, it will call its callback
        again. This will deliver a new set of parameters for us to
        evaluate.

        This method propagates any exception that terminates the everted
        function. It may furthermore raise the following exceptions.

        Raises:
            `OptFinished`: if the everted function has terminated
                instead of calling back. The exception has an attribute
                *result* that contains the return value.
            `RuntimeError`: if you call this method when you were
                expected to call `tell()` instead.
        """
        return self._runner.run(self._inner.ask())

    def tell(self, loss: Loss) -> None:
        """Progress the everted function by finishing its callback.

        This should be called after every call to `ask()`. The argument
        is passed back as the return value of the callback that the
        everted function uses.

        After this, you're expected to call `ask()` again to wait for
        the next callback.

        This method propagates any exception that terminates the everted
        function. It may furthermore raise the following exceptions.

        Raises:
            `OptFinished`: if the everted function has terminated
                instead of calling back. This can only happen if `ask()`
                has previously also raised this exception.
            `RuntimeError`: if you call this method when you were
                expected to call `ask()` instead.
        """
        return self._runner.run(self._inner.tell(loss))

    def as_generator(self) -> t.Generator[Params, Loss, OptResult]:
        """Alternative API based on generator functions.

        This function returns an :term:`generator`. It combines calls to
        `ask()` and `tell()` in a single function ``gen.send()``.

        To receive the first set of callback arguments, call either
        ``next(gen)`` or ``gen.send(None)``.

        The generator terminates by raising `StopIteration`. You can
        retrieve the result of the everted function either from the
        *args* attribute of the exception, or by calling `join()`
        explicitly.

        The generator propagates any exception that terminates the everted
        function.
        """
        agen = self._inner.as_generator()
        # Wrappers to make Mypy understand that built-in async
        # generators return coroutine objects, not just awaitables.
        next_ = t.cast(t.Callable[[], t.Coroutine[None, t.Any, Params]], agen.__anext__)
        send = t.cast(t.Callable[[Loss], t.Coroutine[None, t.Any, Params]], agen.asend)
        try:
            params = self._runner.run(next_())
            while True:
                loss = yield params
                params = self._runner.run(send(loss))
        except StopAsyncIteration:
            pass
        self._logger.info("Joining thread …")
        result = self.join()
        self._logger.info("Result: %s", result)
        return result

    def cancel(self) -> None:
        """Cancel the execution of the everted function.

        Note that this merely schedules an exception to be raised within
        the background thread that runs the function. To ensure that it
        has properly shut down, you have to `join()` and catch the
        resulting `~asyncio.CancelledError`.
        """
        return self._inner.cancel()

    def join(self) -> OptResult:
        """Await termination of the everted function.

        If the function isn't done yet, it is cancelled. The return
        value is that of the everted function if it was successful.

        If the everted function raised any exception, it is re-raised by
        this method.

        If the everted function was not finished yet,
        `asyncio.CancelledError` is raised.
        """
        if self._closed:
            # Can't use runner anymore, side-step it so that `OptResult`
            # remains available after the with block. This will never
            # deadlock, but produce CancelledError at worst.
            awaitable = self._inner.join().__await__()
            try:
                while True:
                    next(awaitable)
            except StopIteration as stop:
                return stop.value
        return self._runner.run(self._inner.join())

    def set_logger(self, logger: logging.Logger) -> None:
        """Replace the logger used internally by this class."""
        self._logger = logger
        self._inner.set_logger(logger)
