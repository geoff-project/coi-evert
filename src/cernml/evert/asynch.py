"""Asynchronous version of the Eversion API.

This is implemented using `asyncio` and a background thread.
Communicating with the other thread requires waiting, which allows the
foreground thread to do other work while waiting for the background
thread.

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
    >>> async def main():
    ...     ev = evert(solve, "args 0")
    ...     try:
    ...         i = 0
    ...         while True:
    ...             i += 1
    ...             print("main received:", await ev.ask())
    ...             await ev.tell(f"result {i}")
    ...     except OptFinished as exc:
    ...         print("main received:", exc.result)
    ...
    >>> asyncio.run(main())
    main received: args 0
    solve received: result 1
    main received: args 1
    solve received: result 2
    main received: args 2
    solve received: result 3
    solve finished
    main received: final result
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import sys
import typing as t
from types import TracebackType

from ._types import Loss, MethodOrderError, OptFinished, OptResult, Params, SolveFunc
from .channel import Connection, channel
from .rendezvous import QueueEmpty, QueueFull

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypeAlias
else:
    from typing import Self, TypeAlias

__all__ = [
    "CancelledError",
    "Eversion",
    "MethodOrderError",
    "OptFinished",
    "SolveFunc",
    "evert",
]


CancelledError = asyncio.CancelledError


def evert(solve: SolveFunc[Params, Loss, OptResult], x0: Params) -> Eversion:
    """Turn a long-running function inside out.

    This assumes that *solve* is a function that receives a second
    function which it will call many times before eventually returning
    a result. This includes most numerical optimization algorithms.

    The returned `Eversion` object has methods `~Eversion.ask()` and
    `~Eversion.tell()` that manually progress the state machine inside
    *solve*, meaning that you can run it without yielding the control
    flow.
    """
    return Eversion(solve, x0)


class Eversion(t.Generic[Params, Loss, OptResult]):
    """Long-running function turned inside out. Instantiate via `evert()`."""

    def __init__(self, solve: SolveFunc[Params, Loss, OptResult], x0: Params) -> None:
        self._logger = logging.getLogger(__name__ + ".Eversion")
        self._worker: t.Union[
            _LazyArgs[Params, Loss, OptResult],
            _BackgroundWorker[Params, Loss, OptResult],
            None,
        ] = (solve, x0)

    async def __aenter__(self) -> Self:
        # Access the connection to implicitly start the background thread.
        _ = self._conn
        return self

    async def __aexit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> bool:
        # pylint: disable=broad-exception-caught
        try:
            await self.join()
        except (MethodOrderError, Exception) as exc:
            # Reraise the exception if there's no other exception in
            # flight.
            if exc_val is None:
                raise
            # If an exception is already in flight, it has already
            # overwritten the __context__ of `exc`, so we may as well
            # ignore i.
            exc.__suppress_context__ = True
            # If an exception is coming through, just log this one.
            # Otherwise, there are too many exceptions in flight.
            self._logger.exception(
                "an exception occurred in the background thread", exc_info=exc
            )
        # Suppress `OptFinished` if we were the originator.
        return isinstance(exc_val, OptFinished) and exc_val.origin is self

    @property
    def _conn(self) -> Connection[Loss, Params]:
        # Hide `self._worker` behind property access to ensure it's
        # always initialized before it's required.
        if self._worker is None:
            raise asyncio.CancelledError("solve() never started")
        if not isinstance(self._worker, _BackgroundWorker):
            self._logger.debug("Starting background thread …")
            solve, x0 = self._worker
            # Make sure not to hold onto `conn` so that it is dropped at
            # the right time!
            ours: Connection[Loss, Params]
            theirs: Connection[Params, Loss]
            ours, theirs = channel()
            self._worker = _BackgroundWorker(solve, x0, theirs)
            vars(self)["_conn"] = ours
            return ours
        return vars(self)["_conn"]

    def cancel(self) -> None:
        """Cancel the execution of the everted function.

        If the everted function has already terminated this does nothing.

        Note that this merely schedules an exception to be raised within
        the background thread that runs the function. To ensure that it
        has properly shut down, you have to `join()` and catch the
        resulting `~asyncio.CancelledError`.
        """
        if isinstance(self._worker, _BackgroundWorker):
            self._logger.debug("Canceling background thread …")
            self._conn.close()
        elif self._worker is not None:
            self._logger.debug("Canceling to prevent background thread from starting …")
            # Ensure that worker won't get started later.
            self._worker = None

    async def join(self) -> OptResult:
        """Await termination of the everted function.

        If the function isn't done yet, it is cancelled. The return
        value is that of the everted function if it was successful.

        If the everted function raised any exception, it is re-raised by
        this method.

        If the everted function was not finished yet,
        `asyncio.CancelledError` is raised.
        """
        # If worker hadn't started yet, `cancel()` will make it
        # unstartable.
        self.cancel()
        if inspect.isawaitable(self._worker):
            self._logger.debug("Joining thread …")
            return await self._worker
        raise asyncio.CancelledError("solve() never started")

    async def _join_ignore_exceptions(self) -> None:
        """Helper function to `ask()` and `get()`.

        Called when handling deadlock due to `MethodOrderError`.
        """
        # pylint: disable = bare-except
        try:
            await self.join()
        except:  # noqa: E722
            self._logger.debug("ignoring exception:", exc_info=True)

    async def ask(self) -> Params:
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
        try:
            self._logger.debug("Wait for params …")
            received = await self._conn.get()
        except QueueEmpty as exc:
            await self._join_ignore_exceptions()
            # coverage.py fails to recognize this line even though a
            # unit test asserts that it is reached.
            raise MethodOrderError(  # pragma: no cover
                called="ask", expected="tell", context="receiving params"
            ) from exc
        except asyncio.CancelledError:
            self._logger.debug("connection closed by background thread")
            result = await self.join()
            raise OptFinished(result, self) from None
        self._logger.debug("Get params: %s", received)
        return received

    async def tell(self, loss: Loss) -> None:
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
        self._logger.debug("Put loss: %s", loss)
        try:
            await self._conn.put(loss)
        except QueueFull as exc:
            await self._join_ignore_exceptions()
            # coverage.py fails to recognize this line even though a
            # unit test asserts that it is reached.
            raise MethodOrderError(  # pragma: no cover
                called="tell", expected="ask", context="sending loss"
            ) from exc
        except asyncio.CancelledError:
            self._logger.debug("connection closed by background thread")
            result = await self.join()
            raise OptFinished(result, self) from None

    async def as_generator(self) -> t.AsyncGenerator[Params, Loss]:
        """Alternative API based on asynchronous generator functions.

        This function returns an :term:`asynchronous generator`. It
        combines calls to `ask()` and `tell()` in a single function
        `agen.asend()`.

        To receive the first set of callback arguments, call either
        ``await anext(agen)`` (Python 3.10+) or ``await
        agent.__anext__()`` (until Python 3.9) or ``await
        agen.asend(None)`` (all versions).

        Note that generators terminate execution by raises
        `StopAsyncIteration` instead of `OptFinished`. To receive the
        final return value of the everted function, await `join()`.
        """
        try:
            i = 0
            while True:
                i += 1
                self._logger.debug("Iteration #%d", i)
                params = await self.ask()
                self._logger.debug("Yielding …")
                loss = yield params
                await self.tell(loss)
        except OptFinished:
            return

    def set_logger(self, logger: logging.Logger) -> None:
        """Replace the logger used internally by this class."""
        self._logger = logger


_LazyArgs: TypeAlias = tuple[SolveFunc[Params, Loss, OptResult], Params]


class _BackgroundWorker(t.Generic[Params, Loss, OptResult]):
    """Worker thread that runs `solve()` on a background thread.

    This could be a single function but that confuses Mypy.
    """

    # pylint: disable = too-few-public-methods

    def __init__(
        self,
        solve: SolveFunc[Params, Loss, OptResult],
        x0: Params,
        conn: Connection[Params, Loss],
    ) -> None:
        self.solve = solve
        self.x0 = x0
        # Important: `self` is reachable from the main thread and
        # `conn` closes when dropped. We should make sure not to
        # hold onto it ourselves.
        loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(
            asyncio.to_thread(self._solve_thread, conn, loop)
        )

    def __await__(self) -> t.Generator[t.Any, None, OptResult]:
        return self._task.__await__()

    def _solve_thread(
        self, conn: Connection[Params, Loss], loop: asyncio.AbstractEventLoop
    ) -> OptResult:
        logger = logging.getLogger(__name__ + ".SolveThread")

        def _put(params: Params) -> None:
            future = asyncio.run_coroutine_threadsafe(conn.put(params), loop)
            try:
                return future.result()
            except QueueFull as exc:
                raise MethodOrderError(
                    called="tell", expected="ask", context="sending params"
                ) from exc

        def _get() -> Loss:
            future = asyncio.run_coroutine_threadsafe(conn.get(), loop)
            try:
                return future.result()
            except QueueEmpty as exc:
                raise MethodOrderError(
                    called="ask", expected="tell", context="receiving loss"
                ) from exc

        def _objective(params: Params) -> Loss:
            logger.debug("Put params: %s", params)
            _put(params)
            logger.debug("Wait for loss …")
            loss = _get()
            logger.debug("Get loss: %s", loss)
            return loss

        try:
            logger.debug("Start optimization routine")
            result = self.solve(_objective, self.x0)
            logger.debug("Optimization routine finished!")
            logger.debug("Achieved result: %s", result)
            logger.debug("Closing connection to main thread")
            loop.call_soon_threadsafe(conn.close)
            return result
        except concurrent.futures.CancelledError:  # noqa: E722
            logger.debug("exiting after cancellation")
            raise
        except:  # noqa: E722
            logger.exception("exiting due to exception")
            raise
        finally:
            loop.call_soon_threadsafe(conn.close)
