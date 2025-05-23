# SPDX-FileCopyrightText: 2001-2023 Python Software Foundation
# SPDX-FileCopyrightText: Copyright 2023, The Python Typing Team
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: PSF-2.0 AND Apache-2.0

"""Polyfill of `asyncio.runners.Runner`.

The code has been copied from the Python 3.11 standard library,
available at
<https://github.com/python/cpython/blob/3.11/Lib/asyncio/runners.py>.
Type annotations have been copied from Typeshed, avaliable at
<https://github.com/python/typeshed/blob/main/stdlib/asyncio/runners.pyi>.

Finally, a workaround to bug
<https://github.com/python/cpython/issues/89553> has been added in the
method `Runner.run()` and marked with comments.

Their respective licenses apply. See the source file as well as COPYING
for more information.
"""

import asyncio
import contextvars
import enum
import functools
import signal
import sys
import threading
import typing as t

if t.TYPE_CHECKING:
    from types import FrameType

    from _typeshed import Unused

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self


__all__ = ["Runner"]


T = t.TypeVar("T")


class _State(enum.Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    CLOSED = "closed"


@t.final
class Runner:
    """A context manager that controls event loop life cycle.

    The context manager always creates a new event loop,
    allows to run async functions inside it,
    and properly finalizes the loop at the context manager exit.

    If debug is True, the event loop will be run in debug mode.
    If loop_factory is passed, it is used for new event loop creation.

    asyncio.run(main(), debug=True)

    is a shortcut for

    with asyncio.Runner(debug=True) as runner:
        runner.run(main())

    The run() method can be called multiple times within the runner's context.

    This can be useful for interactive console (e.g. IPython),
    unittest runners, console tools, -- everywhere when async code
    is called from existing sync framework and where the preferred single
    asyncio.run() call doesn't work.

    """

    # Note: the class is final, it is not intended for inheritance.

    def __init__(
        self,
        *,
        debug: t.Optional[bool] = None,
        loop_factory: t.Optional[t.Callable[[], asyncio.AbstractEventLoop]] = None,
    ) -> None:
        self._state = _State.CREATED
        self._debug = debug
        self._loop_factory = loop_factory
        self._loop: t.Optional[asyncio.AbstractEventLoop] = None
        self._context: t.Optional[contextvars.Context] = None
        self._interrupt_count = 0
        self._set_event_loop = False

    def __enter__(self) -> "Self":
        self._lazy_init()
        return self

    def __exit__(self, exc_type: "Unused", exc_val: "Unused", exc_tb: "Unused") -> None:
        self.close()

    def close(self) -> None:
        """Shutdown and close event loop."""
        if self._state is not _State.INITIALIZED:
            return
        assert self._loop is not None
        try:
            loop = self._loop
            _cancel_all_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            if self._set_event_loop:
                asyncio.set_event_loop(None)
            loop.close()
            self._loop = None
            self._state = _State.CLOSED

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Return embedded event loop."""
        self._lazy_init()
        assert self._loop is not None
        return self._loop

    def run(
        self,
        coro: t.Coroutine[None, t.Any, T],
        *,
        context: t.Optional[contextvars.Context] = None,
    ) -> T:
        """Run a coroutine inside the embedded event loop."""
        if not asyncio.iscoroutine(coro):
            raise ValueError(f"a coroutine was expected, got {coro!r}")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # fail fast with short traceback
            raise RuntimeError(
                "Runner.run() cannot be called from a running event loop"
            ) from None
        self._lazy_init()
        assert self._loop is not None

        if context is None:
            context = self._context
        # Can't pass `context` keyword argument, it didn't exist before
        # Python 3.11.
        task: asyncio.Task[T]
        if context is not None:
            task = context.run(self._loop.create_task, coro)
        else:
            task = self._loop.create_task(coro)

        if (
            threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGINT) is signal.default_int_handler
        ):
            sigint_handler: t.Optional[
                t.Callable[[int, t.Optional["FrameType"]], t.Any]
            ] = functools.partial(self._on_sigint, main_task=task)
            try:
                signal.signal(signal.SIGINT, sigint_handler)
            except ValueError:
                # `signal.signal` may throw if `threading.main_thread` does
                # not support signals (e.g. embedded interpreter with signals
                # not registered - see gh-91880)
                sigint_handler = None
        else:
            sigint_handler = None

        self._interrupt_count = 0
        try:
            return self._loop.run_until_complete(task)
        except asyncio.CancelledError as exc:
            if self._interrupt_count > 0:
                uncancel = getattr(task, "uncancel", None)
                if uncancel is not None and uncancel() == 0:
                    # pylint: disable = raise-missing-from
                    raise KeyboardInterrupt
            # Workaround to <https://github.com/python/cpython/issues/89553>
            # The `CancelledError` raised inside our coroutine is wrapped
            # inside another `CancelledError(msg=None)` before Python
            # 3.11. Undo the wrapping and reraise the original
            # exception. With Python 3.11+,
            # `asyncio.futures.Future._make_cancelled_error()` prevents
            # this wrapping from happening in the first place.
            if sys.version_info < (3, 11):
                if not exc.args and isinstance(exc.__context__, asyncio.CancelledError):
                    raise exc.__context__ from None
            raise  # CancelledError
        finally:
            if (
                sigint_handler is not None
                and signal.getsignal(signal.SIGINT) is sigint_handler
            ):
                signal.signal(signal.SIGINT, signal.default_int_handler)

    def _lazy_init(self) -> None:
        if self._state is _State.CLOSED:
            raise RuntimeError("Runner is closed")
        if self._state is _State.INITIALIZED:
            return
        if self._loop_factory is None:
            self._loop = asyncio.new_event_loop()
            if not self._set_event_loop:
                # Call set_event_loop only once to avoid calling
                # attach_loop multiple times on child watchers
                asyncio.set_event_loop(self._loop)
                self._set_event_loop = True
        else:
            self._loop = self._loop_factory()
        if self._debug is not None:
            self._loop.set_debug(self._debug)
        self._context = contextvars.copy_context()
        self._state = _State.INITIALIZED

    def _on_sigint(
        self, signum: int, frame: "FrameType", main_task: asyncio.Task[t.Any]
    ) -> None:
        # pylint: disable = unused-argument
        self._interrupt_count += 1
        if self._interrupt_count == 1 and not main_task.done():
            assert self._loop is not None
            main_task.cancel()
            # wakeup loop if it is blocked by select() with long timeout
            self._loop.call_soon_threadsafe(lambda: None)
            return
        raise KeyboardInterrupt


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


# On Python 3.11+, just use the stdlib Runner.
if sys.version_info >= (3, 11):
    from asyncio import Runner  # type: ignore[assignment] # noqa: F811
