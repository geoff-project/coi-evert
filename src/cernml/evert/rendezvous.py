"""Rendez-vous queue with zero capacity for `asyncio`."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Deque, Generic, Optional, Tuple, TypeVar, Union, cast

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypeGuard
else:
    from typing import Self, TypeGuard

__all__ = ["RendezVousQueue", "QueueFull", "QueueEmpty"]


ItemT = TypeVar("ItemT")
"""Type variable for our queue."""

_Getter = asyncio.Future[ItemT]
"""A receiving end of our queue.

If a putter finds that there's already a getter waiting, it will
complete the getter's future with the sent item.
"""

_Putter = Tuple[ItemT, asyncio.Future[None]]
"""A sending end of our queue.

If a getter finds that there's already a putter waiting, it will take
the sent item and complete the putter's future.
"""


def _has_getters(
    queue: Union[Deque[_Getter[ItemT]], Deque[_Putter[ItemT]]]
) -> TypeGuard[Deque[_Getter]]:
    """Type guard for getter queues.

    Return True if there's at least one getter waiting in the queue.
    Return False if the queue is empty or the queue contains putters.

    This relies on the fact that a queue contains either nothing or
    *only* getters or *only* putters.
    """
    return bool(queue) and not isinstance(queue[0], tuple)


def _has_putters(
    queue: Union[Deque[_Getter[ItemT]], Deque[_Putter[ItemT]]]
) -> TypeGuard[Deque[_Putter]]:
    """Type guard for putter queues.

    Returns True if there's at least one putter waiting in the queue.
    Return False if the queue is empty or the queue contains getters.

    This relies on the fact that a queue contains either nothing or
    *only* getters or *only* putters.
    """
    return bool(queue) and isinstance(queue[0], tuple)


def _remove_silently(queue: Deque[ItemT], item: ItemT) -> None:
    """Like `list.remove()` but do nothing if *item* is missing."""
    try:
        queue.remove(item)
    except ValueError:
        pass


class QueueEmpty(Exception):
    """The item couldn't be received from the queue."""


class QueueFull(Exception):
    """The item couldn't be sent into the queue."""


class RendezVousQueue(Generic[ItemT]):
    """Zero-capacity (or rendez-vous) queue for `asyncio`.

    Unlike with actual queues, an unpaired `get()` or `put()` always
    blocks. Once two tasks are synchronized with each other (i.e. one
    blocks on getting and one on putting), they transfer the queued item
    and both unblock.

    The queue is fair, i.e. blocked tasks do not starve and will
    eventually unblock.

    Examples:

        >>> import asyncio
        ...
        >>> async def runner():
        ...     queue = RendezVousQueue()
        ...
        ...     async def putter(item):
        ...         await queue.put(item)
        ...
        ...     async def getter():
        ...         print(await queue.get())
        ...
        ...     task1 = asyncio.create_task(putter("Hello world!"))
        ...     task2 = asyncio.create_task(getter())
        ...     await asyncio.gather(task1, task2)
        ...
        >>> asyncio.run(runner())
        Hello world!
    """

    __slots__ = ("_loop", "_waiters")

    def __init__(self) -> None:
        self._loop = asyncio.get_event_loop()
        # Internally, we maintain a double-ended queue of getters or
        # putters. It's always all-putters or all-getters. This is
        # because if we'd try to put while a getter is in the deque,
        # we'll just remove the getter and pair up with it. The same
        # goes vice versa.
        # If the deque is empty (``bool(self._waiters) is False``), it
        # can be both a getter or a putter queue. Some care has to be
        # taken about this edge case.
        # We replace the deque with `None` once the queue has been
        # closed. This ensures no-one can operate on it anymore.
        self._waiters: Union[None, Deque[_Getter[ItemT]], Deque[_Putter[ItemT]]]
        self._waiters = Deque[Any]()

    def _getter_queue(self) -> Optional[Deque[_Getter[ItemT]]]:
        """Return the queue if it is empty or contains getters.

        If the queue contains putters, return `None` instead. If the
        queue has been closed, raise `~asyncio.CancelledError`.
        """
        if self._waiters is None:
            raise asyncio.CancelledError("queue has been closed")
        if _has_putters(self._waiters):
            return None
        # SAFETY: At this point, the queue is either empty or only
        # contains getters.
        return cast(Deque[_Getter[ItemT]], self._waiters)

    def _putter_queue(self) -> Optional[Deque[_Putter[ItemT]]]:
        """Return the queue if it is empty or contains putters.

        If the queue contains getters, return `None` instead. If the
        queue has been closed, raise `~asyncio.CancelledError`.
        """
        if self._waiters is None:
            raise asyncio.CancelledError("queue has been closed")
        if _has_getters(self._waiters):
            return None
        # SAFETY: At this point, the queue is either empty or only
        # contains putters.
        return cast(Deque[_Putter[ItemT]], self._waiters)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_args: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        """This is True if `close()` has been called, otherwise False."""
        return self._waiters is None

    def empty(self) -> bool:
        """Return True if `get_nowait()` would raise an exception.

        This is the case when there's currently no task blocked on
        `put()`. Note that a queue starts out as empty and `full()` at
        the same time.

        Example:

            >>> import asyncio
            ...
            >>> async def runner():
            ...     queue = RendezVousQueue()
            ...     for i in range(1, 4):
            ...         asyncio.create_task(queue.put(i))
            ...
            ...     await asyncio.sleep(0)
            ...     received = []
            ...     while not queue.empty():
            ...         received.append(queue.get_nowait())
            ...     return sorted(received)
            ...
            >>> asyncio.run(runner())
            [1, 2, 3]
        """
        return self._waiters is None or not _has_putters(self._waiters)

    def full(self) -> bool:
        """Return True if `put_nowait()` would raise an exception.

        This is the case when there's currently no task blocked on
        `get()`. Note that a queue starts out as empty and `full()` at
        the same time.

        Example:

            >>> import asyncio
            ...
            >>> async def runner():
            ...     queue = RendezVousQueue()
            ...     tasks = [
            ...         asyncio.create_task(queue.get()) for _ in range(3)
            ...     ]
            ...     await asyncio.sleep(0)
            ...     while not queue.full():
            ...         queue.put_nowait(None)
            ...     return await asyncio.gather(*tasks)
            ...
            >>> asyncio.run(runner())
            [None, None, None]
        """
        return self._waiters is None or not _has_getters(self._waiters)

    async def get(self) -> ItemT:
        """Synchronize with a `put()` and return its item.

        If there is no partner to synchronize with, wait until one
        becomes available. If the queue is closed while waiting, raise
        `~asyncio.CancelledError`.
        """
        # Check if we can get a getter queue. If not, it means there are
        # putters waiting that we can synchronize with. Repeat in case
        # all waiting putters are in the cancelled state.
        while (getters := self._getter_queue()) is None:
            try:
                return self.get_nowait()
            except QueueEmpty:
                pass
        # There are no putters in the queue, append a getter and wait
        # for it.
        getter = self._loop.create_future()
        getters.append(getter)
        try:
            return await getter
        except:  # noqa: E722
            getter.cancel("queue has been closed")
            _remove_silently(getters, getter)
            raise

    def get_nowait(self) -> ItemT:
        """Attempt to synchronize with a `put()` and return its item.

        If there is no partner to synchronize with, raise `QueueEmpty`
        without blocking. If the queue has already been closed, raise
        `~asyncio.CancelledError`.
        """
        putters = self._putter_queue()
        while putters:
            item, putter = putters.popleft()
            if not putter.done():
                putter.set_result(None)
                return item
        raise QueueEmpty()

    async def put(self, item: ItemT) -> None:
        """Synchronize with a `get()` and return its item.

        If there is no partner to synchronize with, wait until one
        becomes available. If the queue is closed while waiting, raise
        `~asyncio.CancelledError`.
        """
        # Check if we can get a putter queue. If not, it means there are
        # getters waiting that we can synchronize with. Repeat in case
        # all waiting getters are in the cancelled state.
        while (putters := self._putter_queue()) is None:
            try:
                return self.put_nowait(item)
            except QueueFull:
                pass
        # There are no getters in the queue, append a putter and wait
        # for it.
        putter = self._loop.create_future()
        putters.append((item, putter))
        try:
            await putter
        except:  # noqa: E722
            putter.cancel("queue has been closed")
            _remove_silently(putters, (item, putter))
            raise

    def put_nowait(self, item: ItemT) -> None:
        """Attempt to synchronize with a `get()` and return its item.

        If there is no partner to synchronize with, raise `QueueFull`
        without blocking. If the queue has already been closed, raise
        `~asyncio.CancelledError`.
        """
        getters = self._getter_queue()
        while getters:
            getter = getters.popleft()
            if not getter.done():
                getter.set_result(item)
                return
        raise QueueFull()

    def close(self) -> None:
        """Close the queue, preventing all further interactions.

        Because rendez-vous queues don't buffer their items, any task
        blocked when it is closed will immediately raise a
        `~asyncio.CancelledError`.

        Calling this method a second time does nothing.

        Example:

            >>> import asyncio
            ...
            >>> async def runner():
            ...     queue = RendezVousQueue()
            ...     task = asyncio.create_task(queue.put(1))
            ...     await asyncio.sleep(0)
            ...     assert not task.done()
            ...     queue.close()
            ...     await asyncio.sleep(0)
            ...     assert task.cancelled()
            ...
            >>> asyncio.run(runner())
        """
        if self._waiters is None:
            return
        waiters, self._waiters = self._waiters, None
        if _has_getters(waiters):
            for getter in waiters:
                getter.cancel("queue has been closed")
            return
        # SAFETY: At this point, the deque cannot contain getters, so it
        # either contains putters or is empty.
        waiters = cast(Deque[_Putter[ItemT]], waiters)
        for _, putter in waiters:
            putter.cancel("queue has been closed")
