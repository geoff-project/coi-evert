"""Two-way channels for communication between tasks."""

from __future__ import annotations
import asyncio
import typing as t
from .rendezvous import RendezVousQueue, QueueEmpty, QueueFull

SendT = t.TypeVar("SendT")
RecvT = t.TypeVar("RecvT")


def channel() -> t.Tuple[Connection[SendT, RecvT], Connection[RecvT, SendT]]:
    """Create a pair of `Connection` objects."""
    forward = RendezVousQueue[SendT]()
    backward = RendezVousQueue[RecvT]()
    return Connection(forward, backward), Connection(backward, forward)


class Connection(t.Generic[SendT, RecvT]):
    """Duplex connection with another task via `RendezVousQueue`.

    This class is typically instantiated via `channel()`. It creates two
    paired connection objects that can be used to exchange messages:

        >>> async def worker(conn):
        ...     item = await conn.get()
        ...     print("ping!")
        ...     await conn.put(item)
        ...     conn.close()
        ...
        >>> async def runner():
        ...     mine, yours = channel()
        ...     task = asyncio.create_task(worker(yours))
        ...     await mine.put("message")
        ...     item = await mine.get()
        ...     print("pong :)")
        ...     print("received a", item)
        ...     await task
        ...     assert mine.closed
        ...
        >>> asyncio.run(runner())
        ping!
        pong :)
        received a message
    """

    __slots__ = ("_send", "_recv")

    def __init__(
        self, send: RendezVousQueue[SendT], recv: RendezVousQueue[RecvT]
    ) -> None:
        self._send = send
        self._recv = recv

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection.

        This marks both sides of the connection as closed. If the other
        side is waiting while the channel is being closed, they receive
        a `~asyncio.CancelledError`.

        Calling this method a second time does nothing.
        """
        self._send.close()
        self._recv.close()

    @property
    def closed(self) -> bool:
        """Return True if either side has closed the connection."""
        return self._send.closed or self._recv.closed

    def empty(self) -> bool:
        """Return True if `get_nowait()` would raise an exception.

        This is the case when the other side's currently not trying to
        send a message. Note that a connection is usually empty and
        `full()` at the same time.
        """
        return self._recv.empty()

    def full(self) -> bool:
        """Return True if `put_nowait()` would raise an exception.

        This is the case when we're currently trying to send a message
        to the other side. Note that a queue starts out as empty and
        `full()` at the same time.
        """
        return self._send.full()

    async def get(self) -> RecvT:
        """Synchronize with the other side and return its message.

        If the other side is waiting for us, this returns immediately.
        If not, we wait until the other side becomes ready to send
        a message.

        Raises:
            `QueueEmpty`: if the other side is currently waiting for
                a message from us.
            `~asyncio.CancelledError`: if the queue is closed before or
                while waiting.
        """
        # If the other side is waiting on _send.get(), they cannot
        # _recv.put().
        if not self._send.full():
            raise QueueEmpty()
        try:
            return await self._recv.get()
        except asyncio.CancelledError:
            self._send.close()
            raise

    def get_nowait(self) -> RecvT:
        """Attempt to synchronize with the other side and return its message.

        If the other side is not ready, raise `QueueEmpty` without
        blocking. If the queue has already been closed, raise
        `~asyncio.CancelledError`.
        """
        try:
            return self._recv.get_nowait()
        except asyncio.CancelledError:
            self._send.close()
            raise

    async def put(self, item: SendT) -> None:
        """Synchronize with the other side and send a message.

        If the other side is waiting for us, this returns immediately.
        If not, we wait until the other side becomes ready to receive
        a message.

        Raises:
            `QueueFull`: if the other side is currently waiting to
                send a message to us.
            `~asyncio.CancelledError`: if the queue is closed before or
                while waiting.
        """
        # If the other side is waiting on _recv.put(), they cannot
        # _send.get().
        if not self._recv.empty():
            raise QueueFull()
        try:
            return await self._send.put(item)
        except asyncio.CancelledError:
            self._recv.close()
            raise

    def put_nowait(self, item: SendT) -> None:
        """Attempt to synchronize with the other side and send a message.

        If the other side is not ready, raise `QueueFull` without
        blocking. If the queue has already been closed, raise
        `~asyncio.CancelledError`.
        """
        try:
            return self._send.put_nowait(item)
        except asyncio.CancelledError:
            self._recv.close()
            raise
