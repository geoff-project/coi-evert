# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Duplex channels for communication between two tasks.

A duplex channel faciliates *bidirectional* communication between
exactly two partners. Both may send and receive items to and from each
other. The intended use case is communication between a primary task and
a background worker task.

Channels internally use two instances of `~.rendezvous.RendezvousQueue`,
so both sides will progress in lockstep with each other.

Call `channel()` to create a channel with two `Connection` sides. Pass
one connection to the partner task and keep the other connection for
yourself. Then, the partner can `~Connection.get()` all items
that you `~Connection.put()` and vice versa.

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

Each connection is a :term:`context manager` that will close itself when
leaving its associated :keyword:`with` block:

    >>> async def runner():
    ...     conn, _ = channel()
    ...     with conn:
    ...         print("closed inside with-block?", conn.closed)
    ...     print("closed outside with-block?", conn.closed)
    ...
    >>> asyncio.run(runner())
    closed inside with-block? False
    closed outside with-block? True

In addition, to prevent dead-locks, a connection will also close itself
when finalized by the garbage collector. This will prevent you from
waiting for a done task that will never reply again:

    >>> async def connection_refuser(conn):
    ...     pass
    ...
    >>> async def runner():
    ...     mine, yours = channel()
    ...     partner = asyncio.create_task(connection_refuser(yours))
    ...     # Delete reference to the other side to avoid keeping
    ...     # it alive on accident.
    ...     del yours
    ...     await partner
    ...     # At this point, the other side is done and all its local
    ...     # variables have been garbage-collected.
    ...     print("closed?", mine.closed)
    ...
    >>> asyncio.run(runner())
    closed? True
"""

from __future__ import annotations

import asyncio
import sys
import typing as t
from logging import getLogger

from .rendezvous import QueueEmpty, QueueFull, RendezvousQueue

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

__all__ = [
    "Connection",
    "QueueEmpty",
    "QueueFull",
    "channel",
]


LOG = getLogger(__name__)


SendT = t.TypeVar("SendT")
RecvT = t.TypeVar("RecvT")


def channel() -> t.Tuple[Connection[SendT, RecvT], Connection[RecvT, SendT]]:
    """Create a pair of `Connection` objects."""
    # pylint: disable = unused-argument
    forward = RendezvousQueue[SendT]()
    backward = RendezvousQueue[RecvT]()
    return Connection(forward, backward), Connection(backward, forward)


class Connection(t.Generic[SendT, RecvT]):
    """Duplex connection with another task.

    Instantiate connections via `channel()`. It creates two paired
    connection objects that can be used to exchange messages. To prevent
    bugs, you should ensure that each task holds onto only one
    connection object.

    The API of this class follows roughly that of `multiprocessing.Pipe`
    as inspiration.
    """

    __slots__ = ("_send", "_recv")

    def __init__(
        self, send: RendezvousQueue[SendT], recv: RendezvousQueue[RecvT]
    ) -> None:
        self._send = send
        self._recv = recv

    def __del__(self) -> None:
        # Note that we use `and`, but `self.closed` uses `or`!
        if not self._send.closed and not self._recv.closed:
            LOG.warning("channel %s deleted without closing", self)
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_args: object) -> None:
        self.close()

    async def get(self) -> RecvT:
        """Synchronize with the other side and return its message.

        If the other side is waiting for us, this returns immediately.
        If not, we wait until the other side becomes ready to send
        a message.

        Raises:
            `QueueEmpty`: if the other side is currently waiting for
                a message from us. (This would be a deadlock otherwise.)
            `~asyncio.CancelledError`: if the channel is closed before
                or while waiting.
        """
        # If the other side is waiting on _send.get(), they won't
        # _recv.put(). That's a deadlock.
        if not self._send.full():
            raise QueueEmpty
        try:
            return await self._recv.get()
        except asyncio.CancelledError:
            self._send.close()
            raise

    def get_nowait(self) -> RecvT:
        """Attempt to synchronize with the other side and return its message.

        Raises:
            `QueueEmpty`: if the other side is not ready.
            `~asyncio.CancelledError`: if the channel has already been
                closed.
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
                send a message to us. (This would be a deadlock
                otherwise.)
            `~asyncio.CancelledError`: if the channel is closed before
                or while waiting.
        """
        # If the other side is waiting on _recv.put(), they won't
        # _send.get(). That's a deadlock.
        if not self._recv.empty():
            raise QueueFull
        try:
            return await self._send.put(item)
        except asyncio.CancelledError:
            self._recv.close()
            raise

    def put_nowait(self, item: SendT) -> None:
        """Attempt to synchronize with the other side and send a message.

        Raises:
            `QueueFull`: if the other side is not ready.
            `~asyncio.CancelledError`: if the channel has already been
                closed.
        """
        try:
            return self._send.put_nowait(item)
        except asyncio.CancelledError:
            self._recv.close()
            raise

    @property
    def closed(self) -> bool:
        """Return True if either side has closed its connection."""
        return self._send.closed or self._recv.closed

    def close(self) -> None:
        """Close the channel.

        This marks both connection of the channel as closed. If the
        other task is waiting when ``close()`` is called,
        a `~asyncio.CancelledError` is raised in it.

        This method is idempotent: calling it a second time does
        nothing.
        """
        self._send.close()
        self._recv.close()

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
        to the other side. Note that a queue starts out as `empty()` and
        full at the same time.
        """
        return self._send.full()
