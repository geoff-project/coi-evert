"""Tests for `channel()`."""

import asyncio
import typing as t
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from cernml.evert.channel import Connection, QueueEmpty, QueueFull, channel

T = t.TypeVar("T")  # pylint: disable=invalid-name


@contextmanager
def autocancel_task(
    coro: t.Coroutine[None, t.Any, T], name: t.Optional[str] = None
) -> t.Iterator[asyncio.Task[T]]:
    """Context manager to create a task that is cancelled upon exit.

    This is a thin wrapper around `asyncio.create_task()` that ensures
    that tasks are cancelled if an exception is raised. This prevents
    noisy warnings if a test fails.
    """
    try:
        task = asyncio.create_task(coro, name=name)
        yield task
    finally:
        task.cancel()


async def yield_to_other_tasks() -> None:
    """Give other tasks a chance to run.

    This is simply ``asyncio.sleep(0)``. It's purpose is to let other
    tasks progress in their state so that certain test invariants are
    met.
    """
    await asyncio.sleep(0)


async def test_default_is_empty_and_full() -> None:
    send: Connection[None, None]
    recv: Connection[None, None]
    send, recv = channel()
    assert send.empty()
    assert send.full()
    assert recv.empty()
    assert recv.full()


async def test_not_empty_if_receiving() -> None:
    async def sender(conn: Connection[int, int]) -> None:
        await conn.put(1)

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(sender(send)) as task:
        await yield_to_other_tasks()  # wait for put() to start waiting
        assert not recv.empty()
        assert recv.get_nowait() == 1
        await task
        assert recv.empty()


async def test_not_full_if_receiving() -> None:
    async def receiver(conn: Connection[int, int]) -> None:
        assert await conn.get() == 1

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(receiver(recv)) as task:
        await yield_to_other_tasks()  # wait for get() to start waiting
        assert not send.full()
        send.put_nowait(1)
        await task
        assert recv.full()


async def test_getting_while_getting() -> None:
    async def receiver(conn: Connection[int, int]) -> None:
        await conn.get()

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(receiver(recv)):
        await yield_to_other_tasks()  # wait for get() to start waiting
        with pytest.raises(QueueEmpty):
            await send.get()


async def test_putting_while_putting() -> None:
    async def sender(conn: Connection[int, int]) -> None:
        await conn.put(1)

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(sender(send)):
        await yield_to_other_tasks()  # wait for put() to start waiting
        with pytest.raises(QueueFull):
            await recv.put(1)


async def test_close_closes_both_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    close = Mock(name="close")
    monkeypatch.setattr("cernml.evert.channel.RendezvousQueue.close", close)
    for i in range(2):
        conns: t.Tuple[Connection[int, int], Connection[int, int]]
        conns = channel()
        conn = conns[i]
        # Location is important because the Connection finalizer also
        # calls `close()`
        close.reset_mock()
        assert close.call_count == 0, f"i = {i}"
        conn.close()
        assert close.call_count == 2, f"i = {i}"


async def test_dropping_sender_closes() -> None:
    async def sender(conn: Connection[int, int]) -> None:
        await conn.put(1)

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(sender(send)) as task:
        del send
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Danger: this is sensitive to when the GC runs. It works
        # reliably on Python 3.11, but other versions might break it.
        assert not recv.closed
        await yield_to_other_tasks()
        assert recv.closed
        with pytest.raises(asyncio.CancelledError):
            recv.get_nowait()


async def test_dropping_receiver_closes() -> None:
    async def receiver(conn: Connection[int, int]) -> None:
        await conn.get()

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(receiver(recv)) as task:
        del recv
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Danger: this is sensitive to when the GC runs. It works
        # reliably on Python 3.11, but other versions might break it.
        assert not send.closed
        await yield_to_other_tasks()
        assert send.closed
        with pytest.raises(asyncio.CancelledError):
            send.put_nowait(1)


async def test_context_manager() -> None:
    async def task(conn: Connection[int, int]) -> None:
        with conn:
            pass

    send: Connection[int, int]
    recv: Connection[int, int]
    send, recv = channel()
    with autocancel_task(task(recv)) as t:
        await t
    assert send.closed
