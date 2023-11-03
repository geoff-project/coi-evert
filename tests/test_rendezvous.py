"""Tests for `RendezvousQueue`."""

import asyncio
import random
import typing as t
from contextlib import contextmanager

import pytest

from cernml.evert.rendezvous import QueueEmpty, QueueFull, RendezvousQueue

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


async def test_context_manager() -> None:
    with RendezvousQueue[t.Any]() as queue:
        assert not queue.closed
    assert queue.closed


async def test_default_is_empty_and_full() -> None:
    queue = RendezvousQueue[t.Any]()
    assert queue.empty()
    assert queue.full()


async def test_empty_after_get() -> None:
    queue = RendezvousQueue[t.Any]()
    with autocancel_task(queue.get()):
        await yield_to_other_tasks()
        assert queue.empty()


async def test_not_empty_after_put() -> None:
    queue = RendezvousQueue[t.Any]()
    with autocancel_task(queue.put(object())):
        await yield_to_other_tasks()
        assert not queue.empty()


async def test_not_full_after_get() -> None:
    queue = RendezvousQueue[t.Any]()
    with autocancel_task(queue.get()):
        await yield_to_other_tasks()
        assert not queue.full()


async def test_full_after_put() -> None:
    queue = RendezvousQueue[t.Any]()
    with autocancel_task(queue.put(object())):
        await yield_to_other_tasks()
        assert queue.full()


async def test_get_nowait() -> None:
    queue = RendezvousQueue[t.Any]()
    sent = object()
    task = asyncio.create_task(queue.put(sent))
    with pytest.raises(QueueEmpty):
        queue.get_nowait()
    await yield_to_other_tasks()
    assert not task.done()
    received = queue.get_nowait()
    await task
    assert sent is received
    assert task.done()


async def test_put_nowait() -> None:
    queue = RendezvousQueue[t.Any]()
    sent = object()
    task = asyncio.create_task(queue.get())
    with pytest.raises(QueueFull):
        queue.put_nowait(sent)
    await yield_to_other_tasks()
    assert not task.done()
    queue.put_nowait(sent)
    received = await task
    assert sent is received
    assert task.done()


async def test_put_get() -> None:
    queue = RendezvousQueue[t.Any]()
    sent = object()
    received, none = await asyncio.gather(queue.get(), queue.put(sent))
    assert none is None
    assert sent is received


async def test_get_after_cancelled_put() -> None:
    queue = RendezvousQueue[t.Any]()
    putter = asyncio.create_task(queue.put(False))
    await yield_to_other_tasks()
    putter.cancel()
    assert not putter.done()
    assert not queue.empty()
    _ = asyncio.create_task(queue.put(True))
    assert await queue.get()


async def test_put_after_cancelled_get() -> None:
    queue = RendezvousQueue[t.Any]()
    getter = asyncio.create_task(queue.get())
    await yield_to_other_tasks()
    getter.cancel()
    assert not getter.done()
    assert not queue.full()
    _ = asyncio.create_task(queue.get())
    await queue.put(True)


async def test_many_puts_before_get() -> None:
    queue = RendezvousQueue[t.Any]()
    items = [object() for _ in range(100)]
    tasks = [asyncio.create_task(queue.put(i)) for i in items]
    await yield_to_other_tasks()
    received = [queue.get_nowait() for _ in range(len(items))]
    assert items == received
    done, pending = await asyncio.wait(tasks)
    assert not pending
    assert len(done) == len(items)


async def test_many_gets_before_put() -> None:
    queue = RendezvousQueue[t.Any]()
    items = [object() for _ in range(100)]
    tasks = [asyncio.create_task(queue.get()) for _ in range(len(items))]
    await yield_to_other_tasks()
    for item in items:
        queue.put_nowait(item)
    done, pending = await asyncio.wait(tasks)
    received = {await i for i in done}
    assert not pending
    assert received == set(items)


async def test_close_after_get() -> None:
    queue = RendezvousQueue[t.Any]()
    tasks = [asyncio.create_task(queue.get() if i else queue.put(i)) for i in range(4)]
    await yield_to_other_tasks()
    queue.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert queue.empty()
    assert queue.full()
    assert results[:2] == [None, 0]
    assert all(isinstance(exc, asyncio.CancelledError) for exc in results[2:])


async def test_close_after_put() -> None:
    queue = RendezvousQueue[t.Any]()
    tasks = [asyncio.create_task(queue.put(i) if i else queue.get()) for i in range(4)]
    await yield_to_other_tasks()
    queue.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert queue.empty()
    assert queue.full()
    assert results[:2] == [1, None]
    assert all(isinstance(exc, asyncio.CancelledError) for exc in results[2:])


async def test_state_after_close() -> None:
    queue = RendezvousQueue[t.Any]()
    queue.close()
    assert queue.empty()
    assert queue.full()
    with pytest.raises(asyncio.CancelledError):
        queue.get_nowait()
    with pytest.raises(asyncio.CancelledError):
        queue.put_nowait(None)
    with pytest.raises(asyncio.CancelledError):
        await queue.get()
    with pytest.raises(asyncio.CancelledError):
        await queue.put(None)
    queue.close()


@pytest.mark.slow()
async def test_free_for_all() -> None:
    queue = RendezvousQueue[t.Any]()

    async def putter(item: object) -> None:
        await asyncio.sleep(random.random())
        await queue.put(item)

    async def getter() -> object:
        await asyncio.sleep(random.random())
        return await queue.get()

    async def cancelled_getter() -> object:
        task = asyncio.create_task(queue.get())
        await yield_to_other_tasks()
        task.cancel()
        return await task

    async def cancelled_putter(item: object) -> None:
        task = asyncio.create_task(queue.put(item))
        await yield_to_other_tasks()
        task.cancel()
        await task

    items = frozenset(range(10000))
    tasks = [asyncio.create_task(getter()) for _ in items]
    tasks.extend(asyncio.create_task(putter(i)) for i in items)
    tasks.extend(asyncio.create_task(cancelled_getter()) for _ in items)
    tasks.extend(asyncio.create_task(cancelled_putter(i)) for i in items)
    random.shuffle(tasks)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    none_count = sum(1 for res in results if res is None)
    cancel_count = sum(1 for res in results if isinstance(res, asyncio.CancelledError))
    received = frozenset(res for res in results if isinstance(res, int))
    assert none_count == len(items)
    assert cancel_count == 2 * len(items)
    assert items == received
