"""Tests for `RendezVousQueue`."""

import asyncio
import random
import typing as t
from contextlib import contextmanager

import pytest

from cernml.evert.rendezvous import QueueEmpty, QueueFull, RendezVousQueue


@contextmanager
def autocancel_task(
    coro: t.Coroutine, name: t.Optional[str] = None
) -> t.Iterator[asyncio.Task]:
    try:
        task = asyncio.create_task(coro, name=name)
        yield task
    finally:
        task.cancel()


async def test_default_is_empty_and_full() -> None:
    queue = RendezVousQueue[t.Any]()
    assert queue.empty() and queue.full()


async def test_empty_after_get() -> None:
    queue = RendezVousQueue[t.Any]()
    with autocancel_task(queue.get()):
        await asyncio.sleep(0)
        assert queue.empty()


async def test_not_empty_after_put() -> None:
    queue = RendezVousQueue[t.Any]()
    with autocancel_task(queue.put(object())):
        await asyncio.sleep(0)
        assert not queue.empty()


async def test_not_full_after_get() -> None:
    queue = RendezVousQueue[t.Any]()
    with autocancel_task(queue.get()):
        await asyncio.sleep(0)
        assert not queue.full()


async def test_full_after_put() -> None:
    queue = RendezVousQueue[t.Any]()
    with autocancel_task(queue.put(object())):
        await asyncio.sleep(0)
        assert queue.full()


async def test_get_nowait() -> None:
    queue = RendezVousQueue[t.Any]()
    sent = object()
    task = asyncio.create_task(queue.put(sent))
    with pytest.raises(QueueEmpty):
        queue.get_nowait()
    await asyncio.sleep(0)
    assert not task.done()
    received = queue.get_nowait()
    await task
    assert sent is received
    assert task.done()


async def test_put_nowait() -> None:
    queue = RendezVousQueue[t.Any]()
    sent = object()
    task = asyncio.create_task(queue.get())
    with pytest.raises(QueueFull):
        queue.put_nowait(sent)
    await asyncio.sleep(0)
    assert not task.done()
    queue.put_nowait(sent)
    received = await task
    assert sent is received
    assert task.done()


async def test_put_get() -> None:
    queue = RendezVousQueue[t.Any]()
    sent = object()
    received, none = await asyncio.gather(queue.get(), queue.put(sent))
    assert none is None
    assert sent is received


async def test_get_after_cancelled_put() -> None:
    queue = RendezVousQueue[t.Any]()
    putter = asyncio.create_task(queue.put(False))
    await asyncio.sleep(0)
    putter.cancel()
    assert not putter.done() and not queue.empty()
    asyncio.create_task(queue.put(True))
    assert await queue.get()


async def test_put_after_cancelled_get() -> None:
    queue = RendezVousQueue[t.Any]()
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    getter.cancel()
    assert not getter.done() and not queue.full()
    asyncio.create_task(queue.get())
    await queue.put(True)


async def test_many_puts_before_get() -> None:
    queue = RendezVousQueue[t.Any]()
    items = [object() for _ in range(100)]
    tasks = [asyncio.create_task(queue.put(i)) for i in items]
    await asyncio.sleep(0)
    received = [queue.get_nowait() for _ in range(len(items))]
    assert items == received
    done, pending = await asyncio.wait(tasks)
    assert not pending and len(done) == len(items)


async def test_many_gets_before_put() -> None:
    queue = RendezVousQueue[t.Any]()
    items = [object() for _ in range(100)]
    tasks = [asyncio.create_task(queue.get()) for _ in range(len(items))]
    await asyncio.sleep(0)
    for item in items:
        queue.put_nowait(item)
    done, pending = await asyncio.wait(tasks)
    received = {await i for i in done}
    assert not pending and received == set(items)


async def test_close_after_get() -> None:
    queue = RendezVousQueue[t.Any]()
    tasks = [asyncio.create_task(queue.get() if i else queue.put(i)) for i in range(4)]
    await asyncio.sleep(0)
    queue.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert queue.empty() and queue.full()
    assert results[:2] == [None, 0]
    assert all(isinstance(exc, asyncio.CancelledError) for exc in results[2:])


async def test_close_after_put() -> None:
    queue = RendezVousQueue[t.Any]()
    tasks = [asyncio.create_task(queue.put(i) if i else queue.get()) for i in range(4)]
    await asyncio.sleep(0)
    queue.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert queue.empty() and queue.full()
    assert results[:2] == [1, None]
    assert all(isinstance(exc, asyncio.CancelledError) for exc in results[2:])


async def test_state_after_close() -> None:
    queue = RendezVousQueue[t.Any]()
    queue.close()
    assert queue.empty() and queue.full()
    with pytest.raises(asyncio.CancelledError):
        queue.get_nowait()
    with pytest.raises(asyncio.CancelledError):
        queue.put_nowait(None)
    with pytest.raises(asyncio.CancelledError):
        await queue.get()
    with pytest.raises(asyncio.CancelledError):
        await queue.put(None)
    queue.close()


@pytest.mark.slow
async def test_free_for_all() -> None:
    queue = RendezVousQueue[t.Any]()

    async def putter(item: object) -> None:
        await asyncio.sleep(random.random())
        await queue.put(item)

    async def getter() -> object:
        await asyncio.sleep(random.random())
        return await queue.get()

    async def cancelled_getter() -> object:
        task = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        task.cancel()
        return await task

    async def cancelled_putter(item: object) -> None:
        task = asyncio.create_task(queue.put(item))
        await asyncio.sleep(0)
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
