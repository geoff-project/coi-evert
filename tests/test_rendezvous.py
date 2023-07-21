"""Tests for `RendezVousQueue`."""

# pylint: disable = missing-function-docstring

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


async def test_free_for_all() -> None:
    queue = RendezVousQueue[t.Any]()

    async def putter(item: object) -> None:
        await asyncio.sleep(random.random())
        await queue.put(item)

    async def getter() -> object:
        await asyncio.sleep(random.random())
        return await queue.get()

    items = frozenset(object() for _ in range(10000))
    tasks = [asyncio.create_task(getter()) for _ in items]
    tasks.extend(asyncio.create_task(putter(i)) for i in items)
    random.shuffle(tasks)

    results = await asyncio.gather(*tasks)
    received: t.FrozenSet[object] = frozenset(filter(None, results))
    assert items == received
