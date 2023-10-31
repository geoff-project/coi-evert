"""Tests for async `evert()`."""

import asyncio
import typing as t
from unittest import mock

import pytest

from cernml.evert._async import evert, OptFinished, MethodOrderError


async def yield_to_other_tasks() -> None:
    """Give other tasks a chance to run.

    This is simply ``asyncio.sleep(0)``. It's purpose is to let other
    tasks progress in their state so that certain test invariants are
    met.
    """
    await asyncio.sleep(0)


async def test_context_manager() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    async with evert(solve, x0) as eversion:
        pass
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert await eversion.join() == solve.return_value


async def test_empty() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    result = await eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert result == solve.return_value


async def test_once() -> None:
    def solve_side_effect(obj: t.Callable[[...], t.Any], x0: t.Any) -> t.Any:
        obj(x0)
        return mock.DEFAULT

    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = solve_side_effect
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    agen = eversion.as_generator()
    assert await agen.asend(None) == x0
    with pytest.raises(StopAsyncIteration):
        await agen.asend(mock.Mock(name="loss"))
    result = await eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert result == solve.return_value


@pytest.mark.parametrize("wait", [False, True])
async def test_ask_twice(wait: bool) -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    assert await eversion.ask() == x0
    if wait:
        await asyncio.sleep(1)
    # Then:
    step = f"receiving {'params' if wait else 'loss'}"
    with pytest.raises(MethodOrderError, match=f"^called ask().*when {step}$"):
        await eversion.ask()


@pytest.mark.parametrize("wait", [False, True])
async def test_tell_twice(wait: bool) -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: (obj(x0), obj(x0))
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    assert await eversion.ask() == x0
    await eversion.tell(0)
    if wait:
        await asyncio.sleep(0.1)
    # Then:
    step = f"sending {'loss' if wait else 'params'}"
    with pytest.raises(MethodOrderError, match=f"^called tell().*{step}$"):
        await eversion.tell(0)


async def test_tell_after_success() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    loss = mock.Mock(name="loss")
    # When:
    eversion = evert(solve, x0)
    assert await eversion.ask() == x0
    await eversion.tell(loss)
    with pytest.raises(OptFinished) as exc_info:
        await eversion.ask()
    assert exc_info.value.result == loss
    with pytest.raises(OptFinished) as exc_info:
        await eversion.tell(mock.Mock(name="wrong loss"))
    assert exc_info.value.result == loss
