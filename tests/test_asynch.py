"""Tests for async `evert()`."""

import asyncio
import typing as t
from unittest import mock

import pytest

from cernml.evert.asynch import MethodOrderError, OptFinished, evert


class MockException(Exception):
    """Exception raised specifically for testing."""


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
    with pytest.raises(asyncio.CancelledError, match=r"solve\(\) never started"):
        await eversion.join()
    # Then:
    solve.assert_not_called()


async def test_double_cancel() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    eversion.cancel()
    eversion.cancel()
    # Then:
    solve.assert_not_called()


async def test_ask_after_cancel() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    eversion.cancel()
    with pytest.raises(asyncio.CancelledError, match=r"solve\(\) never started"):
        await eversion.ask()
    # Then:
    solve.assert_not_called()


async def test_once() -> None:
    def solve_side_effect(obj: t.Callable[..., t.Any], x0: t.Any) -> t.Any:
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


async def test_exit_lets_cancelled_error_pass() -> None:
    def solve_side_effect(obj: t.Callable[..., t.Any], x0: t.Any) -> t.Any:
        obj(x0)
        return mock.DEFAULT

    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = solve_side_effect
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(asyncio.CancelledError, match="^$"):
        async with evert(solve, x0):
            pass
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)


async def test_ask_reraises_exception() -> None:
    # Given:
    error = MockException()
    solve = mock.Mock(name="solve")
    solve.side_effect = error
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(MockException) as exc_info:
        async with evert(solve, x0) as eversion:
            await eversion.ask()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert exc_info.value is error


async def test_exit_logs_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    # Given:
    bg_error = MockException("bg_error")
    fg_error = MockException("fg_error")
    solve = mock.Mock(name="solve")
    solve.side_effect = bg_error
    x0 = mock.Mock(name="x0")
    # When:
    with caplog.at_level("ERROR"), pytest.raises(MockException) as exc_info:
        async with evert(solve, x0):
            raise fg_error
    # Then:
    assert exc_info.value is fg_error
    bg_record, fg_record = caplog.records
    assert bg_record.name == "cernml.evert.asynch.SolveThread"
    assert bg_record.getMessage() == "exiting due to exception"
    assert bg_record.exc_info
    assert bg_record.exc_info[1] is bg_error
    assert fg_record.name == "cernml.evert.asynch.Eversion"
    assert fg_record.getMessage() == "an exception occurred in the background thread"
    assert fg_record.exc_info
    assert fg_record.exc_info[1] is bg_error
    assert fg_record.exc_info[1].__suppress_context__


async def test_exit_reraises_bg_exception() -> None:
    # Given:
    ctx_error = MockException("ctx_error")
    bg_error = MockException("bg_error")
    bg_error.__context__ = ctx_error
    solve = mock.Mock(name="solve")
    solve.side_effect = bg_error
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(MockException) as exc_info:
        async with evert(solve, x0):
            pass
    # Then:
    assert exc_info.value is bg_error
    assert exc_info.value.__context__ is ctx_error
    assert not exc_info.value.__suppress_context__


async def test_exit_does_not_suppress_cancelled_error() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(asyncio.CancelledError):
        async with evert(solve, x0) as eversion:
            await eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)


async def test_exit_suppresses_opt_finished() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    async with evert(solve, x0) as eversion:
        await eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert (await eversion.join()) == solve.return_value


async def test_exit_suppresses_only_own_opt_finished() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(OptFinished):
        async with evert(solve, x0):
            # Then:
            # `with evert` must not capture this exception:
            await evert(solve, x0).ask()
