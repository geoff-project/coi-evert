"""Tests for `evert()`."""

import typing as t
from unittest import mock

import pytest

from cernml.evert.synch import CancelledError, MethodOrderError, OptFinished, evert


class MockException(Exception):
    """Exception raised specifically for testing."""


def test_context_manager() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    with evert(solve, x0) as eversion:
        pass
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert eversion.join() == solve.return_value


def test_context_manager_cancelled() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    with evert(solve, x0) as eversion:
        eversion.cancel()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert eversion.join() == solve.return_value


def test_empty() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    with pytest.raises(CancelledError, match=r"solve\(\) never started"):
        eversion.join()
    # Then:
    solve.assert_not_called()


def test_once() -> None:
    def solve_side_effect(obj: t.Callable[[...], t.Any], x0: t.Any) -> t.Any:
        obj(x0)
        return mock.DEFAULT

    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = solve_side_effect
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    gen = eversion.as_generator()
    assert gen.send(None) == x0
    with pytest.raises(StopIteration) as exc_info:
        gen.send(mock.Mock(name="loss"))
    result = eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert result == solve.return_value
    assert result == exc_info.value.value


def test_ask_twice() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    assert eversion.ask() == x0
    # Then:
    with pytest.raises(MethodOrderError, match="^called ask().*when receiving"):
        eversion.ask()


def test_tell_twice() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: (obj(x0), obj(x0))
    x0 = mock.Mock(name="x0")
    # When:
    eversion = evert(solve, x0)
    assert eversion.ask() == x0
    eversion.tell(0)
    # Then:
    with pytest.raises(MethodOrderError, match="^called tell().*when sending"):
        eversion.tell(0)


def test_tell_after_success() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    loss = mock.Mock(name="loss")
    # When:
    eversion = evert(solve, x0)
    assert eversion.ask() == x0
    eversion.tell(loss)
    with pytest.raises(OptFinished) as exc_info:
        eversion.ask()
    assert exc_info.value.result == loss
    with pytest.raises(OptFinished) as exc_info:
        eversion.tell(mock.Mock(name="wrong loss"))
    assert exc_info.value.result == loss


def test_set_logger() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    logger = mock.Mock(name="logger")
    # When:
    eversion = evert(solve, x0)
    eversion.set_logger(logger)
    # Then:
    # pylint: disable = protected-access
    assert eversion._logger == logger
    assert eversion._inner._logger == logger


def test_exit_catches_cancelled_error() -> None:
    def solve_side_effect(obj: t.Callable[[...], t.Any], x0: t.Any) -> t.Any:
        obj(x0)
        return mock.DEFAULT

    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = solve_side_effect
    x0 = mock.Mock(name="x0")
    # When:
    with evert(solve, x0) as eversion:
        pass
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    with pytest.raises(CancelledError, match="^$"):
        eversion.join()


def test_ask_reraises_exception() -> None:
    # Given:
    error = MockException()
    solve = mock.Mock(name="solve")
    solve.side_effect = error
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(MockException) as exc_info, evert(solve, x0) as eversion:
        eversion.ask()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert exc_info.value is error


def test_exit_logs_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    # Given:
    bg_error = MockException("bg_error")
    fg_error = MockException("fg_error")
    solve = mock.Mock(name="solve")
    solve.side_effect = bg_error
    x0 = mock.Mock(name="x0")
    # When:
    caplog.set_level("ERROR")
    with pytest.raises(MockException) as exc_info, evert(solve, x0):
        raise fg_error
    # Then:
    assert exc_info.value is fg_error
    bg_record, fg_record = caplog.records
    assert bg_record.name == "cernml.evert.asynch.SolveThread"
    assert bg_record.getMessage() == "exiting due to exception"
    assert bg_record.exc_info
    assert bg_record.exc_info[1] is bg_error
    assert fg_record.name == "cernml.evert.synch.Eversion"
    assert fg_record.getMessage() == "an exception occurred in the background thread"
    assert fg_record.exc_info
    assert fg_record.exc_info[1] is bg_error
    assert fg_record.exc_info[1].__suppress_context__


def test_exit_reraises_bg_exception() -> None:
    # Given:
    ctx_error = MockException("ctx_error")
    bg_error = MockException("bg_error")
    bg_error.__context__ = ctx_error
    solve = mock.Mock(name="solve")
    solve.side_effect = bg_error
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(MockException) as exc_info, evert(solve, x0):
        pass
    # Then:
    assert exc_info.value is bg_error
    assert exc_info.value.__context__ is ctx_error
    assert not exc_info.value.__suppress_context__


def test_exit_does_not_suppress_cancelled_error() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    solve.side_effect = lambda obj, x0: obj(x0)
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(CancelledError), evert(solve, x0) as eversion:
        eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)


def test_exit_suppresses_opt_finished() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    with evert(solve, x0) as eversion:
        eversion.join()
    # Then:
    solve.assert_called_once_with(mock.ANY, x0)
    assert eversion.join() == solve.return_value


def test_exit_suppresses_only_own_opt_finished() -> None:
    # Given:
    solve = mock.Mock(name="solve")
    x0 = mock.Mock(name="x0")
    # When:
    with pytest.raises(OptFinished), evert(solve, x0):
        # Then:
        # `with evert` must not capture this exception:
        evert(solve, x0).ask()
