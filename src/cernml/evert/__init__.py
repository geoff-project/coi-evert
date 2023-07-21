import logging
import queue
import typing as t
import warnings
from threading import Thread

import gym
import numpy as np
import scipy.optimize

Params = np.ndarray
Loss = np.floating

Objective = t.Callable[[Params], Loss]
SolveFunc = t.Callable[[Objective, Params], Params]


SendT = t.TypeVar("SendT")
RecvT = t.TypeVar("RecvT")


class CancelledError(BaseException):
    pass


class _ChannelSentinel:
    __slots__ = ()


class ChannelSide(t.Generic[SendT, RecvT]):
    __slots__ = ("_send", "_recv")

    def __init__(
        self,
        send: queue.SimpleQueue[t.Union[SendT, _ChannelSentinel]],
        recv: queue.SimpleQueue[t.Union[RecvT, _ChannelSentinel]],
    ) -> None:
        self._send: t.Optional[
            queue.SimpleQueue[t.Union[SendT, _ChannelSentinel]]
        ] = send
        self._recv: t.Optional[
            queue.SimpleQueue[t.Union[RecvT, _ChannelSentinel]]
        ] = recv

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._send:
            self._send.put(_ChannelSentinel())
            self._send = None

    def empty(self) -> bool:
        return self._recv.empty() if self._recv else True

    def qsize(self) -> int:
        return self._recv.qsize() if self._recv else 0

    def get(self, block: bool = True, timeout: t.Optional[float] = None) -> RecvT:
        if not self._recv:
            raise CancelledError()
        item = self._recv.get(block, timeout)
        if isinstance(item, _ChannelSentinel):
            self._recv = None
            raise CancelledError()
        return item

    def get_nowait(self) -> RecvT:
        if not self._recv:
            raise CancelledError()
        item = self._recv.get_nowait()
        if isinstance(item, _ChannelSentinel):
            self._recv = None
            raise CancelledError()
        return item

    def put(
        self, item: SendT, block: bool = True, timeout: t.Optional[float] = None
    ) -> None:
        if not self._send:
            raise CancelledError()
        return self._send.put(item, block, timeout)

    def put_nowait(self, item: SendT) -> None:
        if not self._send:
            raise CancelledError()
        return self._send.put_nowait(item)


def channel() -> t.Tuple[ChannelSide[SendT, RecvT], ChannelSide[RecvT, SendT]]:
    forward: queue.SimpleQueue[t.Union[SendT, _ChannelSentinel]] = queue.SimpleQueue()
    backward: queue.SimpleQueue[t.Union[RecvT, _ChannelSentinel]] = queue.SimpleQueue()
    return ChannelSide(forward, backward), ChannelSide(backward, forward)


class Cobyla:
    def __init__(self, maxfun: int = 100, ftol: float = 1e-4) -> None:
        self.maxfun = maxfun
        self.ftol = ftol

    def make_solve_func(
        self, bounds: tuple[Params, Params], constraints: list
    ) -> SolveFunc:
        warnings.warn(f"ignoring constraints: {constraints!r}")
        lb, ub = bounds

        def solve(objective: Objective, x0: Params) -> Params:
            res = scipy.optimize.minimize(
                objective,
                x0,
                bounds=scipy.optimize.Bounds(lb, ub),
                method="Powell",
                options={"maxiter": self.maxfun, "ftol": self.ftol},
            )
            return res.x

        return solve

    def make_solve_func_from_env(self, env: "ParabolaEnv") -> SolveFunc:
        bounds = (env.optimization_space.low, env.optimization_space.high)
        constraints = list(env.constraints)
        return self.make_solve_func(bounds, constraints)


class ParabolaEnv:
    def __init__(self) -> None:
        self.optimization_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=float)
        self.constraints: list[scipy.optimize.LinearConstraint] = []

    def get_initial_params(self) -> Params:
        return self.optimization_space.sample()

    def compute_single_objective(self, params: Params) -> Loss:
        return np.linalg.norm(params)


class SolveThread(Thread):
    def __init__(
        self, solve: SolveFunc, x0: Params, recv: ChannelSide[Params, Loss]
    ) -> None:
        super().__init__(target=self._solve_wrapper, args=(solve, x0), daemon=True)
        self._channel = recv
        self._logger = logging.getLogger(__name__ + ".SolveThread")

    def _solve_wrapper(self, solve: SolveFunc, x0: Params) -> None:
        self._logger.info("Start optimization routine")
        res = solve(self._objective, x0)
        self._logger.info("Optimization routine finished!")
        self._logger.info("Put result: %s", res)
        self._channel.put(res)
        self._channel.close()

    def _objective(self, params: Params) -> Loss:
        while True:
            self._logger.info("Put params: %s", params)
            self._channel.put(params)
            self._logger.info("Wait for loss …")
            loss = self._channel.get()
            self._logger.info("Get loss: %s", loss)
            if not np.isnan(loss):
                return loss
            self._logger.warning("NaN loss, ignoring")


class Eversion:
    def __init__(self, solve: SolveFunc, x0: Params) -> None:
        sender: ChannelSide[Loss, Params]
        receiver: ChannelSide[Params, Loss]
        sender, receiver = channel()
        self._thread = SolveThread(solve, x0, receiver)
        self._thread.start()
        self._channel = sender
        self._prev_params: t.Optional[Params] = None
        self._logger = logging.getLogger(__name__ + ".Eversion")

    def cancel(self) -> None:
        try:
            self._logger.info("Canceling …")
            self._channel.close()
        except CancelledError:
            self._logger.info("… already cancelled!")
        except CancelledError:
            self._logger.info("… already cancelled!")

    def join(self) -> None:
        self.cancel()
        self._logger.info("Joining thread …")
        self._thread.join()

    def start_coroutine(self) -> t.Generator[Params, Loss, Params]:
        i = 0
        while self._thread.is_alive():
            i += 1
            self._logger.info("Iteration #%d", i)
            try:
                params = self.ask()
            except CancelledError:
                break
            self._logger.info("Yielding …")
            loss = yield params
            self.tell(loss)
        res = params
        self._logger.info("Joining thread …")
        self._thread.join()
        self._logger.info("Result: %s", res)
        return res

    def ask(self) -> Params:
        try:
            self._logger.info("Wait for params …")
            self._prev_params = self._channel.get()
        except CancelledError:
            self._logger.info("Final params: %s", self._prev_params)
            raise StopIteration(self._prev_params.copy()) from None
        self._logger.info("Get params: %s", self._prev_params)
        return self._prev_params.copy()

    def tell(self, loss: Loss) -> None:
        self._logger.info("Put loss: %s", loss)
        self._channel.put(loss)


def evert(solve: SolveFunc, x0: Params) -> Eversion:
    return Eversion(solve, x0)


def run_normal(opt: Cobyla, env: ParabolaEnv) -> Params:
    solve = opt.make_solve_func_from_env(env)
    return solve(env.compute_single_objective, env.get_initial_params())


def run_everted(opt: Cobyla, env: ParabolaEnv) -> Params:
    solve = opt.make_solve_func_from_env(env)
    eversion = evert(solve, env.get_initial_params())
    try:
        while True:
            params = eversion.ask()
            eversion.tell(env.compute_single_objective(params))
    except StopIteration as exc:
        [res] = exc.args
        return res


def main() -> None:
    logging.basicConfig(level="INFO")
    opt = Cobyla()
    env = ParabolaEnv()
    optimum = run_everted(opt, env)
    print(
        f"f({', '.join(format(f, '.3g') for f in optimum)}) =",
        f"{env.compute_single_objective(optimum):.3g}",
    )


if __name__ == "__main__":
    main()
