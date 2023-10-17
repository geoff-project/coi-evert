#!/usr/bin/env python3

import logging

import gym.spaces
import numpy as np

from cernml import coi
from cernml.optimizers.scipy import Cobyla
from cernml.evert import Loss, Objective, Params, evert


class ParabolaEnv(coi.SingleOptimizable):
    def __init__(self) -> None:
        self.optimization_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=float)

    def get_initial_params(self) -> Params:
        return self.optimization_space.sample()

    def compute_single_objective(self, params: Params) -> Loss:
        return float(np.linalg.norm(params))


def run_normal(opt: Cobyla, env: ParabolaEnv) -> Params:
    bounds = env.optimization_space.low, env.optimization_space.high
    solve = opt.make_solve_func(bounds, env.constraints)
    result = solve(env.compute_single_objective, env.get_initial_params())
    return result.x


def run_everted(opt: Cobyla, env: ParabolaEnv) -> Params:
    bounds = env.optimization_space.low, env.optimization_space.high
    solve = opt.make_solve_func(bounds, env.constraints)

    def _solve_wrapper(obj: Objective, x_0: Params) -> Params:
        result = solve(obj, x_0)
        return result.x

    eversion = evert(_solve_wrapper, env.get_initial_params())
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
