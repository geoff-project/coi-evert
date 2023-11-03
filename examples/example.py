#!/usr/bin/env python3

"""Simple show-case of how evert() fits into the CernML ecosystem.

This creates a `cernml.coi.SingleOptimizable` and
a `cernml.optimizers.Optimizer`. It combines them into a single *solve*
function and then runs this function twice: once normally, once everted.
"""

import logging

import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from cernml import coi
from cernml.evert import OptFinished, evert
from cernml.optimizers import Optimizer, OptimizeResult, make, make_solve_func


class ParabolaEnv(coi.SingleOptimizable):
    """A trivial quadratic 2D function."""

    def __init__(self) -> None:
        self.optimization_space = Box(-1, 1, shape=(2,), dtype=np.double)

    def get_initial_params(self) -> NDArray[np.double]:
        return self.optimization_space.sample()

    def compute_single_objective(self, params: NDArray[np.double]) -> float:
        return float(np.linalg.norm(params))


def run_normal(opt: Optimizer, env: ParabolaEnv) -> OptimizeResult:
    """Solve the problem with normal control flow.

    We enter the solve function and do not leave it until it is
    completed. On every iteration, it calls into
    `compute_single_objective()` with new arguments.
    """
    solve = make_solve_func(opt, env)
    return solve(env.compute_single_objective, env.get_initial_params())


def run_everted(opt: Optimizer, env: ParabolaEnv) -> OptimizeResult:
    """Solve the problem with everted control flow.

    We never enter the solve function and instead keep the control flow
    to ourselves. Instead, the everted solve function acts like a state
    machine that we query for a new set of arguments and feed with the
    resulting cost value calculated via `compute_single_objective()`.

    Eventually, the state machine terminates by raising an `OptFinished`
    exception, from which we retrieve the result. Alternatively, we
    could've also used a `with` block and explicit ``eversion.join()``.
    """
    solve = make_solve_func(opt, env)
    try:
        eversion = evert(solve, env.get_initial_params())
        while True:
            params = eversion.ask()
            eversion.tell(env.compute_single_objective(params))
    except OptFinished as finished:
        return finished.result


def main() -> None:
    """Main function."""
    logging.basicConfig(level="DEBUG")
    opt = make("COBYLA")
    env = ParabolaEnv()
    for func in run_normal, run_everted:
        result = func(opt, env)
        print(
            f"f({', '.join(format(f, '.3g') for f in result.x)}) = {result.fun:.3g}",
        )


if __name__ == "__main__":
    main()
