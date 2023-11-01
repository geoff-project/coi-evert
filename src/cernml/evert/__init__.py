"""A tool to turn looping functions inside out.

See the :doc:`guide` for installation instructions, motivation and
a brief usage example. A non-trivial example use-case is shown in the
:doc:`examples/index`.

For detailed documentation, refer to the :doc:`api/index`. The eversion
API comes in two flavors:

- synchronous and blocking in `cernml.evert.synch` for the interactive
  prompt and most traditional ecosystems;
- asynchronous and non-blocking in `cernml.evert.asynch` for ecosystems
  based on `asyncio`.

For convenience, the synchronous flavor is also exposed in the root
package ``cernml.evert``:

    >>> from cernml.evert import evert, OptFinished
    >>>
    >>> def solve(objective, x0):
    ...     ...
    ...
    >>> with evert(solve, "initial") as eversion:
    ...     try:
    ...         args = eversion.ask()
    ...         result = calculate(args)
    ...         eversion.tell(result)
    ...     except OptFinished as finished:
    ...         use_this = finished.result
"""

from . import asynch, synch
from ._types import MethodOrderError, Objective, OptFinished, OptResult, SolveFunc
from .synch import CancelledError, Eversion, evert

__all__ = [
    "CancelledError",
    "Eversion",
    "MethodOrderError",
    "Objective",
    "OptFinished",
    "OptResult",
    "SolveFunc",
    "asynch",
    "evert",
    "synch",
]
