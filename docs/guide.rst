..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

User Guide
==========

.. currentmodule:: cernml.evert

This package provides `~synch.evert()`, a function that turns a long-running,
callback-based algorithm inside out so that you can progress through its
procedure without yielding control flow.

This works by moving the algorithm onto a background thread and providing
convenient methods `~synch.Eversion.ask()` and `~synch.Eversion.tell()` to
communicate with it.

Installation
------------

To install this package from the `Acc-Py Repository`_, run the following line
while on the CERN network:

.. _Acc-Py Repository:
   https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py

.. code-block:: shell-session

    $ pip install cernml-coi-evert

You can also install this package directly from Git:

.. code-block:: shell-session

    $ pip install git+https://gitlab.cern.ch/geoff/cernml-coi-evert

Finally, you can also clone the repository first and then install it:

.. code-block:: shell-session

    $ git clone https://gitlab.cern.ch/geoff/cernml-coi-evert
    $ cd ./cernml-coi-evert/
    $ pip install .

Motivation
----------

Many optimization routines are implemented as a function called *solve* that
takes a callback function *objective* and an initial argument *x0*. The *solve*
function then runs a loop, potentially for a long time, and calls *objective*
repeatedly until it has calculated a result.

    >>> import numpy as np
    >>> from functools import partial
    ...
    >>> def luus_jaakola(objective, x0, annealing, bounds, seed):
    ...     x0 = np.asanyarray(x0)
    ...     rng = np.random.default_rng(seed)
    ...     rate = 1.0
    ...     lower, upper = bounds
    ...     value = objective(x0)
    ...     while rate > 0.01:
    ...         max_step = min(abs(upper - x0), abs(x0 - lower))
    ...         step = rng.uniform(lower - x0, upper - x0, x0.shape)
    ...         x = x0 + rate * step
    ...         new_value = objective(x0 + rate * step)
    ...         if new_value < value:
    ...             x0 = x
    ...             value = new_value
    ...         else:
    ...             rate *= annealing
    ...     return x0
    ...
    >>> solve = partial(
    ...     luus_jaakola, annealing=0.98, seed=42, bounds=(-2, 2)
    ... )

Because *solve* seizes the *control flow*, its caller is stuck waiting for
however long it takes and cannot do anything else. This is inconvenient if the
callback function is computationally expensive (it might need to do I/O) and
you're writing an application that is expected to remain responsive during this
time.

    >>> def objective(x):
    ...     # Imagine this function had to communicate with the
    ...     # outside world to calculate its result.
    ...     return 3 * x**2 - 6*x + 1
    ...
    >>> # No intervention is possible while `solve()` runs!
    >>> xbest = solve(objective, 0.0)
    >>> np.isclose(xbest, 1.0, rtol=1e-3)
    True

Usage
-----

To evert_ such a function means to turn it *inside-out*: instead of it seizing
control and calling back to you, you maintain control and call into it whenever
you can and want to progress its internal state:

   >>> from cernml.evert import evert
   ...
   >>> eversion = evert(solve, 0.0)
   >>> x = eversion.ask()
   >>> x
   array(0.)

.. _evert: https://en.wiktionary.org/wiki/evert

Once *solve* is everted, we maintain control flow and can do whatever is
necessary to remain reactive. If there are any GUI events, we can process them.
We just need to remember to eventually tell the everted *solve* the return
value of its callback:

   >>> result = objective(x)
   >>> eversion.tell(result)

This will progress through *solve*'s internal procedure and make it ready to
produce the next value:

    >>> x = eversion.ask()
    >>> x
    1.0958241942238534

If you drive this cycle for long enough, `~synch.Eversion.ask()` will
eventually raise an exception instead of returning a value. This exception
contains the return value of *solve*:

    >>> from cernml.evert import OptFinished
    ...
    >>> try:
    ...     while True:
    ...         eversion.tell(objective(x))
    ...         x = eversion.ask()
    ... except OptFinished as finished:
    ...     xbest = finished.result
    >>> np.isclose(xbest, 1.0, rtol=1e-3)
    True

Resource Cleanup
----------------

In order to execute *solve*, `~synch.evert()` starts a background thread. You
need to ensure this thread is properly cleaned up. There are multiple ways to
do this.

1.  Catch `OptFinished` being raised from `~synch.Eversion.ask()`. As soon as
    `OptFinished` is raised, the thread is already cleaned up.

        >>> try:
        ...     while True:
        ...         args = eversion.ask()
        ...         eversion.tell(objective(args))
        ... except OptFinished as finished:
        ...     xbest = finished.result
        >>> np.isclose(xbest, 1.0, rtol=1e-3)
        True

2.  Call `~synch.Eversion.join()` to shut down all further communication and
    wait for the result of *solve*.

        >>> xbest = eversion.join()
        >>> np.isclose(xbest, 1.0, rtol=1e-3)
        True

    This is what `~synch.Eversion.ask()` does internally before raising
    `OptFinished`. If *solve* cannot actually complete yet,
    a `~asyncio.CancelledError` will
    be raised.

3.  Use the eversion as a :term:`context manager` that automatically joins the
    background thread upon exit:

        >>> with eversion:
        ...     while True:
        ...         args = eversion.ask()
        ...         eversion.tell(objective(args))
        >>> xbest = eversion.join()
        >>> np.isclose(xbest, 1.0, rtol=1e-3)
        True

    It automatically catches and suppresses the `OptFinished` exception. It
    also doesn't raise an exception if you exit the context before *solve* has
    completed:

        >>> with evert(solve, 0.0) as ev:
        ...     pass  # Forgetting to complete `solve`.
        >>> # But if you join afterwards, you get an exception:
        >>> ev.join()
        Traceback (most recent call last):
        ...
        asyncio...CancelledError

Asynchronous Eversion
---------------------

Eversion internally uses `asyncio` and its default executor to drive the call
to *solve*. Under `cernml.evert.asynch`, you can find an asynchronous variant
of the Eversion API. In it, all the blocking methods of the synchronous API
(which are also available under `cernml.evert.synch`) are replaced with
:term:`coroutine functions <coroutine function>`. Similarly, the eversion is
also an :term:`asynchronous context manager`:

    >>> import asyncio
    >>> from cernml.evert.asynch import evert
    ...
    >>> async def main():
    ...     async with evert(luus_jaakola, 0.0) as eversion:
    ...         args = await eversion.ask()
    ...         await eversion.tell(objective(args))
    ...     return await eversion.join()
    ...
    >>> xbest = asyncio.run(main())
    >>> isclose(xbest, 1.0, rtol=1e-3)
    True

This makes it possible to use the same event loop for the eversion as for any
sort of I/O or GUI processing that you need to do.
