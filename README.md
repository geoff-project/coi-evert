<!--
SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
SPDX-FileNotice: All rights not expressly granted are reserved.

SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+
-->

Eversion – Turning Optimization Loops Inside Out
================================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

[CernML-COI][] defines common interfaces that facilitate using numerical
optimization and reinforcement learning (RL) on the same optimization problems.
This makes it possible to unify both approaches into a generic optimization
application in the CERN Control Center.

CernML-COI-Evert defines a function `evert()` that turns the control flow of an
optimization loop inside out. Instead of a `solve()` function that calls
a callback `loss = objective(params)` in a loop, you get an `Eversion` object
from which you get `params = eversion.ask()` and to which you give feedback via
`eversion.tell(loss)`. The API comes in a regular, blocking variant and an
asynchronous one. It is compatible and meant to be used with
[CernML-COI-Optimizers][] and [CernML-COI-Loops][].

This repository can be found online on CERN's [Gitlab][].

[Gitlab]: https://gitlab.cern.ch/geoff/cernml-coi-evert/
[CernML-COI]: https://gitlab.cern.ch/geoff/cernml-coi/
[CernML-COI-Loops]: https://gitlab.cern.ch/geoff/cernml-coi-loops/
[CernML-COI-Optimizers]: https://gitlab.cern.ch/geoff/cernml-coi-optimizers/

Table of Contents
-----------------

[[_TOC_]]

Motivation
----------

Many optimization routines are implemented as a function called *solve* that
takes a callback function *objective* and an initial argument *x0*. The *solve*
function then runs a loop, potentially for a long time, and calls *objective*
repeatedly until it has calculated a result.

```python
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
```

Because *solve* seizes the *control flow*, its caller is stuck waiting for
however long it takes and cannot do anything else. This is inconvenient if the
callback function is computationally expensive (it might need to do I/O) and
you're writing an application that is expected to remain responsive during this
time.

```python
>>> def objective(x):
...     # Imagine this function had to communicate with the
...     # outside world to calculate its result.
...     return 3 * x**2 - 6*x + 1
...
>>> # No intervention is possible while `solve()` runs!
>>> xbest = solve(objective, 0.0)
>>> np.isclose(xbest, 1.0, rtol=1e-3)
True
```

Installation
------------

To install this package from the [Acc-Py Repository][], run the following line
while on the CERN network:

```shell-session
$ pip install cernml-coi-evert
```

You can also install this package directly from Git:

```shell-session
$ pip install git+https://gitlab.cern.ch/geoff/cernml-coi-evert
```

Finally, you can also clone the repository first and then install it:

```shell-session
$ git clone https://gitlab.cern.ch/geoff/cernml-coi-evert
$ cd ./cernml-coi-evert/
$ pip install .
```

[Acc-Py Repository]: https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py

Quickstart
----------

If you have a package created with [`acc-py init`][], add this package to your
dependencies:

```python
REQUIREMENTS: dict = {
    'core': [
        'cernml-coi-evert ~= 1.0',
        ...
    ],
    ...
}
```

Use the registry APIs of the [COI][CernML-COI] and of the [COI
Optimizers][CernML-COI-Optimizers] to create optimizer and optimization
problem:

```python
# Run `pip install cern_awake_env` for this particular example.
import cern_awake_env
from cernml import coi, optimizers

env = coi.make("AwakeSimEnvH-v1")
opt = optimizers.make("BOBYQA")
```

Then combine the two into a *solve function* and pass it into an eversion
context. In this context, you can call `ask()` and `tell()` in a loop:

```python
from cernml.evert import evert

solve = optimizers.make_solve_func(opt, env)
with evert(solve, env.get_initial_params()) as eversion:
    while True:
        params = eversion.ask()
        loss = env.compute_single_objective(params)
        eversion.tell(loss)
```

When the optimization is finished, `ask()` will raise an exception that is
caught by the context. Once you have exited the context, you can proceed as
normal. Call `eversion.join()` to retrieve the optimization result:

```python
result = eversion.join()
print("Optimization finished")
print(f"f({result.x}) = {result.fun} after {result.niter} iterations")
```

Documentation
-------------

Inside the CERN network, you can read the package documentation on the [Acc-Py
documentation server][acc-py-docs]. The API is also documented via extensive
Python docstrings.

[acc-py-docs]: https://acc-py.web.cern.ch/gitlab/geoff/cernml-coi-evert/

Changelog
---------

[See here](https://acc-py.web.cern.ch/gitlab/geoff/cernml-coi-evert/docs/stable/changelog.html).

Stability
---------

This package uses [Semantic Versioning](https://semver.org/).

License
-------

Except as otherwise noted, this work is licensed under either of [GNU Public
License, Version 3.0 or later](LICENSES/GPL-3.0-or-later.txt), or [European
Union Public License, Version 1.2 or later](LICENSES/EUPL-1.2.txt), at your
option. See [COPYING](COPYING) for details.

Unless You explicitly state otherwise, any contribution intentionally submitted
by You for inclusion in this Work (the Covered Work) shall be dual-licensed as
above, without any additional terms or conditions.

For full authorship information, see the version control history.
