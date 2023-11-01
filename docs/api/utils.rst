..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Type Aliases and Exception Classes
==================================

.. currentmodule:: cernml.evert

This is a description of type aliases and exceptions used by both variants of
the eversion API. They are available at the package root (``cernml.evert``) as
well as both API-specific packages (`cernml.evert.synch` and
`cernml.evert.asynch`).

.. data:: SolveFunc
    :type: typing.TypeAlias

    :ref:`Type alias <type-aliases>` of
    `~typing.Callable`\ [[`Objective`\ [`Params`, `Loss`]], `OptResult`].

    This is the function to be everted. We expect that it contains
    a long-running synchronous loop that calls the `Objective` callback
    function repeatedly until some kind of condition is met and it returns. No
    assumptions are made about the type variables, except that they can be
    passed between threads.

.. data:: Objective
    :type: typing.TypeAlias

    :ref:`Type alias <type-aliases>` of
    `~typing.Callable`\ [[`Params`], `Loss`]

    This is the type of callback function that will be passed to `SolveFunc`
    during eversion. It may block indefinitely and it may raise exceptions at
    any point. The two exceptions to look out for in particular are:

    - `MethodOrderError` (which is not an `std:Exception`, but
      a `std:RuntimeError`) and

    - `std:concurrent.futures.CancelledError` (not to be mixed up with
      `std:asyncio.CancelledError`!).

    The first is raised if the user causes a deadlock by calling the eversion
    functions in the wrong order; the latter if the user decides to abandon the
    procedure for any reason.

.. autoexception:: OptFinished()

.. autoexception:: MethodOrderError()
