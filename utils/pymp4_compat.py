"""Compat shim so ``pymp4`` works on Python 3.10+.

``pymp4`` (1.4.0) hard-pins ``construct==2.8.8`` (2016). construct 2.8.8 still
references ``collections.Sequence`` / ``collections.Mapping`` etc., which were
moved to ``collections.abc`` and removed from the top-level ``collections``
module in Python 3.10. Importing ``pymp4`` and *parsing* mostly works without
the shim, but ``Box.build`` (serialisation, which the SI progressive virtual
remux needs to emit a constructed ``moov``) raises::

    AttributeError: module 'collections' has no attribute 'Sequence'

Importing this module first installs the missing aliases, then re-exports
``Box``. Always import pymp4 through here:

    from utils.pymp4_compat import Box
"""
from __future__ import annotations

import collections
import collections.abc as _abc

for _name in (
    "Sequence",
    "MutableSequence",
    "Mapping",
    "MutableMapping",
    "Callable",
    "Iterable",
    "Hashable",
    "Set",
    "MutableSet",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))

from pymp4.parser import Box  # noqa: E402  (must follow the shim above)

__all__ = ["Box"]
