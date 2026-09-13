"""Standalone offline DLSS 5 Neural Rendering entry point.

The sibling `dlss5_offline` module is the engine; this is the command line the
UI's offline DLSS5 page spawns, and it is the same shape as
`superres_offline`: pin `--engine dlss5` and hand the rest to `offline.convert`.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline.convert import main as convert_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("single or batch command required")
    return convert_main([args[0], "--engine", "dlss5", *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
