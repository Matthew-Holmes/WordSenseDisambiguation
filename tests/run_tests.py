"""Minimal test runner (pytest is not installed in this project's venv).

Run from the repository root, in WSL:

    .venv/bin/python -m tests.run_tests                 # all fast tests
    .venv/bin/python -m tests.run_tests data evaluation # only modules whose name contains a filter
    WSD_SLOW_TESTS=1 .venv/bin/python -m tests.run_tests # also run slow tests (model parity)

Test modules are ordinary pytest-style files (`tests/test_*.py` with `test_*` functions), so
`pytest tests/` also works if pytest is installed later. A module that sets `SLOW = True` is
skipped unless WSD_SLOW_TESTS=1. Raise `unittest.SkipTest("reason")` to skip a single test.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
import time
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    filters = argv[1:]
    run_slow = os.environ.get("WSD_SLOW_TESTS") == "1"
    passed = failed = skipped = 0
    for info in sorted(pkgutil.iter_modules([str(Path(__file__).parent)]), key=lambda m: m.name):
        if not info.name.startswith("test_"):
            continue
        if filters and not any(f in info.name for f in filters):
            continue
        module = importlib.import_module(f"tests.{info.name}")
        if getattr(module, "SLOW", False) and not run_slow:
            print(f"SKIP  {info.name} (slow module; set WSD_SLOW_TESTS=1)")
            skipped += 1
            continue
        tests = [
            (name, fn)
            for name, fn in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and fn.__module__ == module.__name__
        ]
        tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
        for name, fn in tests:
            start = time.time()
            try:
                fn()
            except unittest.SkipTest as exc:
                skipped += 1
                print(f"SKIP  {info.name}.{name}: {exc}")
            except Exception:  # noqa: BLE001 - report every failure and keep going
                failed += 1
                print(f"FAIL  {info.name}.{name} ({time.time() - start:.1f}s)")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS  {info.name}.{name} ({time.time() - start:.1f}s)")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
