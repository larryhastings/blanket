#!/usr/bin/env python3

_license = """
blanket
Copyright 2025-2026 Larry Hastings
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import io
import pathlib
import unittest

stats = {
    "failures": 0,
    "errors": 0,
    "expected failures": 0,
    "unexpected successes": 0,
    "skipped": 0,
    }

attribute_names = {
    "failures": "failures",
    "errors": "errors",
    "expected failures": "expectedFailures",
    "unexpected successes": "unexpectedSuccesses",
    "skipped": "skipped",
    }

def preload_local_blanket():
    """
    Pre-load the local "blanket" module, to preclude finding
    an already-installed one on the path.
    """
    import pathlib
    import sys

    argv_0 = pathlib.Path(sys.argv[0])
    blanket_dir = argv_0.resolve().parent
    while True:
        blanket_init = blanket_dir / "blanket" / "__init__.py"
        if blanket_init.is_file():
            break
        parent = blanket_dir.parent
        if parent == blanket_dir:
            raise FileNotFoundError(
                f"could not find local blanket package starting from {argv_0.resolve()!s}")
        blanket_dir = parent

    # this almost certainly *is* a git checkout
    # ... but that's not required, so don't assert it.
    # assert (blanket_dir / ".git" / "config").is_file()

    if blanket_dir not in sys.path:
        sys.path.insert(1, str(blanket_dir))

    import blanket
    assert blanket.__file__.startswith(str(blanket_dir))
    return blanket_dir


def run(name, module, permutations=None):
    if name:
        print(f"Testing {name}...")

    # this is a lot of work to suppress the "\nOK"!
    sio = io.StringIO()
    runner = unittest.TextTestRunner(stream=sio)
    if name:
        argv0 = name
    elif isinstance(module, str):
        argv0 = module
    else:
        argv0 = module.__name__
    t = unittest.main(module=module, exit=False, testRunner=runner, argv=[argv0])
    result = t.result
    for name in stats:
        value = getattr(result, attribute_names[name], ())
        stats[name] += len(value)

    for line in sio.getvalue().split("\n"):
        if not line.startswith("Ran"):
            print(line)
            continue

        if permutations:
            fields = line.split()
            tests = fields[2]
            assert tests in ("test", "tests"), f"expected fields[2] to be 'test' or 'tests', but fields={fields}"
            fields[2] = f"{tests}, with {permutations()} total permutations,"
            line = " ".join(fields)
        print(line)
        print()
        # prevent printing the  \n and OK
        break

def finish():
    if not (stats['failures'] or stats['errors']):
        result = "OK"
    else:
        result = "FAILED"

    fields = [f"{name}={value}" for name, value in stats.items() if value]
    if fields:
        addendum = ", ".join(fields)
        result = f"{result} ({addendum})"
    print(result)
