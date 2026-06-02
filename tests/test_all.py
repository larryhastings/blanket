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

import blankettestlib
import inspect
import queue
import sys
import threading


def test_modules():
    modules = [
        "test_release_metadata",
        "test_stdlib_fidelity",
        "test_injector",
    ]
    if sys.version_info >= (3, 11):
        modules.append("test_injector_py311_plus")
    modules.append("test_primitives")
    modules.append("test_primitives_internal_coverage")
    if hasattr(queue, "SimpleQueue"):
        modules.append("test_primitives_py37_plus")
    if 'n' in inspect.signature(threading.Semaphore.release).parameters:
        modules.append("test_primitives_py39_plus")
    if hasattr(threading.Lock(), "acquire_lock"):
        modules.append("test_primitives_cpython_lock_aliases")
    if hasattr(queue.Queue, "shutdown"):
        modules.append("test_primitives_py313_plus")
    modules.append("test_harness")
    return modules


def run_modules(modules):
    for test_module in modules:
        module = __import__(test_module)
        module.run_tests()


def main():
    blankettestlib.preload_local_blanket()
    run_modules(test_modules())
    blankettestlib.finish()


if __name__ == '__main__':
    main()
