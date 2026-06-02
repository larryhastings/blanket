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

import inspect
import queue
import threading
import unittest

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Scenario


def _signature(obj, method):
    return inspect.signature(getattr(obj, method))


class TestCurrentStdlibFidelity(unittest.TestCase):
    """
    The regulated public objects should look like this interpreter's
    stdlib objects, not like some other Python's stdlib objects.
    """

    def setUp(self):
        self.scenario = Scenario()

    def assert_method_signature_matches(self, stdlib_obj, blanket_obj, method):
        self.assertEqual(
            hasattr(blanket_obj, method),
            hasattr(stdlib_obj, method),
            f"{type(blanket_obj).__name__}.{method} availability differs from {type(stdlib_obj).__name__}.{method}",
        )
        if not hasattr(stdlib_obj, method):
            return
        self.assertEqual(
            _signature(blanket_obj, method),
            _signature(stdlib_obj, method),
            f"{type(blanket_obj).__name__}.{method} signature differs from {type(stdlib_obj).__name__}.{method}",
        )

    def assert_cooked_and_raw_match(self, stdlib_obj, blanket_obj, methods):
        raw_obj = self.scenario.raw(blanket_obj)
        for method in methods:
            with self.subTest(obj=type(blanket_obj).__name__, method=method, handle="cooked"):
                self.assert_method_signature_matches(stdlib_obj, blanket_obj, method)
            with self.subTest(obj=type(blanket_obj).__name__, method=method, handle="raw"):
                self.assert_method_signature_matches(stdlib_obj, raw_obj, method)

    def test_threading_method_signatures_match_current_stdlib(self):
        cases = [
            (threading.Lock(), self.scenario.Lock(), [
                "acquire", "release", "locked",
                "acquire_lock", "release_lock", "locked_lock",
                ]),
            (threading.RLock(), self.scenario.RLock(), [
                "acquire", "release", "locked",
                ]),
            (threading.Condition(), self.scenario.Condition(), [
                "acquire", "release", "wait", "wait_for", "notify", "notify_all", "locked",
                ]),
            (threading.Semaphore(), self.scenario.Semaphore(), [
                "acquire", "release",
                ]),
            (threading.BoundedSemaphore(), self.scenario.BoundedSemaphore(), [
                "acquire", "release",
                ]),
            (threading.Event(), self.scenario.Event(), [
                "wait", "set", "clear", "is_set",
                ]),
            (threading.Barrier(1), self.scenario.Barrier(1), [
                "wait", "reset", "abort",
                ]),
            ]
        for stdlib_obj, blanket_obj, methods in cases:
            self.assert_cooked_and_raw_match(stdlib_obj, blanket_obj, methods)

    def test_queue_method_signatures_match_current_stdlib(self):
        queue_classes = ["Queue", "LifoQueue", "PriorityQueue"]
        if hasattr(queue, "SimpleQueue"):
            queue_classes.insert(0, "SimpleQueue")

        method_names = [
            "qsize", "empty", "full",
            "put", "put_nowait", "get", "get_nowait",
            "task_done", "join", "shutdown",
            ]
        for class_name in queue_classes:
            with self.subTest(class_name=class_name):
                stdlib_obj = getattr(queue, class_name)()
                blanket_obj = getattr(self.scenario, class_name)()
                self.assert_cooked_and_raw_match(stdlib_obj, blanket_obj, method_names)

    def test_threading_impersonator_matches_current_stdlib_availability(self):
        for name in [
            "Lock", "RLock", "Condition", "Semaphore", "BoundedSemaphore",
            "Event", "Barrier",
            ]:
            with self.subTest(name=name):
                self.assertEqual(
                    hasattr(self.scenario.threading, name),
                    hasattr(threading, name),
                )

    def test_queue_impersonator_matches_current_stdlib_availability(self):
        for name in ["SimpleQueue", "Queue", "LifoQueue", "PriorityQueue", "ShutDown"]:
            with self.subTest(name=name):
                self.assertEqual(
                    hasattr(self.scenario.queue, name),
                    hasattr(queue, name),
                )


def run_tests():
    blankettestlib.run(name="blanket.stdlib_fidelity", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
