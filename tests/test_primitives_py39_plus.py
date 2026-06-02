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
import threading
import unittest

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Scenario


if 'n' in inspect.signature(threading.Semaphore.release).parameters:

    class TestSemaphoreReleaseN(unittest.TestCase):
        """Semaphore.release(n), on runtimes whose stdlib supports it."""

        def test_semaphore_release_n_main_thread_and_raw(self):
            scenario = Scenario()
            for factory_name in ('Semaphore', 'BoundedSemaphore'):
                with self.subTest(primitive=factory_name):
                    sem = getattr(scenario, factory_name)(3)
                    raw = scenario.raw(sem)
                    for _ in range(3):
                        self.assertTrue(sem.acquire(blocking=False))
                    self.assertFalse(sem.acquire(blocking=False))

                    sem.release(2)
                    self.assertTrue(sem.acquire(blocking=False))
                    self.assertTrue(sem.acquire(blocking=False))
                    self.assertFalse(sem.acquire(blocking=False))

                    raw.release(n=3)
                    for _ in range(3):
                        self.assertTrue(sem.acquire(blocking=False))
                    self.assertFalse(sem.acquire(blocking=False))

        def test_semaphore_release_n_transaction_api(self):
            scenario = Scenario()
            sem = scenario.Semaphore(0)

            def releaser():
                sem.release(3)

            with scenario:
                t = scenario.thread(releaser)
                scenario.wait(sem.release)
                tx = scenario.transaction(t)
                self.assertEqual(tx.n, 3)
                scenario.skip(t, sem.release)

            for _ in range(3):
                self.assertTrue(sem.acquire(blocking=False))
            self.assertFalse(sem.acquire(blocking=False))

        def test_bounded_semaphore_release_n_transaction_api(self):
            scenario = Scenario()
            sem = scenario.BoundedSemaphore(2)
            sem.acquire(); sem.acquire()

            def releaser():
                sem.release(n=2)

            with scenario:
                t = scenario.thread(releaser)
                scenario.wait(sem.release)
                tx = scenario.transaction(t)
                self.assertEqual(tx.n, 2)
                scenario.skip(t, sem.release)

            self.assertTrue(sem.acquire(blocking=False))
            self.assertTrue(sem.acquire(blocking=False))
            self.assertFalse(sem.acquire(blocking=False))

        def test_bounded_semaphore_release_n_overrelease_is_atomic(self):
            scenario = Scenario()
            sem = scenario.BoundedSemaphore(1)

            with self.assertRaises(ValueError):
                sem.release(2)

            # The failed over-release must not have changed the semaphore.
            self.assertTrue(sem.acquire(blocking=False))
            self.assertFalse(sem.acquire(blocking=False))


def run_tests():
    blankettestlib.run('blanket.primitives.py39_plus', __name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
