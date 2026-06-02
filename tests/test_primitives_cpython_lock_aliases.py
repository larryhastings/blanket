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

import threading
import unittest

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Scenario


if hasattr(threading.Lock(), 'acquire_lock'):
    class TestCPythonLockLegacyAliases(unittest.TestCase):
        """CPython Lock legacy aliases, on runtimes that expose them."""

        def test_legacy_aliases_exist_and_work_raw(self):
            """Lock mirrors the stdlib's legacy acquire_lock aliases."""
            scenario = Scenario()
            lock = scenario.Lock()
            raw = scenario.raw(lock)

            for obj in (lock, raw):
                self.assertTrue(obj.acquire_lock())
                self.assertTrue(obj.locked_lock())
                obj.release_lock()
                self.assertFalse(obj.locked_lock())

        def test_legacy_aliases_are_regulated_and_normalized(self):
            """Legacy Lock aliases enter the same scheduler method family."""
            scenario = Scenario()
            lock = scenario.Lock()
            results = []

            def worker():
                lock.acquire_lock()
                results.append('acquired')
                lock.release_lock()
                results.append('released')

            with scenario:
                t = scenario.thread(worker)
                # Either spelling sees the same normalized acquire tx.
                self.assertIn(lock.acquire_lock, scenario.wait(lock.acquire_lock))
                tx = scenario.transaction(t)
                self.assertEqual(tx.method, lock.acquire)
                scenario.skip(t, lock.acquire_lock)

                self.assertIn(lock.release_lock, scenario.wait(lock.release_lock))
                tx = scenario.transaction(t)
                self.assertEqual(tx.method, lock.release)
                scenario.skip(t, lock.release_lock)

            self.assertEqual(results, ['acquired', 'released'])

        def test_locked_lock_alias_normalizes_to_locked(self):
            scenario = Scenario()
            lock = scenario.Lock()

            def worker():
                lock.locked_lock()

            with scenario:
                t = scenario.thread(worker)
                self.assertIn(lock.locked_lock, scenario.wait(lock.locked_lock))
                tx = scenario.transaction(t)
                self.assertEqual(tx.method, lock.locked)
                scenario.skip(t, lock.locked_lock)


def run_tests():
    blankettestlib.run('blanket.primitives.cpython_lock_aliases', __name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
