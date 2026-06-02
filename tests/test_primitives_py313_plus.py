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
blankettestlib.preload_local_blanket()

import queue
import unittest

from blanket import Scenario, Call, State


if hasattr(queue.Queue, 'shutdown'):

    class TestQueueShutdown(unittest.TestCase):
        """Tests for Python versions whose stdlib Queue has shutdown()."""

        VARIANTS = ('Queue', 'LifoQueue', 'PriorityQueue')

        def test_shutdown_basic_behavior(self):
            s = Scenario()
            for name in self.VARIANTS:
                with self.subTest(variant=name):
                    q = getattr(s, name)()
                    q.put_nowait('x')
                    q.shutdown()
                    with self.assertRaises(queue.ShutDown):
                        q.put_nowait('y')
                    self.assertEqual(q.get_nowait(), 'x')
                    with self.assertRaises(queue.ShutDown):
                        q.get_nowait()

        def test_shutdown_immediate_and_transaction_api(self):
            s = Scenario()
            q = s.Queue()
            q.put_nowait('x')
            with s:
                t = s.thread(lambda: q.shutdown(immediate=True))
                s.wait(q.shutdown)
                tx = s.transaction(t)
                self.assertTrue(hasattr(s.api(q), 'shutdown'))
                self.assertTrue(tx.immediate)
                s.skip(t, q.shutdown)
            with self.assertRaises(queue.ShutDown):
                q.get_nowait()

        def test_shutdown_wakes_blocked_get_and_put(self):
            s = Scenario(); q = s.Queue(); results = []
            def getter():
                try:
                    q.get()
                except queue.ShutDown:
                    results.append('get_shutdown')

            with s:
                tg = s.thread(getter)
                s.wait(q.get)
                txg = s.transaction(tg)
                txg.unblock()
                s.wait(Call(tg, q.get, State.WAITING))
                ts = s.thread(lambda: q.shutdown())
                s.wait(q.shutdown)
                s.skip(ts, q.shutdown)
                s.wait(txg)
            self.assertEqual(results, ['get_shutdown'])

            s = Scenario(); q = s.Queue(maxsize=1); q.put_nowait('x'); results = []
            def putter():
                try:
                    q.put('y')
                except queue.ShutDown:
                    results.append('put_shutdown')

            with s:
                tp = s.thread(putter)
                s.wait(q.put)
                txp = s.transaction(tp)
                txp.unblock()
                s.wait(Call(tp, q.put, State.WAITING))
                ts = s.thread(lambda: q.shutdown())
                s.wait(q.shutdown)
                s.skip(ts, q.shutdown)
                s.wait(txp)
            self.assertEqual(results, ['put_shutdown'])

        def test_shutdown_immediate_wakes_blocked_join(self):
            s = Scenario(); q = s.Queue(); q.put_nowait('x'); results = []
            def joiner():
                q.join()
                results.append('joined')

            with s:
                tj = s.thread(joiner)
                s.wait(q.join)
                txj = s.transaction(tj)
                txj.unblock()
                s.wait(Call(tj, q.join, State.COMMIT))
                ts = s.thread(lambda: q.shutdown(immediate=True))
                s.wait(q.shutdown)
                s.skip(ts, q.shutdown)
                s.wait(txj)
            self.assertEqual(results, ['joined'])


def run_tests():
    blankettestlib.run(name="blanket.primitives.py313_plus", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
