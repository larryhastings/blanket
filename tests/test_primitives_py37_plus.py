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

from blanket import Scenario, State


if hasattr(queue, 'SimpleQueue'):

    class TestSimpleQueue(unittest.TestCase):
        """Tests for the SimpleQueue primitive."""

        def test_basic_put_get_main_thread(self):
            s = Scenario(); q = s.SimpleQueue()
            self.assertTrue(q.empty())
            self.assertEqual(q.qsize(), 0)
            q.put('a'); q.put('b')
            self.assertFalse(q.empty())
            self.assertEqual(q.qsize(), 2)
            self.assertEqual(q.get(), 'a')
            self.assertEqual(q.get(), 'b')
            self.assertTrue(q.empty())

        def test_put_nowait_and_get_nowait(self):
            s = Scenario(); q = s.SimpleQueue()
            q.put_nowait('x')
            self.assertEqual(q.get_nowait(), 'x')
            with self.assertRaises(queue.Empty):
                q.get_nowait()

        def test_raw_handle(self):
            s = Scenario(); q = s.SimpleQueue()
            raw = s.raws[q]
            raw.put('z')
            self.assertEqual(raw.qsize(), 1)
            self.assertEqual(raw.get(), 'z')
            self.assertTrue(raw.empty())

        def test_api_alias_and_isinstance(self):
            s = Scenario(); q = s.SimpleQueue()
            self.assertTrue(hasattr(Scenario, 'SimpleQueueAPI'))
            self.assertIs(Scenario.SimpleQueueAPI, s.SimpleQueueAPI)
            self.assertIsInstance(s.api(q), Scenario.SimpleQueueAPI)

        def test_fancy_reprs_and_api_qsize(self):
            s = Scenario(); q = s.SimpleQueue()
            q.name = 'named-simple'
            self.assertIn('SimpleQueue', repr(q))
            self.assertIn('SimpleQueueCore', repr(q._core))
            self.assertIn('SimpleQueueAPI', repr(s.api(q)))
            self.assertIn('SimpleQueue.raw', repr(s.raw(q)))
            q.put('x')
            self.assertEqual(q.qsize(), 1)
            self.assertEqual(q.get(), 'x')

        def test_raw_put_nowait_get_nowait_and_repr(self):
            s = Scenario(); q = s.SimpleQueue(); raw = s.raw(q)
            self.assertIn('SimpleQueue.raw', repr(raw))
            raw.put_nowait('n')
            self.assertEqual(raw.get_nowait(), 'n')

        def test_transaction_reprs(self):
            s = Scenario(); q = s.SimpleQueue()
            methods = [q.put, q.qsize, q.empty, q.get, q.put_nowait, q.get_nowait]

            def worker():
                q.put('a')
                q.qsize()
                q.empty()
                q.get()
                q.put_nowait('b')
                q.get_nowait()

            with s:
                t = s.thread(worker)
                for method in methods:
                    s.wait(method)
                    tx = s.transaction(t)
                    self.assertIn('SimpleQueue.', repr(tx))
                    self.assertIn('SimpleQueue.', repr(tx._core))
                    s.skip(t, method)

        def test_blocking_get_woken_by_put(self):
            # get on an empty queue parks at BLOCKED, then blocks in
            # COMMIT inside actual.get; a concurrently-driven put enqueues
            # an item and wakes it.  Validates the opaque-commit approach
            # end to end (no introspection of the queue's internals).
            s = Scenario(); q = s.SimpleQueue(); out = []
            def getter(): out.append(q.get())
            def putter(): q.put('x')
            with s:
                tg = s.thread(getter); tp = s.thread(putter)
                s.wait(q.get)
                s.wait(q.put)
                self.assertEqual(s.transactions[tg].state, State.BLOCKED)
                self.assertEqual(s.transactions[tp].state, State.BLOCKED)
                s.skip(tg, q.get, tp, q.put)
            self.assertEqual(out, ['x'])

        def test_get_expire_raises_empty(self):
            # get is a TimeoutTransaction; the scenario-level api.expire
            # convenience forces its commit's actual.get(timeout=0) to
            # raise queue.Empty -- parity with Lock/Semaphore.
            s = Scenario(); q = s.SimpleQueue(); api = s.api(q); err = []
            def getter():
                try:
                    q.get()
                except queue.Empty:
                    err.append('Empty')
            with s:
                tg = s.thread(getter)
                s.wait(q.get)
                self.assertEqual(api.expire(q.get, tg), (tg,))
                s.skip(tg, q.get)
            self.assertEqual(err, ['Empty'])

        def test_get_disregard_and_revert(self):
            # disregard drops a get's timeout; revert restores it.  After
            # the round-trip the (still-blocking) get is woken by a put.
            s = Scenario(); q = s.SimpleQueue(); api = s.api(q); out = []
            def getter(): out.append(q.get(timeout=99))
            def putter(): q.put('v')
            with s:
                tg = s.thread(getter); tp = s.thread(putter)
                s.wait(q.get); s.wait(q.put)
                self.assertEqual(api.disregard(q.get, tg), (tg,))
                self.assertEqual(api.revert(q.get, tg), (tg,))
                s.skip(tg, q.get, tp, q.put)
            self.assertEqual(out, ['v'])

        def test_expire_wrong_method_rejected(self):
            s = Scenario(); q = s.SimpleQueue(); api = s.api(q)
            def getter():
                try:
                    q.get()
                except queue.Empty:
                    pass
            with s:
                tg = s.thread(getter)
                s.wait(q.get)
                with self.assertRaises(ValueError):
                    api.expire(q.put, tg)
                api.expire(q.get, tg)
                s.skip(tg, q.get)

        def test_get_nowait_raises_empty_under_scheduler(self):
            s = Scenario(); q = s.SimpleQueue(); err = []
            def getter():
                try:
                    q.get_nowait()
                except queue.Empty:
                    err.append('Empty')
            with s:
                tg = s.thread(getter)
                s.wait(q.get_nowait)
                s.skip(tg, q.get_nowait)
            self.assertEqual(err, ['Empty'])

        def test_deliver_reorders_get_after_put(self):
            s = Scenario(); q = s.SimpleQueue(); out = []
            def getter(): out.append(q.get())
            def putter(): q.put('x')
            with s:
                tg = s.thread(getter)
                tp = s.thread(putter)
                txs = s.api(q).deliver(tg, tp)
            self.assertEqual(out, ['x'])
            self.assertEqual([tx.method for tx in txs], [q.get, q.put])
            self.assertTrue(all(tx.state is State.RETURNED for tx in txs))

        def test_deliver_supports_nowait_methods(self):
            s = Scenario(); q = s.SimpleQueue(); out = []
            def getter(): out.append(q.get_nowait())
            def putter(): q.put_nowait('x')
            with s:
                tg = s.thread(getter)
                tp = s.thread(putter)
                txs = s.api(q).deliver(tg, tp)
            self.assertEqual(out, ['x'])
            self.assertEqual([tx.method for tx in txs], [q.get_nowait, q.put_nowait])


def run_tests():
    blankettestlib.run(name="blanket.primitives.py37_plus", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
