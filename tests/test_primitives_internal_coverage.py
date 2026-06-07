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
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import types
import unittest
import threading
import queue

import blankettestlib
blankettestlib.preload_local_blanket()

from blanket import Scenario, Not, Call, Use, State, Terminated, Nested, Waiting, Predicate, Action, CompetingDriversError, DriverStatus
from blanket import primitives as primitives_module

NEVER = 1e9


class FakeDriver:
    def __init__(self, score, name, *, signals=(), done=False):
        self.score = score
        self.thread = types.SimpleNamespace(name=name)
        self.state = score.Driver.success
        self.mode = self.state
        self.owner = None
        self.signals = frozenset(signals)
        self.done = done
        self.routed = False
        self.closed = 0
        self.calls = []

    def __repr__(self):
        return f"<FakeDriver {self.thread.name} owner={self.owner!r}>"

    def register(self, owner):
        if self.owner is not None:
            raise RuntimeError(f"already owned by {self.owner!r}")
        self.owner = owner
        self.calls.append(("register", owner))

    def unregister(self):
        if self.owner is None:
            raise RuntimeError("not owned")
        self.calls.append(("unregister", self.owner))
        self.owner = None

    def close(self):
        self.closed += 1
        self.calls.append(("close",))

    def route(self, route):
        self.routed = True
        self._route_iterator = route(self)

    def _call_once(self):
        self.calls.append(("call",))

    def __call__(self):
        iterator = getattr(self, "_route_iterator", None)
        if iterator is not None:
            self._route_iterator = None
            while True:
                try:
                    next(iterator)
                except StopIteration:
                    return
                self._call_once()
        else:
            self._call_once()

    def scan(self):
        self.calls.append(("scan",))

    def drive(self):
        self.calls.append(("drive",))
        self.done = False

    def proceed(self):
        self.drive()
        return self.signals

    def signal(self, signals):
        self.calls.append(("signal", frozenset(signals)))
        self.signals = frozenset()
        return self.signals

    def arm_blanket_pause(self, tx=None):  # coverage: fake Driver helper
        tx = self.tx if tx is None else tx
        if getattr(tx, "blanket_pause", False):
            return False
        tx.blanket_pause = True
        return True

    def release_blanket_pause(self, tx=None):  # coverage: fake Driver helper
        tx = self.tx if tx is None else tx
        if getattr(tx, "blanket_pause", False):
            tx.blanket_pause = False
        return True

    def handoff_blanket_pause_to_scheduler_pause(self, tx=None):  # coverage: fake Driver helper
        tx = self.tx if tx is None else tx
        if hasattr(tx, "scheduler_pause"):
            tx.scheduler_pause = True
        if hasattr(tx, "pause"):
            tx.pause = True
        if hasattr(tx, "blanket_pause"):
            tx.blanket_pause = False
        return True

    def pause_internal(self):  # coverage: fake Driver helper
        self.calls.append(("pause_internal",))
        if hasattr(self, "tx"):
            self.arm_blanket_pause(self.tx)


def make_routeable_fake(driver, call_once):
    """Give a small internal Driver fake enough route protocol for tests."""
    driver.routed = False
    driver._route_iterator = None

    def route(route):
        driver.routed = True
        driver._route_iterator = route(driver)

    def call(self):
        iterator = driver._route_iterator
        if iterator is not None:
            driver._route_iterator = None
            while True:
                try:
                    next(iterator)
                except StopIteration:
                    return
                call_once()
        else:
            call_once()

    driver.route = route
    driver.__class__.__call__ = call


class TestChainCoverage(unittest.TestCase):
    def test_fake_driver_helper_guards_and_signal_return(self):
        score = Scenario()._core
        d = FakeDriver(score, "helper", signals={object()})
        owner = object()
        d.register(owner)
        with self.assertRaisesRegex(RuntimeError, "already owned"):
            d.register(object())
        d.unregister()
        with self.assertRaisesRegex(RuntimeError, "not owned"):
            d.unregister()
        d.scan()
        self.assertIn(("scan",), d.calls)
        self.assertEqual(d.signal({"signal"}), frozenset())
        self.assertIn(("signal", frozenset({"signal"})), d.calls)

        d.tx = types.SimpleNamespace(blanket_pause=False,
                                     scheduler_pause=False, pause=False)
        self.assertTrue(d.arm_blanket_pause())
        self.assertFalse(d.arm_blanket_pause())
        self.assertTrue(d.tx.blanket_pause)
        self.assertTrue(d.release_blanket_pause())
        self.assertFalse(d.tx.blanket_pause)
        self.assertTrue(d.handoff_blanket_pause_to_scheduler_pause())
        self.assertTrue(d.tx.scheduler_pause)
        self.assertTrue(d.tx.pause)
        self.assertFalse(d.tx.blanket_pause)
        d.pause_internal()
        self.assertIn(("pause_internal",), d.calls)
        self.assertTrue(d.tx.blanket_pause)

    def test_chain_register_unregister_error_paths(self):
        score = Scenario()._core
        chain = score.Chain()
        owner = object()
        chain.register(owner)
        self.assertIs(chain.owner, owner)
        with self.assertRaisesRegex(RuntimeError, "already owned"):
            chain.register(object())
        chain.unregister()
        self.assertIsNone(chain.owner)
        with self.assertRaisesRegex(RuntimeError, "not owned"):
            chain.unregister()

    def test_chain_iteration_remove_and_close_assert_state(self):
        score = Scenario()._core
        d1 = FakeDriver(score, "one")
        d2 = FakeDriver(score, "two")
        missing = FakeDriver(score, "missing")
        chain = score.Chain(d1, d2)

        self.assertTrue(chain)
        self.assertEqual(len(chain), 2)
        self.assertIn("one", repr(chain))
        self.assertIs(iter(chain), chain)

        chain.remove(d2)
        self.assertNotIn(d2, chain)
        self.assertIsNone(d2.owner)
        with self.assertRaisesRegex(ValueError, "not in Chain"):
            chain.remove(missing)

        yielded = next(chain)
        self.assertIs(yielded, d1)
        self.assertIn(("call",), d1.calls)
        self.assertIsNone(d1.owner)
        with self.assertRaises(StopIteration):
            next(chain)

        chain.append(d2)
        chain.close()
        self.assertEqual(d2.closed, 1)
        self.assertFalse(chain)
        self.assertIsNone(d2.owner)

    def test_direct_iteration_rejects_owned_chain(self):
        score = Scenario()._core
        d = FakeDriver(score, "owned")
        chain = score.Chain(d)
        dispatch = score.Dispatch()
        dispatch.add(chain)
        self.assertIs(chain.owner, dispatch)
        with self.assertRaisesRegex(RuntimeError, "can't iterate Chain directly"):
            next(chain)


class TestDispatchCoverage(unittest.TestCase):
    def test_dispatch_add_contains_remove_and_repr(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d = FakeDriver(score, "d")

        self.assertIn("0 drivers", repr(dispatch))
        self.assertNotIn(d, dispatch)
        dispatch.add(d)
        dispatch.add(d)  # duplicate add is a no-op
        self.assertIn(d, dispatch)
        self.assertIs(d.owner, dispatch)
        self.assertEqual(len(dispatch.recent), 1)

        dispatch.remove(d)
        self.assertIsNone(d.owner)
        self.assertNotIn(d, dispatch)
        with self.assertRaisesRegex(ValueError, "unknown driver"):
            dispatch.remove(d)

    def test_dispatch_remove_unknown_chain_raises(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        chain = score.Chain()
        with self.assertRaisesRegex(ValueError, "unknown Chain"):
            dispatch.remove(chain)
        self.assertNotIn(chain, dispatch)

    def test_dispatch_drains_driver_to_queue_and_yields_it(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d = FakeDriver(score, "queued")
        dispatch.add(d)

        yielded = next(dispatch)
        self.assertIs(yielded, d)
        self.assertIn(("drive",), d.calls)
        self.assertIsNone(d.owner)
        with self.assertRaises(StopIteration):
            next(dispatch)

    def test_dispatch_drains_done_driver_by_proceeding(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d = FakeDriver(score, "done", done=True)
        dispatch.add(d)

        yielded = next(dispatch)
        self.assertIs(yielded, d)
        self.assertIn(("drive",), d.calls)

    def test_dispatch_chain_promotion_and_advance(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d1 = FakeDriver(score, "first")
        d2 = FakeDriver(score, "second")
        chain = score.Chain(d1, d2)
        dispatch.add(chain)

        first = next(dispatch)
        self.assertIs(first, d1)
        self.assertIs(chain.owner, dispatch)
        self.assertIsNone(d1.owner)
        self.assertIn(chain, dispatch.recent)

        second = next(dispatch)
        self.assertIs(second, d2)
        self.assertIsNone(d2.owner)
        self.assertFalse(chain.pending)
        self.assertIs(chain.owner, dispatch)

    def test_dispatch_discard_chain_with_promoted_driver(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d = FakeDriver(score, "promoted")
        chain = score.Chain()
        chain.register(dispatch)
        d.register(dispatch)
        dispatch.drivers[d] = frozenset({object()})
        dispatch.driver_to_chain[d] = chain

        dispatch.discard(chain)
        self.assertIsNone(chain.owner)
        self.assertIsNone(d.owner)
        self.assertNotIn(d, dispatch.drivers)
        self.assertNotIn(d, dispatch.driver_to_chain)

    def test_dispatch_close_closes_drivers_and_chains(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        active = FakeDriver(score, "active")
        queued = FakeDriver(score, "queued")
        recent = FakeDriver(score, "recent")
        chained = FakeDriver(score, "chained")
        chain = score.Chain(chained)

        for d in (active, queued, recent):
            d.register(dispatch)
        chain.register(dispatch)
        dispatch.drivers[active] = frozenset({object()})
        dispatch.queue.append(queued)
        dispatch.recent.append(recent)
        dispatch.driver_to_chain[active] = chain

        dispatch.close()
        self.assertEqual(active.closed, 1)
        self.assertEqual(queued.closed, 1)
        self.assertEqual(recent.closed, 1)
        self.assertEqual(chained.closed, 1)
        self.assertIsNone(active.owner)
        self.assertIsNone(queued.owner)
        self.assertIsNone(recent.owner)
        self.assertIsNone(chain.owner)
        self.assertFalse(dispatch.drivers)
        self.assertFalse(dispatch.queue)
        self.assertFalse(dispatch.recent)


    def test_dispatch_discard_driver_from_recent_active_queue_and_chain(self):
        score = Scenario()._core
        dispatch = score.Dispatch()

        recent = FakeDriver(score, "recent")
        recent.register(dispatch)
        dispatch.recent.append(recent)
        self.assertIn(recent, dispatch)
        dispatch.discard(recent)
        self.assertIsNone(recent.owner)
        self.assertNotIn(recent, dispatch.recent)

        active = FakeDriver(score, "active")
        active.register(dispatch)
        dispatch.drivers[active] = frozenset({object()})
        self.assertIn(active, dispatch)
        dispatch.discard(active)
        self.assertIsNone(active.owner)
        self.assertNotIn(active, dispatch.drivers)

        queued = FakeDriver(score, "queued")
        queued.register(dispatch)
        dispatch.queue.append(queued)
        self.assertIn(queued, dispatch)
        dispatch.discard(queued)
        self.assertIsNone(queued.owner)
        self.assertNotIn(queued, dispatch.queue)

        promoted = FakeDriver(score, "promoted")
        next_driver = FakeDriver(score, "next")
        chain = score.Chain(next_driver)
        promoted.register(dispatch)
        dispatch.drivers[promoted] = frozenset({object()})
        dispatch.driver_to_chain[promoted] = chain
        dispatch.discard(promoted)
        self.assertIsNone(promoted.owner)
        self.assertNotIn(promoted, dispatch.driver_to_chain)
        self.assertIn(chain, dispatch.recent)

    def test_dispatch_discard_chain_removes_promoted_queued_driver(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        chain = score.Chain()
        queued = FakeDriver(score, "queued-promoted")

        chain.register(dispatch)
        queued.register(dispatch)
        dispatch.queue.append(queued)
        dispatch.driver_to_chain[queued] = chain

        dispatch.discard(chain)

        self.assertIsNone(chain.owner)
        self.assertIsNone(queued.owner)
        self.assertNotIn(queued, dispatch.queue)
        self.assertNotIn(queued, dispatch.driver_to_chain)

    def test_dispatch_drain_empty_chain_does_not_enqueue_driver(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        chain = score.Chain()
        chain.register(dispatch)
        dispatch.recent.append(chain)

        dispatch.drain_recent()

        self.assertFalse(dispatch.recent)
        self.assertFalse(dispatch.drivers)
        self.assertFalse(dispatch.queue)
        self.assertIs(chain.owner, dispatch)


class TestDriverAndContextManagerInternals(unittest.TestCase):
    def test_core_driver_claim_slot_and_scan_staging(self):
        scenario = Scenario()
        score = scenario._core
        thread = threading.Thread(target=lambda: None, name="driver-errors")
        d1 = score.Driver(thread)
        d2 = score.Driver(thread)
        score.drivers[thread] = d1
        with self.assertRaises(CompetingDriversError):
            d2.claim_slot()

        score.entered = True
        try:
            d1.scan()
            self.assertEqual(d1.directive, d1.scan)
            self.assertEqual(d1.directive_args, (None,))
            self.assertIsNotNone(d1.closure)
        finally:
            score.entered = False

    def test_context_manager_exit_closes_active_core_drivers(self):
        scenario = Scenario()
        score = scenario._core
        fake = FakeDriver(score, "active-exit")
        thread = threading.Thread(target=lambda: None)

        scenario.__enter__()
        score.drivers[thread] = fake
        try:
            scenario.__exit__(None, None, None)
        finally:
            if score.entered:
                scenario.__exit__(None, None, None)

        self.assertEqual(fake.closed, 1)


class TestParserAndSignalCoverage(unittest.TestCase):
    def test_parse_thread_or_base_tuple_errors(self):
        score = Scenario()._core
        thread = threading.Thread(target=lambda: None, name="parse-thread")
        self.assertEqual(score.parse_thread_or_base_tuple(thread, "parse"), (thread, None))
        self.assertIsNone(score.parse_thread_or_base_tuple("not a thread", "parse"))

        with self.assertRaisesRegex(TypeError, "2-tuple"):
            score.parse_thread_or_base_tuple((thread,), "parse")
        with self.assertRaisesRegex(TypeError, "first item"):
            score.parse_thread_or_base_tuple(("not-thread", object()), "parse")
        with self.assertRaisesRegex(TypeError, "second item"):
            score.parse_thread_or_base_tuple((thread, object()), "parse")

    def test_parse_park_skip_args_validation(self):
        scenario = Scenario()
        score = scenario._core
        thread = threading.Thread(target=lambda: None, name="parser")
        lock = scenario.Lock()

        with self.assertRaisesRegex(ValueError, "no thread specified"):
            score.parse_park_skip_args((), "skip")
        with self.assertRaisesRegex(ValueError, "first argument"):
            score.parse_park_skip_args((lock.acquire,), "skip")
        with self.assertRaisesRegex(TypeError, "expected thread"):
            score.parse_park_skip_args((thread, object()), "skip")
        with self.assertRaisesRegex(ValueError, "has no method"):
            score.parse_park_skip_args((thread,), "skip")
        with self.assertRaisesRegex(ValueError, "exactly one method"):
            score.parse_park_skip_args((thread, lock.acquire, lock.release), "block")

    def test_parse_thread_base_pairs_validation(self):
        score = Scenario()._core
        thread = threading.Thread(target=lambda: None, name="pair")
        self.assertEqual(score.parse_thread_base_pairs((thread,), "pairs"), [(thread, None)])
        with self.assertRaisesRegex(TypeError, "expected thread"):
            score.parse_thread_base_pairs((object(),), "pairs")

    def test_drive_named_rejects_inconsistent_base_for_same_thread(self):
        scenario = Scenario()
        score = scenario._core
        lock = scenario.Lock()
        thread = threading.Thread(target=lambda: None, name="inconsistent-base")
        plan = [
            (thread, object(), [lock.acquire]),
            (thread, object(), [lock.release]),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent base"):
            score._drive_named(plan, "skip")

    def test_private_signal_boxes_validate_and_repr(self):
        scenario = Scenario()
        score = scenario._core
        thread = threading.Thread(target=lambda: None, name="boxed")
        lock = scenario.Lock()

        boxed_thread = score.Thread(thread)
        self.assertEqual(boxed_thread.thread, thread)
        self.assertIn("boxed", repr(boxed_thread))
        with self.assertRaisesRegex(TypeError, "threading.Thread"):
            score.Thread(object())

        boxed_method = score.BoundMethod(lock.acquire)
        self.assertEqual(boxed_method.method, lock.acquire)
        self.assertIn("acquire", repr(boxed_method))
        with self.assertRaisesRegex(TypeError, "regulated"):
            score.BoundMethod(object())

        not_thread = Not(thread)
        self.assertIs(score.box_signal(not_thread), not_thread)
        with self.assertRaisesRegex(TypeError, "not signalable"):
            score.box_signal(object())

    def test_signal_rejects_low_or_non_signal_items(self):
        scenario = Scenario()
        score = scenario._core
        with self.assertRaises(AssertionError):
            score.signal(object())
        with self.assertRaisesRegex(AssertionError, "not high"):
            score.signal(Terminated(threading.current_thread()))


class _ProbeSignal(primitives_module.Signaling):
    def __init__(self, name, high):
        self.name = name
        self.high = high
        self.sample_calls = 0

    def sample(self, scenario):
        self.sample_calls += 1
        return self.high

    def __repr__(self):
        return f"_ProbeSignal({self.name!r}, {self.high!r})"


class TestScoreWaitInterestInternals(unittest.TestCase):
    def test_unregister_wait_interests_keeps_nonempty_counters(self):
        scenario = Scenario()
        score = scenario._core
        interest_key = object()
        removed_base = object()
        remaining_base = object()

        score.scoped_call_bases[interest_key][removed_base] = 1
        score.scoped_call_bases[interest_key][remaining_base] = 1
        score.scoped_use_bases[interest_key][removed_base] = 1
        score.scoped_use_bases[interest_key][remaining_base] = 1

        score.unregister_wait_interests([
            ('scoped_call', interest_key, removed_base),
            ('scoped_use', interest_key, removed_base),
        ])

        self.assertNotIn(removed_base, score.scoped_call_bases[interest_key])
        self.assertEqual(score.scoped_call_bases[interest_key][remaining_base], 1)
        self.assertNotIn(removed_base, score.scoped_use_bases[interest_key])
        self.assertEqual(score.scoped_use_bases[interest_key][remaining_base], 1)

    def test_unregister_wait_interests_rejects_unknown_kind(self):
        scenario = Scenario()
        with self.assertRaisesRegex(AssertionError, "unknown wait-interest kind"):
            scenario._core.unregister_wait_interests([('bogus', object())])

    def test_sleeping_not_helpers_wake_only_high_keys(self):
        scenario = Scenario()
        score = scenario._core
        helpers = (
            (score.sleeping_not_calls, score.signal_sleeping_not_calls),
            (score.sleeping_not_uses, score.signal_sleeping_not_uses),
            (score.sleeping_not_nested, score.signal_sleeping_not_nested),
            (score.sleeping_not_transaction_states,
             score.signal_sleeping_not_transaction_states),
        )

        for index, (storage, helper) in enumerate(helpers):
            high = _ProbeSignal(f"high-{index}", True)
            low = _ProbeSignal(f"low-{index}", False)
            released = []

            class Waiter:
                pass

            waiter = Waiter()
            waiter.blocker = lambda released=released: released.append(True)
            waiter.signaled = set()
            storage[high] = 1
            storage[low] = 1
            score.waiters[high].add(waiter)

            helper()

            self.assertEqual(released, [True])
            self.assertIsNone(waiter.blocker)
            self.assertEqual(waiter.signaled, {high})
            self.assertNotIn(high, score.waiters)
            self.assertIn(high, storage)
            self.assertIn(low, storage)
            self.assertGreaterEqual(high.sample_calls, 1)
            self.assertEqual(low.sample_calls, 1)

    def test_signal_rejects_non_signaling_and_low_signals(self):
        scenario = Scenario()
        score = scenario._core
        with self.assertRaises(AssertionError):
            score.signal(object())
        with self.assertRaisesRegex(AssertionError, "not high"):
            score.signal(_ProbeSignal("low", False))


class TestDriverWaitRouteInternals(unittest.TestCase):
    def test_resume_route_without_route_raises_undirected(self):
        score = Scenario()._core
        score.entered = True
        d = score.Driver(threading.Thread(target=lambda: None,
                                          name="no-route"))
        with self.assertRaisesRegex(RuntimeError, "no Driver directive staged"):
            d.resume_route()
        self.assertIs(d.state, d.undirected)

    def test_wait_explicit_signal_from_live_tx(self):
        scenario = Scenario()
        gate = threading.Event()
        def worker():
            gate.wait()
        with scenario:
            t = scenario.thread(worker)
            d = scenario.Driver(t)
            target = Terminated(t)
            d.wait(target)
            helper = threading.Thread(target=gate.set)
            helper.start()
            d()
            helper.join()
            self.assertIs(d.state, d.success)
            self.assertEqual(d.signaled, frozenset({target}))
            self.assertEqual(d.motivation, frozenset({target}))

    def test_wait_unlisted_termination_reports_terminated_without_signals(self):
        class NeverSignal(primitives_module.Signaling):
            def sample(self, scenario):
                return False

        scenario = Scenario()
        with scenario:
            t = scenario.thread(lambda: None)
            d = scenario.Driver(t)
            d.wait(NeverSignal())
            d()
            self.assertIs(d.state, d.terminated)
            self.assertEqual(d.signaled, frozenset({Terminated(t)}))
            self.assertEqual(d.motivation, frozenset({Terminated(t)}))

    def test_wait_rejects_unexpected_signals(self):
        scenario = Scenario()
        ev = scenario.Event()
        def worker2():
            ev.wait()
        with scenario:
            t = scenario.thread(worker2)
            scenario.wait(t)
            d = scenario.Driver(t)
            d.wait(_ProbeSignal("never-unexpected", False))
            with d._lock:
                d._core.drive()
                with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                    d._core.signal({object()})
                self.assertIs(d._core.state, d._core.driving)
            d.close()
            scenario.raw(ev).set()

    def test_route_empty_yield_raises_but_empty_return_ends_route(self):
        scenario = Scenario()
        ev = scenario.Event()
        def worker():
            ev.wait()
        def empty_yield(d):
            yield
        def empty_return(d):
            if False:
                yield
        with scenario:
            t = scenario.thread(worker)
            scenario.wait(t)
            d = scenario.Driver(t, route=empty_yield)
            with self.assertRaisesRegex(RuntimeError, "route yielded"):
                d()
            self.assertIs(d.state, d.undirected)
            d.close()

            d = scenario.Driver(t, route=empty_return)
            d()
            self.assertIs(d.state, d.undirected)
            self.assertFalse(d.routed)
            d.close()
            scenario.raw(ev).set()


class TestInjectionSmallEdges(unittest.TestCase):
    def test_injection_context_manager_repr_and_idempotent_close(self):
        target = types.ModuleType("injection_target")
        target.Lock = threading.Lock
        scenario = Scenario()

        with scenario.inject(target) as injection:
            self.assertIs(target.Lock, scenario.Lock)
            self.assertIn("1 replacements", repr(injection))

        self.assertIs(target.Lock, threading.Lock)
        self.assertIn("closed", repr(injection))
        injection.close()
        self.assertIn("closed", repr(injection))

    def test_core_injection_context_methods_and_repr(self):
        target = types.ModuleType("core_injection_target")
        target.Lock = threading.Lock
        scenario = Scenario()
        impersonators = {
            module: scenario._impersonator(module)
            for module in scenario._core.impersonated_modules
        }
        injection = scenario._core.Injection(target, impersonators)
        self.assertIs(injection.__enter__(), injection)
        self.assertIn("1 replacements", repr(injection))
        self.assertIs(target.Lock, scenario.Lock)
        self.assertFalse(injection.__exit__(None, None, None))
        self.assertIs(target.Lock, threading.Lock)
        self.assertIn("closed", repr(injection))


class TestPrimitiveCoreSmallEdges(unittest.TestCase):
    def test_internal_primitive_signal_validation_and_repr(self):
        scenario = Scenario()
        lock = scenario.Lock()
        signal = scenario._core.Primitive(lock)

        self.assertEqual(signal.primitive, lock)
        self.assertIn("_Primitive(", repr(signal))
        with self.assertRaisesRegex(TypeError, "expected a primitive"):
            scenario._core.Primitive(object())

    def test_threads_to_txs_validation_and_thread_to_tx(self):
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core

        with self.assertRaisesRegex(TypeError, "iterable"):
            core.threads_to_txs(object(), caller="unit")
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            core.threads_to_txs([object()], caller="unit")
        with self.assertRaisesRegex(ValueError, "calling thread"):
            core.threads_to_txs([threading.current_thread()], caller="unit")
        self.assertEqual(core.threads_to_txs([], caller="unit"), ((), []))

        def worker():
            lock.acquire()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(t)
            tx = scenario.transaction(t)
            self.assertIs(core.thread_to_tx(t, caller="unit"), tx._core)
            core.unblock(lock.acquire, (t,))
            self.assertIs(tx.wait(), State.RETURNED)

    def test_api_transaction_and_transaction_log_properties(self):
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        def worker():
            lock.acquire()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(t)
            tx = scenario.transaction(t)
            self.assertIs(api.transaction(t), tx)
            self.assertIsNone(api.transaction(threading.current_thread()))
            self.assertEqual(tx.timeout.value, -1)
            self.assertIsNone(tx.timeout.time)
            self.assertTrue(tx.log)
            self.assertIs(tx.parent, None)
            self.assertEqual(tx.depth, 0)
            api.unblock(lock.acquire, t)
            self.assertIs(tx.wait(), State.RETURNED)
            self.assertIsNotNone(tx.end_time)

    def test_condition_constructor_rejects_non_blanket_or_foreign_lock(self):
        scenario = Scenario()
        with self.assertRaisesRegex(TypeError, "blanket Lock or RLock"):
            scenario.Condition(object())
        foreign = Scenario().Lock()
        with self.assertRaisesRegex(TypeError, "from this Scenario"):
            scenario.Condition(foreign)

    def test_core_injection_repr_context_and_close(self):
        import threading as real_threading
        scenario = Scenario()
        target = types.ModuleType("core_injection_target")
        target.Lock = real_threading.Lock
        impersonators = {module: scenario._impersonator(module)
                         for module in scenario._core.impersonated_modules}
        inj = scenario._core.Injection(target, impersonators)
        try:
            self.assertIn("1 replacements", repr(inj))
            self.assertIs(inj.__enter__(), inj)
            self.assertIs(target.Lock, scenario.Lock)
            self.assertFalse(inj.__exit__(None, None, None))
            self.assertIs(target.Lock, real_threading.Lock)
            self.assertIn("closed", repr(inj))
            inj.close()  # idempotent
            self.assertTrue(inj.closed)
        finally:
            inj.close()

    def test_core_thread_translation_validation(self):
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core

        with self.assertRaisesRegex(TypeError, "iterable of threads"):
            core.threads_to_txs(object(), caller="threads_to_txs")
        self.assertEqual(core.threads_to_txs((), caller="threads_to_txs"), ((), []))
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            core.threads_to_txs((object(),), caller="threads_to_txs")
        with self.assertRaisesRegex(ValueError, "calling thread"):
            core.threads_to_txs((threading.current_thread(),), caller="threads_to_txs")

        stop = threading.Event()
        thread = threading.Thread(target=stop.wait, name="tx-translation")
        thread.start()
        fake_tx = object()
        try:
            scenario._core.threads.add(thread)
            scenario._core.transactions[thread] = fake_tx
            self.assertIs(core.thread_to_tx(thread, caller="thread_to_tx"), fake_tx)
        finally:
            scenario._core.transactions.pop(thread, None)
            scenario._core.threads.discard(thread)
            stop.set()
            thread.join()

    def test_direct_transaction_helper_error_paths(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core

        base_tx = core.Transaction(lock.release, time.monotonic(), regulated=False)
        self.assertFalse(base_tx.is_delegate(object()))
        self.assertIsNone(base_tx.timeout)
        base_tx.state = State.RETURNED
        self.assertIs(base_tx.wait(), State.RETURNED)
        with self.assertRaisesRegex(TypeError, "State or None"):
            base_tx.wait("bad-state")
        base_tx.state = State.BLOCKED
        with self.assertRaises(NotImplementedError):
            base_tx.commit()

        callbacks = []
        base_tx.observe(State.COMMIT, lambda: callbacks.append("commit"))
        self.assertEqual(len(base_tx.state_observers), 1)
        with self.assertRaisesRegex(ValueError, "already at"):
            base_tx.observe(State.BLOCKED, lambda: callbacks.append("blocked"))
        base_tx.to(State.COMMIT)
        self.assertEqual(callbacks, ["commit"])
        self.assertFalse(base_tx.state_observers)

        plain_tx = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        with self.assertRaisesRegex(NotImplementedError, "expire"):
            plain_tx.expire()
        with self.assertRaisesRegex(NotImplementedError, "disregard"):
            plain_tx.disregard()
        with self.assertRaisesRegex(NotImplementedError, "revert"):
            plain_tx.revert()
        with self.assertRaisesRegex(RuntimeError, "no blocker"):
            plain_tx.unpark(State.COMMIT)
        plain_tx.state = State.COMMIT
        with self.assertRaisesRegex(RuntimeError, "can't unblock"):
            plain_tx.unblock()
        with self.assertRaisesRegex(RuntimeError, "can't unstall"):
            plain_tx.unstall()
        plain_tx.state = State.RETURNED
        plain_tx.unpause()
        with self.assertRaisesRegex(RuntimeError, "advanced past PAUSED"):
            plain_tx.set_scheduler_pause(True)
        with self.assertRaisesRegex(RuntimeError, "not in a scheduler-controlled"):
            plain_tx.unstick()

        pause_tx = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        pause_tx.set_scheduler_pause(True)
        self.assertTrue(pause_tx.pause)
        self.assertTrue(pause_tx.paused)
        pause_tx.state = State.RETURNED
        with self.assertRaisesRegex(RuntimeError, "advanced past PAUSED"):
            pause_tx.unpause()
        pause_tx.state = State.BLOCKED
        pause_tx.set_scheduler_pause(False)
        self.assertFalse(pause_tx.pause)
        self.assertFalse(pause_tx.paused)
        pause_tx.blanket_pause = True
        self.assertTrue(pause_tx.paused)
        pause_tx.clear_all_pauses()
        self.assertFalse(pause_tx.paused)

        with scenario._core.lock:
            self.assertEqual(
                scenario._core.wait((), timeout=0, all=True),
                frozenset())

    def test_transaction_validate_error_message_shapes(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core
        tx = core.methods[lock.acquire](lock.acquire, time.monotonic(), regulated=False)

        with self.assertRaisesRegex(ValueError, "a transaction in COMMIT"):
            tx.validate(state=State.COMMIT, caller="validate")
        with self.assertRaisesRegex(ValueError, "release on this"):
            tx.validate(method=lock.release, state=None)
        with self.assertRaisesRegex(ValueError, "one of"):
            tx.validate(method=(lock.release, lock.locked), state=(State.COMMIT, State.WAITING))
        with self.assertRaisesRegex(ValueError, "custom thing"):
            tx.validate(method=lock.release, state=State.COMMIT, method_description="custom thing")


class TestTransactionValidationAndObserverCoverage(unittest.TestCase):
    def test_plain_transaction_validate_error_messages_and_observers(self):
        scenario = Scenario()
        lock = scenario.Lock()
        observed = []

        def worker():
            lock.acquire()
            lock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.skip(t, lock.acquire)
            scenario.wait(Call(t, lock.release, State.BLOCKED))
            tx = scenario.transaction(t)
            core = tx._core

            self.assertFalse(core.is_delegate(None))
            self.assertEqual(tx.timeout.value, None)
            self.assertEqual(tx.timeout.time, None)

            with self.assertRaisesRegex(ValueError, "already at state"):
                core.observe(State.BLOCKED, lambda: None)
            core.observe(State.COMMITTED, lambda: observed.append("committed"))

            with self.assertRaisesRegex(ValueError, "a transaction"):
                core.validate(method=None, state=State.WAITING, caller="unit")
            with self.assertRaisesRegex(ValueError, "any state"):
                core.validate(method=lock.acquire, state=None, caller="unit")
            with self.assertRaisesRegex(ValueError, "one of"):
                core.validate(
                    method=(lock.acquire, object()),
                    state=(State.WAITING, State.STALLED),
                    caller="unit")

            for method_name in ("expire", "disregard", "revert"):
                with self.assertRaises(NotImplementedError):
                    getattr(tx, method_name)()

            tx.unblock()
            self.assertIs(tx.wait(), State.RETURNED)
            self.assertEqual(observed, ["committed"])

    def test_context_manager_exit_closes_active_driver(self):
        scenario = Scenario()
        score = scenario._core
        cm = score.ContextManager()
        d = FakeDriver(score, "active-driver")
        score.entered = True
        score.drivers[threading.current_thread()] = d

        self.assertFalse(cm.__exit__(None, None, None))
        self.assertEqual(d.closed, 1)
        self.assertFalse(score.entered)

    def test_assign_lone_acquirer_rejects_locked_lock(self):
        scenario = Scenario()
        lock = scenario.Lock()
        thread = threading.Thread(target=lambda: None, name="would-acquire")
        self.assertTrue(lock.acquire())
        try:
            with self.assertRaisesRegex(RuntimeError, "unlocked, not locked"):
                scenario.api(lock).assign(thread)
        finally:
            lock.release()

    def test_scoped_use_and_call_helpers_find_ancestor_base(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        score = scenario._core
        core = lock._core
        thread = threading.current_thread()

        base = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        middle = core.methods[lock.locked](lock.locked, time.monotonic(), regulated=False)
        child = core.methods[lock.acquire](lock.acquire, time.monotonic(), regulated=False)
        middle.parent = base
        child.parent = middle

        score.scoped_use_bases[(thread, lock)][base] += 1
        use_signals = score.use_signals(thread, lock, child)
        self.assertIn(Use(thread, lock), use_signals)
        self.assertIn(Use((thread, base.api), lock), use_signals)

        score.scoped_call_bases[(thread, lock.acquire)][base] += 1
        call_signals = child.call_signals(State.BLOCKED)
        self.assertIn(Call(thread, lock.acquire, State.BLOCKED), call_signals)
        self.assertIn(Call((thread, base.api), lock.acquire, State.BLOCKED), call_signals)

    def test_transaction_aborted_unwinds_parent_chain(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        score = scenario._core
        core = lock._core
        thread = threading.current_thread()

        parent = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        score.transactions[thread] = parent
        try:
            child = core.methods[lock.acquire](lock.acquire, time.monotonic(), regulated=False)
            self.assertIs(child.parent, parent)
            # Transaction.__call__ now publishes .child; this synthetic
            # direct-constructor test wires it manually to exercise abort
            # cleanup of a published child chain.
            parent.child = child
            self.assertIs(parent.child, child)
            child.aborted()
            self.assertTrue(child.done)
            self.assertTrue(parent.done)
            self.assertIsNone(parent.child)
            self.assertIsInstance(child.result, RuntimeError)
            self.assertIsInstance(parent.result, RuntimeError)
        finally:
            score.transactions.pop(thread, None)

    def make_cycle_base(self, ready_threads=(), *, closed=False):
        scenario = Scenario()
        cond = scenario.Condition()
        cycle = object.__new__(cond._core.CycleBase)
        cycle.core = cond._core
        cycle.extra_waiters = 0
        cycle.closed = closed
        cycle.ready = {}
        for thread in ready_threads:
            d = types.SimpleNamespace(thread=thread)
            cycle.ready[thread] = d
        def wake_drivers(drivers, *, pause=False):
            cycle.last_pause = pause
            for d in drivers:
                cycle.ready.pop(d.thread, None)
            if not cycle.ready:
                cycle.closed = True
            return drivers
        cycle.wake_drivers = wake_drivers
        return cycle

    def test_cycle_base_resolve_and_empty_errors(self):
        t = threading.Thread(target=lambda: None, name="cycle-ready")
        missing = threading.Thread(target=lambda: None, name="cycle-missing")
        cycle = self.make_cycle_base((t,))

        self.assertEqual([d.thread for d in cycle.resolve_drivers((t,))], [t])
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            cycle.resolve_drivers((object(),))
        with self.assertRaisesRegex(ValueError, "specified more than once"):
            cycle.resolve_drivers((t, t))
        with self.assertRaisesRegex(ValueError, "not a ready waiter"):
            cycle.resolve_drivers((missing,))

        empty = self.make_cycle_base(())
        with self.assertRaisesRegex(ValueError, r"wake\(\): no threads"):
            empty.wake(())
        with self.assertRaisesRegex(ValueError, r"pause\(\): no threads"):
            empty.pause(())
        with self.assertRaisesRegex(TypeError, "Condition cycles"):
            empty.wait(())
        self.assertIsNone(empty.next_thread())
        self.assertEqual(empty.close(), ())
        self.assertTrue(empty.closed)
        with self.assertRaises(NotImplementedError):
            empty.repr()

    def test_cycle_base_wake_pause_next_and_closed_errors(self):
        t1 = threading.Thread(target=lambda: None, name="cycle-one")
        t2 = threading.Thread(target=lambda: None, name="cycle-two")

        cycle = self.make_cycle_base((t1, t2))
        self.assertIs(cycle.wake(()), t1)
        self.assertFalse(cycle.last_pause)
        self.assertIn(t2, cycle.ready)
        self.assertFalse(cycle.closed)
        self.assertEqual(cycle.pause((t2,)), (t2,))
        self.assertTrue(cycle.last_pause)
        self.assertTrue(cycle.closed)

        cycle = self.make_cycle_base((t1,))
        self.assertIs(cycle.next_thread(), t1)
        self.assertTrue(cycle.closed)

        closed = self.make_cycle_base((t1,), closed=True)
        with self.assertRaisesRegex(RuntimeError, "cycle is closed"):
            closed.wake(())
        with self.assertRaisesRegex(RuntimeError, "cycle is closed"):
            closed.pause(())
        self.assertEqual(closed.close(), ())


class TestCycleReprCoverage(unittest.TestCase):
    def test_barrier_cycle_repr_reports_ready_count(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        results = []

        def worker(name):
            def run():
                barrier.wait()
                results.append(name)
            return run

        with scenario:
            a = scenario.thread(worker("a"))
            b = scenario.thread(worker("b"))
            scenario.wait(a)
            scenario.wait(b)
            cycle = api.cycle(a, b)
            self.assertIn("Barrier.cycle", repr(cycle))
            self.assertEqual(cycle.close(), (a, b))

        self.assertEqual(sorted(results), ["a", "b"])

    def test_event_cycle_repr_reports_ready_count(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        results = []

        def waiter():
            event.wait()
            results.append("waiter")

        def setter():
            event.set()
            results.append("setter")

        with scenario:
            w = scenario.thread(waiter)
            s = scenario.thread(setter)
            cycle = api.cycle(w, s)
            self.assertIn("Event.cycle", repr(cycle))
            self.assertEqual(cycle.close(), (w, s))

        self.assertEqual(sorted(results), ["setter", "waiter"])



class _FakeCycleCoreForAPI:
    def __init__(self):
        self.ready = []
        self.closed = False
        self.extra_waiters = 3
        self.calls = []
        self._wake_no_thread_results = ["alpha", ValueError("empty")]
        self._next_threads = []

    def repr(self):
        self.calls.append(("repr",))
        return "<fake cycle core>"

    def wake(self, threads):
        self.calls.append(("wake", threads))
        if threads:
            return tuple(threads)
        result = self._wake_no_thread_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def pause(self, threads):
        self.calls.append(("pause", threads))
        return tuple(threads) if threads else "paused"

    def wait(self, threads):
        self.calls.append(("wait", threads))
        return tuple(threads) if threads else "waited"

    def next_thread(self):
        self.calls.append(("next_thread",))
        if self._next_threads:
            return self._next_threads.pop(0)
        return None

    def close(self):
        self.calls.append(("close",))
        self.closed = True
        return ("closed",)


class TestCycleAPIBaseWrapperCoverage(unittest.TestCase):
    def make_cycle_api(self):
        scenario = Scenario()
        condition = scenario.Condition()
        api = scenario.api(condition)
        cycle = object.__new__(api.cycle)
        cycle._lock = threading.Lock()
        cycle._core = _FakeCycleCoreForAPI()
        return cycle

    def test_cycle_api_wrapper_delegates_and_iterates(self):
        cycle = self.make_cycle_api()
        core = cycle._core
        t1 = threading.Thread(target=lambda: None, name="cycle-api-one")
        t2 = threading.Thread(target=lambda: None, name="cycle-api-two")

        self.assertEqual(repr(cycle), "<fake cycle core>")
        self.assertEqual(cycle.ready, ())
        self.assertEqual(cycle.waiters, ())
        self.assertFalse(cycle.closed)
        self.assertEqual(cycle.extra_waiters, 3)

        self.assertEqual(cycle.wake(t1), (t1,))
        self.assertEqual(cycle.pause(t2), (t2,))
        self.assertEqual(cycle.wait(t1, t2), (t1, t2))
        self.assertEqual(list(cycle.iter(t1, t2)), [t1, t2])
        self.assertEqual(list(cycle.iter()), ["alpha"])
        self.assertIs(iter(cycle), cycle)

        core._next_threads = [t1, None]
        self.assertIs(next(cycle), t1)
        with self.assertRaises(StopIteration):
            next(cycle)
        self.assertIsNone(core.next_thread())

        self.assertEqual(cycle(), ("closed",))
        self.assertTrue(cycle.closed)
        self.assertIn(("wake", (t1,)), core.calls)
        self.assertIn(("pause", (t2,)), core.calls)
        self.assertIn(("wait", (t1, t2)), core.calls)

    def test_cycle_api_call_with_threads_and_context_exit_suppression(self):
        cycle = self.make_cycle_api()
        core = cycle._core
        t = threading.Thread(target=lambda: None, name="cycle-api-thread")

        self.assertIs(cycle.__enter__(), cycle)
        self.assertEqual(cycle(t), (t, "closed"))
        self.assertIn(("wake", (t,)), core.calls)

        cycle = self.make_cycle_api()
        def raising_close():
            cycle._core.calls.append(("raising_close",))
            raise RuntimeError("cleanup failed")
        cycle._core.close = raising_close
        self.assertFalse(cycle.__exit__(ValueError, ValueError("body"), None))
        self.assertIn(("raising_close",), cycle._core.calls)


class TestMorePrimitiveEdgeCoverage(unittest.TestCase):
    def test_condition_constructor_rejects_lock_with_wrong_core_score(self):
        scenario = Scenario()
        lock = scenario.Lock()
        original_core = lock._core
        try:
            lock._core = types.SimpleNamespace(score=object())
            with self.assertRaisesRegex(TypeError, "from this Scenario"):
                scenario.Condition(lock)
        finally:
            lock._core = original_core

    def test_cycle_base_pause_without_named_threads_and_closed_next(self):
        t = threading.Thread(target=lambda: None, name="cycle-pause-default")
        base = TestTransactionValidationAndObserverCoverage()
        cycle = base.make_cycle_base((t,))

        self.assertIs(cycle.pause(()), t)
        self.assertTrue(cycle.last_pause)
        self.assertTrue(cycle.closed)
        self.assertIsNone(cycle.next_thread())

    def test_condition_and_semaphore_api_small_wrappers(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cond_api = scenario.api(condition)
        semaphore = scenario.Semaphore(1)
        sem_api = scenario.api(semaphore)

        self.assertIn("ConditionAPI", repr(cond_api))
        self.assertEqual(cond_api.waiters, 0)
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            cond_api.disregard(condition.wait, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            cond_api.revert(condition.wait, object())

        self.assertIn("SemaphoreAPI", repr(sem_api))
        self.assertEqual(sem_api.waiters, 0)
        self.assertEqual(sem_api.value, 1)
        self.assertEqual(sem_api.available, 1)
        with self.assertRaisesRegex(ValueError, "allocate requires"):
            sem_api.allocate()
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            sem_api.revert(semaphore.acquire, object())

        q = scenario.Queue(maxsize=2)
        q_api = scenario.api(q)
        self.assertEqual(q._core.maxsize, 2)
        self.assertEqual(q_api.maxsize, 2)
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            q_api.expire(q.get, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            q_api.disregard(q.get, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            q_api.revert(q.get, object())

        event = scenario.Event()
        event_api = scenario.api(event)
        self.assertIn("EventAPI", repr(event_api))
        self.assertEqual(event_api.waiters, 0)
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            event_api.expire(event.wait, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            event_api.disregard(event.wait, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            event_api.revert(event.wait, object())

        barrier = scenario.Barrier(2)
        barrier_api = scenario.api(barrier)
        self.assertIn("BarrierAPI", repr(barrier_api))
        self.assertEqual(barrier_api.parties, 2)
        self.assertEqual(barrier_api.n_waiting, 0)
        self.assertFalse(barrier_api.broken)
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            barrier_api.expire(barrier.wait, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            barrier_api.disregard(barrier.wait, object())
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            barrier_api.revert(barrier.wait, object())



class TestConditionCycleCoreDirectCoverage(unittest.TestCase):
    def make_condition_cycle(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.Cycle)
        cycle.core = condition._core
        cycle.caller = "cycle"
        cycle.ul_release = (condition._core.underlying.primitive.release,
                            condition._core.underlying.raw.release)
        cycle.previous = None
        return cycle

    def test_condition_cycle_ensure_ul_free_error_paths(self):
        cycle = self.make_condition_cycle()
        held = {"value": True}
        def finish_deep(driver):
            while driver.tx is not None and not driver.tx.done:
                driver.finish()
                driver()

        cycle.core = types.SimpleNamespace(
            score=types.SimpleNamespace(driver_finish_deep=finish_deep),
            underlying=types.SimpleNamespace(
                actual_held=lambda: held["value"]
            )
        )
        release_method = object()
        cycle.ul_release = (release_method,)
        cycle.caller = "cycle"
        cycle.previous = None

        with self.assertRaisesRegex(RuntimeError, "no previous thread"):
            cycle.ensure_ul_free()

        class Previous:
            def __init__(self, state, tx):
                self.thread = threading.Thread(target=lambda: None, name="previous")
                self.success = object()
                self.terminated = object()
                self.raised = object()
                self.impasse = object()
                self.overshot = object()
                self.mutated = object()
                self.state = state
                self.tx = tx
                self.called = 0
                self.finished_calls = 0

            @property
            def done(self):
                return self.state in (self.terminated, self.raised)

            def scan(self):
                pass

            def __call__(self):
                self.called += 1

            def finish(self):
                self.finished_calls += 1
                held["value"] = False
                self.tx.done = True
                self.tx.state = State.RETURNED
                self.state = self.success

        def fake_tx(method=release_method, *, done=False):
            return types.SimpleNamespace(method=method, done=done, state=State.BLOCKED)

        prev = Previous(object(), fake_tx(done=True))
        prev.state = prev.terminated
        self.assertTrue(prev.done)
        cycle.previous = prev
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            cycle.ensure_ul_free()
        self.assertEqual(prev.called, 1)

        prev = Previous(object(), None)
        cycle.previous = prev
        with self.assertRaisesRegex(RuntimeError, "not at lock.release"):
            cycle.ensure_ul_free()

        held["value"] = True
        prev = Previous(object(), fake_tx())
        cycle.previous = prev
        cycle.ensure_ul_free()
        self.assertEqual(prev.called, 1)
        self.assertEqual(prev.finished_calls, 1)
        self.assertFalse(held["value"])

    def test_condition_cycle_resolve_ready_and_act_errors(self):
        cycle = self.make_condition_cycle()
        t1 = threading.Thread(target=lambda: None, name="ready-one")
        t2 = threading.Thread(target=lambda: None, name="ready-two")
        missing = threading.Thread(target=lambda: None, name="missing-ready")
        d1 = types.SimpleNamespace(thread=t1)
        d2 = types.SimpleNamespace(thread=t2)
        cycle.drivers = [d1, d2]
        cycle.ready = []
        cycle.incoming = [d1]
        cycle.waiting = []
        processed = []

        def process():
            processed.append("process")
            cycle.ready.append(d1)
            cycle.incoming.clear()
        cycle.process = process

        self.assertIs(cycle.find_ready(t1), None)
        self.assertEqual(cycle.resolve_ready((t1,)), [d1])
        self.assertEqual(processed, ["process"])
        self.assertIs(cycle.find_ready(t1), d1)

        with self.assertRaisesRegex(TypeError, "expected a thread"):
            cycle.resolve_ready((object(),))
        with self.assertRaisesRegex(ValueError, "specified more than once"):
            cycle.resolve_ready((t1, t1))
        with self.assertRaisesRegex(ValueError, "not a cycle waiter"):
            cycle.resolve_ready((missing,))

        cycle.ready = []
        cycle.incoming = []
        with self.assertRaisesRegex(ValueError, "never became ready"):
            cycle.resolve_ready((t2,))

        cycle.closed = False
        cycle.ready = []
        cycle.incoming = [d1]
        acted = []
        def process_for_act():
            cycle.ready.append(d1)
            cycle.incoming.clear()
        def act_one(driver, verb):
            acted.append((driver.thread, verb))
            cycle.ready.remove(driver)
        cycle.process = process_for_act
        cycle.act_one = act_one
        self.assertIs(cycle.act((), "wake"), t1)
        self.assertEqual(acted, [(t1, "wake")])
        self.assertTrue(cycle.closed)

        cycle.closed = False
        cycle.ready = []
        cycle.incoming = []
        with self.assertRaisesRegex(ValueError, r"wake\(\): no threads"):
            cycle.act((), "wake")



class TestConditionCycleDriveWakerCoverage(unittest.TestCase):
    def make_cycle(self, *, actual_waiters, managed_waiters, notify_n, raised=None):
        import math
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.Cycle)
        notify_method = condition.notify
        cycle.core = types.SimpleNamespace(
            score=scenario._core,
            actual=types.SimpleNamespace(_waiters=[object()] * actual_waiters),
            notify_methods=(notify_method,),
        )
        cycle.caller = "cycle"
        cycle.ul_acquire = object()
        cycle.ensure_ul_free = lambda: None
        cycle.waiting = [object()] * managed_waiters
        cycle.extra_waiters = None
        cycle.notified = False
        cycle.previous = None

        class Waker:
            def __init__(self):
                self.thread = threading.Thread(target=lambda: None, name="cycle-waker")
                self.active = object()
                self.success = object()
                self.state = self.active
                self.tx = types.SimpleNamespace(
                    method=notify_method,
                    n=notify_n,
                    state=State.BLOCKED,
                    result=raised,
                    done=False,
                    validate=lambda **kwargs: None,
                )
                self.finish_calls = 0
                self.drive_calls = 0

            def finish(self):
                self.finish_calls += 1

            def __call__(self):
                self.drive_calls += 1
                self.state = self.success
                self.tx.done = True
                if raised is not None:
                    self.tx.state = State.RAISED

        waker = Waker()
        cycle.incoming = [waker]
        return cycle, waker

    def test_drive_waker_rejects_finite_notify_with_extra_waiters(self):
        cycle, waker = self.make_cycle(
            actual_waiters=2, managed_waiters=1, notify_n=1)
        with self.assertRaisesRegex(ValueError, "extra waiters"):
            cycle.drive_waker()
        waker.state = waker.active
        self.assertIs(waker.state, waker.active)
        self.assertEqual(waker.finish_calls, 0)

    def test_drive_waker_rejects_impossible_notify_all_waiter_count(self):
        import math
        cycle, waker = self.make_cycle(
            actual_waiters=0, managed_waiters=1, notify_n=math.inf)
        with self.assertRaisesRegex(RuntimeError, "less than managed"):
            cycle.drive_waker()
        self.assertEqual(waker.finish_calls, 0)

    def test_drive_waker_surfaces_notifier_exception(self):
        error = RuntimeError("notify failed")
        cycle, waker = self.make_cycle(
            actual_waiters=1, managed_waiters=1, notify_n=1, raised=error)
        with self.assertRaisesRegex(RuntimeError, "notify failed"):
            cycle.drive_waker()
        self.assertEqual(waker.finish_calls, 1)
        self.assertEqual(waker.drive_calls, 1)
        self.assertFalse(cycle.notified)


class TestAdditionalPrimitiveCoverageEdges(unittest.TestCase):
    def test_context_manager_exit_leaves_done_core_driver_alone(self):
        scenario = Scenario()
        score = scenario._core
        fake = FakeDriver(score, "done-exit", done=True)
        thread = threading.Thread(target=lambda: None)

        scenario.__enter__()
        score.drivers[thread] = fake
        try:
            scenario.__exit__(None, None, None)
        finally:
            if score.entered:
                scenario.__exit__(None, None, None)

        self.assertEqual(fake.closed, 0)
        self.assertFalse(score.entered)

    def test_scoped_use_and_call_helpers_ignore_unrelated_base(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        score = scenario._core
        core = lock._core
        thread = threading.current_thread()

        unrelated = core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        parent = core.methods[lock.locked](
            lock.locked, time.monotonic(), regulated=False)
        child = core.methods[lock.acquire](
            lock.acquire, time.monotonic(), regulated=False)
        child.parent = parent

        score.scoped_use_bases[(thread, lock)][unrelated] += 1
        self.assertEqual(score.use_signals(thread, lock, child), {Use(thread, lock)})

        score.scoped_call_bases[(thread, lock.acquire)][unrelated] += 1
        self.assertEqual(
            child.call_signals(State.BLOCKED),
            {Call(thread, lock.acquire, State.BLOCKED)},
        )

    def test_transaction_committed_and_aborted_already_raised_edges(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core

        tx = core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        tx.state = State.COMMITTED
        tx.raised = False
        tx.timed_out = False
        self.assertIs(tx.committed(), State.RETURNED)
        self.assertIs(tx.state, State.COMMITTED)

        error = ValueError("already raised")
        tx = core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        tx.state = State.BLOCKED
        tx.raised = True
        tx.result = error
        tx.aborted()
        self.assertTrue(tx.done)
        self.assertIs(tx.result, error)
        self.assertIs(tx.state, State.RAISED)

    def test_lock_and_rlock_repr_helpers_with_interjection_and_rlock_count(self):
        scenario = Scenario()
        lock = scenario.Lock()
        self.assertIn("unit ", lock._core.repr_helper("unit"))

        rlock = scenario.RLock()
        rlock.acquire()
        rlock.acquire()
        try:
            self.assertIn("unit ", rlock._core.repr_helper("unit"))
            self.assertEqual(scenario.api(rlock).count, 2)
        finally:
            rlock.release()
            rlock.release()
        self.assertEqual(scenario.api(rlock).count, 0)

    def test_raw_condition_wait_smoke_asserts_timeout_result(self):
        scenario = Scenario()
        condition = scenario.Condition()
        raw = scenario.raw(condition)
        with raw:
            self.assertFalse(raw.wait(timeout=0))

    def test_cycle_base_init_and_check_base_edges(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.CycleBase)
        cycle.__init__(())
        self.assertEqual(cycle.drivers, [])
        self.assertFalse(cycle.closed)
        self.assertEqual(cycle.ready, {})

        class Driverish:
            pass

        d = Driverish()
        d.thread = threading.Thread(target=lambda: None, name="cycle-base")
        d.state = scenario._core.Driver.terminated
        d.terminated = scenario._core.Driver.terminated
        d.impasse = scenario._core.Driver.impasse
        d.base_tx = object()
        with self.assertRaisesRegex(RuntimeError, "base tx ended"):
            cycle.check_base(d, "unit role")

        d.state = scenario._core.Driver.impasse
        with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
            cycle.check_base(d, "unit role")

    def test_cycle_base_wake_drivers_pause_partial_and_raised(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.CycleBase)
        cycle.core = condition._core
        cycle.closed = False
        cycle.ready = {}

        class Tx:
            def __init__(self):
                self.pause = False
                self.scheduler_pause = False
                self.blanket_pause = True
                self.state = State.PAUSED
                self.result = None

        class Driverish(FakeDriver):
            def __init__(self, score, name, state=State.RETURNED, result=None):
                super().__init__(score, name)
                self.thread = threading.Thread(target=lambda: None, name=name)
                self.tx = Tx()
                self.tx.state = state
                self.tx.result = result
                self.finish_calls = 0
            def finish(self):
                self.finish_calls += 1
        d1 = Driverish(scenario._core, "pause-one")
        d2 = Driverish(scenario._core, "pause-two")
        cycle.ready = {d1.thread: d1, d2.thread: d2}
        self.assertEqual(cycle.wake_drivers([d1], pause=True), [d1])
        self.assertTrue(d1.tx.pause)
        self.assertFalse(d1.tx.blanket_pause)
        self.assertFalse(cycle.closed)
        self.assertIn(d2.thread, cycle.ready)
        self.assertEqual(cycle.wake_drivers([d2], pause=True), [d2])
        self.assertTrue(cycle.closed)

        error = RuntimeError("cycle driver raised")
        d1 = Driverish(scenario._core, "raised-one", State.RAISED, error)
        d2 = Driverish(scenario._core, "raised-two")
        cycle.ready = {d1.thread: d1, d2.thread: d2}
        cycle.closed = False
        with self.assertRaisesRegex(RuntimeError, "cycle driver raised"):
            cycle.wake_drivers([d1, d2])
        self.assertEqual(d1.finish_calls, 1)
        self.assertEqual(d2.finish_calls, 1)
        self.assertTrue(cycle.closed)


    def test_barrier_cycle_rejects_non_thread_before_count_check(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            barrier._core.Cycle((object(),))

    def test_condition_wait_for_transaction_delegate_predicate_child(self):
        import time
        scenario = Scenario()
        condition = scenario.Condition()
        core = condition._core
        wait_for_tx = core.methods[condition.wait_for](
            condition.wait_for, time.monotonic(), regulated=False,
            predicate=lambda: False, timeout=None)
        wait_child = core.methods[condition.wait](
            condition.wait, time.monotonic(), regulated=False, timeout=None)
        wait_child.parent = wait_for_tx
        self.assertTrue(wait_for_tx.is_delegate(wait_child))
        self.assertFalse(wait_for_tx.is_delegate(object()))


class TestAdditionalPrimitiveCoverageEdges2(unittest.TestCase):
    def test_parse_park_skip_rejects_foreign_bound_method(self):
        score = Scenario()._core
        thread = threading.Thread(target=lambda: None, name="foreign-method")
        class Foreign:
            def method(self):
                pass
        self.assertIsNone(Foreign().method())
        with self.assertRaisesRegex(ValueError, "isn't a regulated method call"):
            score.parse_park_skip_args((thread, Foreign().method), "skip")

    def test_dispatch_discard_and_close_unowned_promoted_edges(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        chain = score.Chain()
        orphan = FakeDriver(score, "orphan-promoted")
        chain.register(dispatch)
        dispatch.driver_to_chain[orphan] = chain
        dispatch.discard(chain)
        self.assertIsNone(chain.owner)
        self.assertNotIn(orphan, dispatch.driver_to_chain)

        dispatch = score.Dispatch()
        already_listed = score.Chain()
        already_listed.register(dispatch)
        dispatch.recent.append(already_listed)
        dispatch.driver_to_chain[FakeDriver(score, "dup-chain")] = already_listed
        unowned_driver = FakeDriver(score, "unowned")
        dispatch.queue.append(unowned_driver)
        dispatch.close()
        self.assertEqual(unowned_driver.closed, 1)
        self.assertIsNone(unowned_driver.owner)
        self.assertFalse(dispatch.driver_to_chain)

    def test_transaction_scheduler_pause_clear_from_paused_unparks(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        tx.state = State.PAUSED
        tx.scheduler_pause = True
        tx.settle = lambda: None
        blocker = threading.Lock()
        blocker.acquire()
        tx.blocker = blocker
        tx.set_scheduler_pause(False)
        self.assertFalse(tx.scheduler_pause)
        self.assertIs(tx.state, State.EXITING)
        self.assertIsNone(tx.blocker)
        self.assertTrue(blocker.acquire(blocking=False))

    def test_assign_with_no_threads_is_rejected(self):
        scenario = Scenario()
        lock = scenario.Lock()
        with self.assertRaisesRegex(ValueError, "no thread specified"):
            scenario.api(lock).assign()

    def test_cycle_base_check_base_terminated_without_base(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.CycleBase)
        cycle.__init__(())

        class Driverish:
            pass

        d = Driverish()
        d.thread = threading.Thread(target=lambda: None, name="cycle-no-base")
        d.state = scenario._core.Driver.terminated
        d.terminated = scenario._core.Driver.terminated
        d.impasse = scenario._core.Driver.impasse
        d.base_tx = None
        with self.assertRaisesRegex(RuntimeError, "terminated before reaching"):
            cycle.check_base(d, "unit role")

    def test_condition_cycle_process_next_thread_and_repr_edges(self):
        scenario = Scenario()
        condition = scenario.Condition()
        cycle = object.__new__(condition._core.Cycle)
        cycle.core = condition._core
        cycle.incoming = []
        cycle.waiting = []
        cycle.ready = []
        cycle.closed = False
        cycle.process()
        self.assertEqual(cycle.ready, [])

        processed = []
        t = threading.Thread(target=lambda: None, name="next-process")
        d = types.SimpleNamespace(thread=t)
        cycle.incoming = [d]
        cycle.ready = []
        cycle.closed = False
        def process():
            processed.append(True)
            cycle.ready.append(d)
            cycle.incoming.clear()
        def act(threads, verb):
            self.assertEqual(threads, ())
            self.assertEqual(verb, "wake")
            cycle.ready.clear()
            cycle.closed = True
            return t
        cycle.process = process
        cycle.act = act
        self.assertIs(cycle.next_thread(), t)
        self.assertEqual(processed, [True])

        cycle.closed = False
        cycle.ready = [d]
        self.assertEqual(cycle.repr(), "<Condition.cycle 1 ready>")
        cycle.closed = True
        self.assertEqual(cycle.repr(), "<Condition.cycle closed>")

    def test_lock_at_fork_reinit_tolerates_actual_without_hook(self):
        scenario = Scenario()
        lock = scenario.Lock()
        original = lock._core.actual
        try:
            lock._core.actual = object()
            self.assertIsNone(lock._at_fork_reinit())
        finally:
            lock._core.actual = original

    def test_lock_and_rlock_repr_helpers_without_interjection(self):
        scenario = Scenario()
        lock = scenario.Lock()
        self.assertIn("unlocked", lock._core.repr_helper())
        rlock = scenario.RLock()
        self.assertIn("unlocked", rlock._core.repr_helper())


class TestAdditionalPrimitiveCoverageEdges3(unittest.TestCase):
    def test_core_unpause_leaves_unpaused_paused_tx_alone(self):
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core
        thread = threading.Thread(target=lambda: None, name="unpause-false")
        calls = []

        class Tx:
            pause = False
            def validate(self, **kwargs):
                calls.append(("validate", kwargs))
            def unpause(self):
                calls.append(("unpause",))

        tx = Tx()
        tx.unpause()
        self.assertEqual(calls, [("unpause",)])
        calls.clear()
        original_threads_to_txs = core.threads_to_txs
        original_settle = core.settle
        try:
            core.threads_to_txs = lambda threads, caller: ((thread,), [tx])
            core.settle = lambda txs: calls.append(("settle", tuple(txs)))
            self.assertEqual(core.unpause(lock.acquire, (thread,)), (thread,))
        finally:
            core.threads_to_txs = original_threads_to_txs
            core.settle = original_settle

        self.assertEqual(calls[0][0], "validate")
        self.assertEqual(calls[1], ("settle", (tx,)))
        self.assertNotIn(("unpause",), calls)

    def test_semaphore_api_base_repr_is_usable(self):
        scenario = Scenario()
        semaphore = scenario.Semaphore(2)
        base_api = semaphore._core.SemaphoreAPIBase(False)
        self.assertIn("SemaphoreAPI", repr(base_api))
        self.assertEqual(base_api.value, 2)
        self.assertEqual(base_api.available, 2)


class TestScoreMonitorDefensiveCleanup(unittest.TestCase):
    def test_monitor_aborts_stuck_transaction_after_thread_exits(self):
        scenario = Scenario()
        score = scenario._core
        thread = threading.Thread(target=lambda: None, name="monitor-stuck")
        calls = []

        class Tx:
            def aborted(self):
                calls.append("aborted")

        thread.start()
        thread.join()
        score.monitors[thread] = threading.current_thread()
        score.managed[thread] = None
        score.transactions[thread] = Tx()

        score.monitor(thread)

        self.assertEqual(calls, ["aborted"])
        self.assertNotIn(thread, score.monitors)
        self.assertNotIn(thread, score.managed)


class TestDriverStateMachineDirectCoverage(unittest.TestCase):
    def make_driver_with_tx(self, *, base=False, tx_state=State.BLOCKED,
                            primitive='lock'):
        import time
        scenario = Scenario()
        score = scenario._core
        if primitive == 'condition':
            primitive_obj = scenario.Condition()
            core = primitive_obj._core
            method = primitive_obj.wait
            tx = core.methods[primitive_obj.wait](
                primitive_obj.wait, time.monotonic(), regulated=False)
        else:
            primitive_obj = scenario.Lock()
            core = primitive_obj._core
            method = primitive_obj.acquire
            tx = core.methods[primitive_obj.acquire](
                primitive_obj.acquire, time.monotonic(), regulated=False)
        thread = threading.Thread(target=lambda: None, name="driver-direct")
        parent = None
        if base:
            parent = core.methods[getattr(primitive_obj, 'locked', method)](
                getattr(primitive_obj, 'locked', method),
                time.monotonic(), regulated=False)
            parent.thread = thread
            parent.state = State.WAITING
            tx.parent = parent
            parent.child = tx
        tx.thread = thread
        tx.state = tx_state
        score.transactions[thread] = tx
        score.transaction_apis[thread] = tx.api
        score.entered = True
        driver = score.Driver(thread, parent if parent is not None else None)
        return scenario, primitive_obj, thread, parent, tx, driver

    def cleanup_driver_fixture(self, scenario, thread):
        scenario._core.transactions.pop(thread, None)
        scenario._core.transaction_apis.pop(thread, None)
        scenario._core.drivers.pop(thread, None)
        scenario._core.entered = False

    def test_scan_is_explicit_and_published(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            self.assertIs(driver.state, driver.undirected)
            self.assertIsNone(driver.tx)
            driver.scan()
            self.assertEqual(driver.directive, driver.scan)
            self.assertEqual(driver.directive_args, (None,))
            driver.drive()
            self.assertIs(driver.state, driver.success)
            self.assertIs(driver.tx, tx)
            self.assertIs(driver.snapshot_tx, tx)
            self.assertIs(driver.snapshot_state, tx.state)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_empty_slot_drive_and_proceed_raise_undirected(self):
        score = Scenario()._core
        score.entered = True
        driver = score.Driver(threading.Thread(target=lambda: None,
                                               name="empty-direct"))
        with self.assertRaisesRegex(RuntimeError, "no Driver directive staged"):
            driver.drive()
        self.assertIs(driver.state, driver.undirected)
        with self.assertRaisesRegex(RuntimeError, "no Driver directive staged"):
            driver.proceed()
        self.assertIs(driver.state, driver.undirected)

    def test_scan_target_scope_validation_and_recovery_from_mutated(self):
        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            foreign = threading.Thread(target=lambda: None, name="foreign")
            tx.thread = foreign
            with self.assertRaisesRegex(ValueError, "belongs to thread"):
                driver.scan(tx)
            tx.thread = thread
            with self.assertRaisesRegex(ValueError, "base tx"):
                driver.scan(base_tx)
            tx.state = State.RETURNED
            with self.assertRaisesRegex(ValueError, "scan target must be active"):
                driver.scan(tx)
            tx.state = State.BLOCKED
            driver.state = driver.mutated
            driver.scan(tx)
            self.assertEqual(driver.directive, driver.scan)
            self.assertEqual(driver.directive_args, (tx,))
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_constructor_rejects_off_thread_and_done_base(self):
        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            other_thread = threading.Thread(target=lambda: None, name="other")
            with self.assertRaisesRegex(ValueError, "belongs to thread"):
                scenario._core.Driver(other_thread, base_tx)
            base_tx.state = State.RETURNED
            with self.assertRaisesRegex(ValueError, "base tx must be active"):
                scenario._core.Driver(thread, base_tx)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_wait_signal_resolution_and_unlisted_tx_exit_error(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.wait(tx.api)
            driver.drive()
            driver.signal({tx.api})
            self.assertIs(driver.state, driver.success)
            self.assertEqual(driver.signaled, frozenset({tx.api}))
            self.assertEqual(driver.motivation, frozenset({tx.api}))

            tx.state = State.BLOCKED
            score = scenario._core
            score.transactions[thread] = tx
            score.transaction_apis[thread] = tx.api
            driver.select_tx(tx)
            driver.wait(_ProbeSignal("never", False))
            driver.drive()
            tx.state = State.RETURNED
            with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                driver.signal({tx})
            self.assertIs(driver.state, driver.driving)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_blanket_pause_and_transaction_handoff_edges(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            self.assertTrue(driver.arm_blanket_pause())
            self.assertTrue(tx.blanket_pause)
            self.assertTrue(driver.owns_pause(tx))
            self.assertTrue(driver.release_blanket_pause(tx))
            self.assertFalse(tx.blanket_pause)
            self.assertFalse(driver.owns_pause(tx))
            self.assertTrue(driver.release_blanket_pause(tx))

            self.assertTrue(driver.arm_blanket_pause(tx))
            self.assertTrue(driver.handoff_blanket_pause_to_scheduler_pause(tx))
            self.assertTrue(tx.pause)
            self.assertFalse(tx.blanket_pause)
            tx.set_scheduler_pause(False)
            driver.set_owns_pause(tx, True)
            tx.blanket_pause = True
            tx.state = State.EXITING
            with self.assertRaisesRegex(RuntimeError, "already advanced past PAUSED"):
                driver.handoff_blanket_pause_to_scheduler_pause(tx)
            tx.state = State.PAUSED
            tx.blanket_pause = False
            with self.assertRaisesRegex(RuntimeError, "no blanket pause"):
                driver.handoff_blanket_pause_to_scheduler_pause(tx)
            driver.set_owns_pause(tx, False)
            with self.assertRaisesRegex(RuntimeError, "does not own"):
                driver.handoff_blanket_pause_to_scheduler_pause(tx)

            fresh = scenario._core.Driver(threading.Thread(target=lambda: None,
                                                          name="no-internal-pause-tx"))
            with self.assertRaisesRegex(RuntimeError, "no active transaction"):
                fresh.arm_blanket_pause()
            tx.state = State.EXITING
            with self.assertRaisesRegex(RuntimeError, "already passed PAUSED"):
                driver.arm_blanket_pause(tx)
        finally:
            self.cleanup_driver_fixture(scenario, thread)
    def test_blanket_pause_two_flag_edge_coverage(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            self.assertTrue(driver.arm_blanket_pause(tx))
            with self.assertRaisesRegex(RuntimeError, "already set"):
                driver.arm_blanket_pause(tx)
            self.assertTrue(driver.release_blanket_pause(tx))

            tx.blanket_pause = True
            self.assertFalse(driver.arm_blanket_pause(tx))
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            with self.assertRaisesRegex(RuntimeError, "no active transaction"):
                scenario._core.Driver(threading.Thread(target=lambda: None)).handoff_blanket_pause_to_scheduler_pause()
            tx.blanket_pause = True
            self.assertFalse(driver.handoff_blanket_pause_to_scheduler_pause(tx))
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            tx.blanket_pause = True
            self.assertFalse(driver.release_blanket_pause(tx))
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)


    def test_auto_release_paused_skips_unpark_if_release_already_moved_tx(self):
        fixture = self.make_driver_with_tx(tx_state=State.PAUSED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            self.assertIs(tx.state, State.PAUSED)
            def release_blanket_pause(moved_tx):
                self.assertIs(moved_tx, tx)
                tx.state = State.EXITING
                return True
            driver.release_blanket_pause = release_blanket_pause
            driver.auto_release_current_state()
            self.assertIs(tx.state, State.EXITING)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_blanket_pause_ownership_identity_edges(self):
        score = Scenario()._core
        driver = score.Driver(threading.Thread(target=lambda: None, name="pause-ownership-edges"))

        parent = types.SimpleNamespace(parent=None, blanket_pause=False, scheduler_pause=False,
                                       state=State.BLOCKED)
        child = types.SimpleNamespace(parent=parent, blanket_pause=False, scheduler_pause=False,
                                      state=State.BLOCKED)

        self.assertFalse(driver.owns_pause())
        driver.tx = parent
        driver.set_owns_pause(parent, True)
        self.assertTrue(driver.owns_pause())
        self.assertTrue(driver.owns_pause(parent))
        self.assertFalse(driver.owns_pause(child))
        self.assertIs(driver.owned_pause_tx, parent)

        driver.tx = child
        self.assertFalse(driver.owns_pause())
        self.assertTrue(driver.owns_pause(parent))
        driver.set_owns_pause(child, False)
        self.assertIs(driver.owned_pause_tx, parent)
        driver.set_owns_pause(parent, False)
        self.assertFalse(driver.owns_pause(parent))
        self.assertIsNone(driver.owned_pause_tx)

        class ParkedTx:
            state = State.PAUSED
            scheduler_pause = False
            blanket_pause = False
            def release_blanket_pause(self):
                self.blanket_pause = False
            def unpark(self, state):
                self.state = state

        parked = ParkedTx()
        parked.blanket_pause = True
        driver.tx = parked
        driver.set_owns_pause(parked, True)
        driver.auto_release_current_state()
        self.assertIs(parked.state, State.EXITING)

    def test_blanket_pause_drive_return_edges(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            tx.blanket_pause = True
            driver.finish(); driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            tx.blanket_pause = True
            driver.until(driver.terminated); driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            original = driver.arm_blanket_pause
            driver.arm_blanket_pause = lambda tx=None: False
            driver.pause_internal(); driver.drive()
            self.assertIs(driver.state, driver.driving)
            driver.arm_blanket_pause = original
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_transaction_blanket_pause_release_direct_edges(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        tx.state = State.PAUSED
        blocker = threading.Lock(); blocker.acquire()
        tx.blocker = blocker
        tx.settle = lambda: None
        tx.blanket_pause = True
        with scenario._core.lock:
            tx.release_blanket_pause()
        self.assertFalse(tx.blanket_pause)
        self.assertIs(tx.state, State.EXITING)
        self.assertTrue(blocker.acquire(blocking=False))

    def test_blanket_pause_remaining_coverage_edges(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            tx.state = State.PAUSED
            tx.blanket_pause = True
            driver.auto_release_current_state()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            tx.scheduler_pause = True
            tx.state = State.EXITING
            driver.scan(); driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            tx.blanket_pause = True
            driver.block(); driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_transaction_release_blanket_pause_edge_coverage(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        tx.release_blanket_pause()  # no blanket pause: no-op branch
        self.assertFalse(tx.blanket_pause)

        tx.state = State.PAUSED
        tx.scheduler_pause = True
        tx.blanket_pause = True
        tx.release_blanket_pause()
        self.assertFalse(tx.blanket_pause)
        self.assertIs(tx.state, State.PAUSED)

    def test_wait_sets_callback_signal_for_observed_callback_edges(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            predicate = Predicate(tx.api)
            driver.wait(predicate)
            driver.drive()
            driver.signal({predicate})
            self.assertIs(driver.callback_signal, driver.thread_signal[Predicate])
            self.assertEqual(driver.motivation, frozenset({predicate}))

            action = Action(tx.api)
            driver.wait(action)
            driver.drive()
            driver.signal({action})
            self.assertIs(driver.callback_signal, driver.thread_signal[Action])
            self.assertEqual(driver.motivation, frozenset({action}))
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_reenter_prearmed_blanket_pause_can_stop_at_paused(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            self.assertTrue(driver.arm_blanket_pause(tx))
            driver.reenter()
            driver.drive()
            tx.state = State.PAUSED
            paused = primitives_module.Paused(tx.api)
            driver.signal({paused})
            self.assertIs(driver.state, driver.success)
            self.assertEqual(driver.motivation, frozenset({paused}))
            self.assertTrue(tx.blanket_pause)
            tx.scheduler_pause = True
            driver.release_blanket_pause(tx)
            tx.scheduler_pause = False
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_reenter_resume_and_until_stage_edges(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.reenter()
            self.assertEqual(driver.directive, driver.reenter)
            driver.drive()
            self.assertIs(driver.state, driver.driving)
            self.assertIn(driver.thread_signal[Predicate], driver.signals)
            callback = driver.thread_signal[Predicate]
            driver.signal({callback})
            self.assertIs(driver.state, driver.success)
            self.assertIs(driver.callback_signal, callback)
            driver.resume()
            self.assertEqual(driver.directive, driver.resume)
            driver.drive()
            self.assertIn(Not(callback), driver.wait_unbox)
            driver.signal({Not(callback)})
            self.assertIs(driver.state, driver.success)
            self.assertIsNone(driver.callback_signal)

            with self.assertRaisesRegex(RuntimeError, r"until\(impasse\) requires"):
                driver.until(driver.impasse)
            with self.assertRaisesRegex(ValueError, r"until\(\) target"):
                driver.until(driver.success)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_mutated_and_overshot_drive_start_detection(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.pause()
            tx.state = State.STALLED
            driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.COMMIT,
                                           primitive='condition')
        scenario, cond, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.waiting()
            tx.state = State.STALLED
            driver.drive()
            self.assertIs(driver.state, driver.overshot)
            self.assertIs(driver.target, State.WAITING)
        finally:
            self.cleanup_driver_fixture(scenario, thread)


    def test_driver_helper_edges_and_stage_time_guards(self):
        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            # Low-level helpers keep a truthful model of the selected tx.
            self.assertIs(driver.live_child_of(None), None)
            driver.select_tx(None)
            self.assertIsNone(driver.tx)
            self.assertIs(driver.thread_signal, driver.base_thread_signal)
            self.assertFalse(driver.live_base_blocks())

            with self.assertRaisesRegex(TypeError, "scan target"):
                driver.validate_scan_target(object())
            tx.parent = None
            with self.assertRaisesRegex(ValueError, "strictly inside"):
                driver.validate_scan_target(tx)
            tx.parent = base_tx

            # Dead / mutated states, base impasses, early mutation, and
            # known-behind targets all fail at directive stage time.
            driver.state = driver.raised
            with self.assertRaisesRegex(RuntimeError, "currently"):
                driver.scan()
            driver.state = driver.mutated
            with self.assertRaisesRegex(RuntimeError, "scan first"):
                driver.wait(_ProbeSignal("mutated-wait", False))
            driver.state = driver.success

            base_tx.state = State.RETURNED
            with self.assertRaisesRegex(RuntimeError, "currently done"):
                driver.scan()
            base_tx.state = State.PAUSED
            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                driver.scan()
            base_tx.state = State.WAITING

            driver.select_tx(tx)
            driver.snapshot_tx = tx
            driver.snapshot_state = State.BLOCKED
            tx.state = State.STALLED
            with self.assertRaisesRegex(RuntimeError, "model is mutated"):
                driver.finish()

            driver.snapshot_state = State.PAUSED
            tx.state = State.PAUSED
            with self.assertRaisesRegex(RuntimeError, "target BLOCKED is behind"):
                driver.block()

            driver.snapshot_state = State.COMMITTED
            tx.state = State.COMMITTED
            with self.assertRaisesRegex(RuntimeError, "currently in COMMITTED"):
                driver.pause()
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_prepare_drive_start_and_release_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.select_tx(tx)
            driver.drive_tx = tx
            tx.state = State.RETURNED
            self.assertTrue(driver.prepare_drive_start())

            tx.state = State.STALLED
            driver.snapshot_tx = tx
            driver.snapshot_state = State.STALLED
            driver.target = State.PAUSED
            tx.state = State.EXITING
            self.assertFalse(driver.prepare_drive_start())
            self.assertIs(driver.state, driver.mutated)

            driver.select_tx(tx)
            driver.snapshot_tx = None
            self.assertTrue(driver.prepare_drive_start())

            with self.assertRaisesRegex(RuntimeError, "can't release"):
                driver.auto_release_current_state()

            other_tx = tx
            driver.select_tx(None)
            ts = driver.make_tx_signals(other_tx)
            self.assertIs(driver.tx, other_tx)
            self.assertIn(State.BLOCKED, ts)
            self.assertIs(driver.make_tx_signals(other_tx), ts)

            self.assertTrue(driver.release_blanket_pause())

            driver.route_iterator = object()
            driver.close(route=False)
            self.assertIsNotNone(driver.route_iterator)
            driver.clear_route()
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, child, driver = fixture
        try:
            driver.select_tx(base_tx)
            driver.drive_tx = base_tx
            self.assertFalse(driver.prepare_drive_start())
            self.assertIs(driver.tx, child)
            self.assertIs(driver.state, driver.nested)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_scan_wait_and_impasse_edges(self):
        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, child, driver = fixture
        try:
            # Scoped scan with no visible child waits on Nested(base_tx).
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.base_thread_signal[Terminated] = _ProbeSignal("base-alive", False)
            driver.scan()
            driver.drive()
            self.assertIn(Nested(base_tx.api), driver.signals)

            # If the base goes blanket-parked before commit, scan rests at impasse.
            driver.rest(driver.success)
            base_tx.state = State.WAITING
            driver.scan()
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.impasse)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.base_thread_signal[Terminated] = _ProbeSignal("watch-alive", False)
            driver.scan(tx)
            driver.drive()
            self.assertIn(Nested(tx.api), driver.signals)
            driver.signal({driver.base_thread_signal[Terminated]})
            self.assertIs(driver.state, driver.terminated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_until_and_signal_resolution_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.finish()
            driver.drive()
            self.assertNotIn(driver.thread_signal[Predicate], driver.signals)
            self.assertNotIn(driver.thread_signal[Action], driver.signals)
            tx.state = State.RETURNED
            driver.signal({tx})
            self.assertIs(driver.state, driver.success)

            # The immediate-done configure path resolves without waiting.
            tx.state = State.RETURNED
            driver.configure_finish_wait(tx)
            self.assertIs(driver.state, driver.success)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.until(driver.raised)
            driver.drive()
            self.assertNotIn(driver.thread_signal[Predicate], driver.signals)
            self.assertNotIn(driver.thread_signal[Action], driver.signals)
            tx.state = State.RETURNED
            driver.signal({tx})
            self.assertIs(driver.state, driver.returned)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.until(driver.terminated)
            driver.drive()
            self.assertEqual(driver.drive_kind, 'until-terminated-finish')
            tx.state = State.RETURNED
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.base_thread_signal[Terminated] = _ProbeSignal("until-alive", False)
            driver.signal({tx})
            self.assertEqual(driver.drive_kind, 'until-terminated-watch')
            driver.signal({driver.base_thread_signal[Terminated]})
            self.assertIs(driver.state, driver.success)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.until(driver.impasse)
            driver.drive()
            self.assertIn(base_tx.api, driver.signals)
            base_tx.state = State.PAUSED
            driver.signal({base_tx.api})
            self.assertIs(driver.state, driver.success)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_callback_wait_route_and_signal_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            with self.assertRaisesRegex(RuntimeError, "not in a callback"):
                driver.resume()

            driver.reenter(); driver.drive()
            action = driver.thread_signal[Action]
            driver.signal({action})
            self.assertIs(driver.state, driver.success)
            self.assertEqual(driver.callback_signal, action)
            driver.resume(); driver.drive()
            self.assertIn(Not(action), driver.wait_unbox)
            driver.signal({Not(action)})
            self.assertIs(driver.state, driver.success)
            self.assertIsNone(driver.callback_signal)

            # A wait may observe a callback edge whose tx is not currently
            # selected; in that case Driver preserves the exact signal.
            driver.tx = None
            other_predicate = Predicate(tx.api)
            driver.wait(other_predicate)
            driver.drive()
            driver.signal({other_predicate})
            self.assertIs(driver.callback_signal, other_predicate)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            # wait() with no current tx is passive: it watches only the
            # asserted signals plus Terminated(thread), not scoped children.
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.wait(_ProbeSignal("scoped-wait", False))
            driver.drive()
            self.assertNotIn(Nested(base_tx.api), driver.signals)
            self.assertIn(driver.base_thread_signal[Terminated], driver.signals)

            driver.callback_signal = _ProbeSignal("callback-no-live", True)
            driver.tx = None
            driver.resume()
            driver.drive()
            self.assertNotIn(base_tx.api, driver.signals)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            # A wait that starts with no selected tx remains passive; it
            # does not reselect the live scoped tx.
            driver.tx = None
            driver.wait(_ProbeSignal("reselect", False))
            driver.drive()
            with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                driver.signal({_ProbeSignal("unexpected", True)})
            self.assertIsNone(driver.tx)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_route_install_and_public_until_wrapper(self):
        scenario = Scenario()
        ev = scenario.Event()
        seen = []
        def worker():
            ev.wait()
        def route_a(d):
            d.route(route_b)
            return
            yield
        def route_b(d):
            seen.append('b')
            return
            yield
        with scenario:
            t = scenario.thread(worker)
            scenario.wait(t)
            d = scenario.Driver(t)
            d.route(route_a)
            d()
            self.assertEqual(seen, ['b'])
            self.assertFalse(d.routed)
            d.close()
            scenario.raw(ev).set()

        score = Scenario()._core
        score.entered = True
        core_driver = score.Driver(threading.Thread(target=lambda: None,
                                                    name="route-direct"))
        core_driver.closure = lambda: None
        self.assertTrue(core_driver.resume_route())

        wrapper = Scenario().Driver(threading.Thread(target=lambda: None,
                                                    name="public-until"))
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            wrapper.until(wrapper.terminated)


    def test_driver_remaining_direct_resolution_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            self.assertFalse(driver.live_base_blocks())

            driver.select_tx(tx)
            driver.drive_tx = tx
            driver.snapshot_tx = tx
            driver.snapshot_state = State.COMMITTED
            driver.target = State.PAUSED
            tx.state = State.EXITING
            self.assertFalse(driver.prepare_drive_start())
            self.assertIs(driver.state, driver.mutated)

            driver.select_tx(tx)
            scenario._core.transactions[thread] = tx
            scenario._core.transaction_apis[thread] = tx.api
            self.assertFalse(driver.surface_nested_child())
            plain_signal = object()
            self.assertTrue(driver.signal_is_high(plain_signal, {plain_signal}))
            self.assertFalse(driver.signal_is_high(plain_signal, set()))
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True, tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.pause()
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.impasse)

            base_tx.state = State.WAITING
            driver.scan(); driver.drive()
            driver.finish()
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.impasse)

            base_tx.state = State.WAITING
            driver.scan(); driver.drive()
            driver.finish()
            tx.state = State.STALLED
            driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_until_configure_edges(self):
        fixture = self.make_driver_with_tx(base=True, tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.until(driver.impasse)
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.success)

            base_tx.state = State.WAITING
            driver.scan(); driver.drive()
            driver.until(driver.terminated)
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.impasse)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.configure_until_terminated_wait(None)
            self.assertIs(driver.state, driver.success)

            driver.base_thread_signal[Terminated] = _ProbeSignal("alive", False)
            scenario._core.transactions[thread] = tx
            scenario._core.transaction_apis[thread] = tx.api
            driver.configure_until_terminated_wait(None)
            self.assertIs(driver.state, driver.persisted)

            tx.state = State.RETURNED
            driver.configure_until_raised_wait(tx)
            self.assertIs(driver.state, driver.returned)

            tx.state = State.BLOCKED
            driver.configure_until_raised_wait(tx)
            self.assertNotIn(driver.thread_signal[Predicate], driver.signals)
            self.assertNotIn(driver.thread_signal[Action], driver.signals)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True, tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.base_thread_signal[Terminated] = _ProbeSignal("impasse-wait", False)
            driver.configure_until_impasse_wait()
            self.assertIn(base_tx.api, driver.signals)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_reenter_resume_and_scan_signal_edges(self):
        fixture = self.make_driver_with_tx(base=True, tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan(); driver.drive()
            driver.reenter()
            base_tx.state = State.PAUSED
            driver.drive()
            self.assertIs(driver.state, driver.impasse)

            base_tx.state = State.WAITING
            driver.scan(); driver.drive()
            driver.reenter()
            tx.state = State.STALLED
            driver.drive()
            self.assertIs(driver.state, driver.mutated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            signal = _ProbeSignal("callback", True)
            driver.callback_signal = signal
            driver.tx = None
            driver.resume()
            driver.drive()
            self.assertIsNone(driver.tx)
            self.assertNotIn(tx, driver.signals)
            self.assertIn(Not(signal), driver.signals)

            driver.drive_kind = 'reenter'
            driver.select_tx(tx)
            term = driver.base_thread_signal[Terminated]
            driver.signal_reenter({term})
            self.assertIs(driver.state, driver.terminated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(base=True, tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.scan_target = tx
            tx.state = State.RETURNED
            driver.signal_scan(set())
            self.assertIs(driver.state, driver.impasse)

            tx.state = State.BLOCKED
            driver.scan_target = None
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            base_tx.state = State.PAUSED
            driver.signal_scan(set())
            self.assertIs(driver.state, driver.impasse)

            base_tx.state = State.WAITING
            driver.base_thread_signal[Terminated] = _ProbeSignal("scan-alive", False)
            driver.signal_scan(set())
            self.assertIs(driver.state, driver.driving)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_wait_and_tx_signal_resolution_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            never = _ProbeSignal("never", False)
            driver.wait(never)
            driver.drive()
            driver.tx = None
            with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                driver.signal_wait(set())
            self.assertIsNone(driver.tx)
            self.assertIs(driver.state, driver.driving)

            tx.state = State.RETURNED
            driver.select_tx(tx)
            driver.drive_kind = 'wait'
            driver.wait_keys = frozenset()
            driver.signals = frozenset()
            with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                driver.signal_wait(set())

            driver.tx = None
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.drive_kind = 'wait'
            with self.assertRaisesRegex(RuntimeError, "unexpected signal"):
                driver.signal_wait(set())

            tx.state = State.BLOCKED
            scenario._core.transactions[thread] = tx
            scenario._core.transaction_apis[thread] = tx.api
            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.PAUSED
            driver.success_signal = _ProbeSignal("nested-false", False)
            driver.state = driver.driving
            driver.signal_tx_drive({driver.thread_signal[Nested]})
            self.assertIs(driver.state, driver.driving)

            driver.select_tx(tx)
            driver.drive_kind = 'until-terminated-watch'
            driver.signal_until_terminated(set())
            self.assertIs(driver.state, driver.persisted)

            driver.drive_kind = 'until-terminated-watch'
            scenario._core.transactions.pop(thread, None)
            scenario._core.transaction_apis.pop(thread, None)
            driver.base_thread_signal[Terminated] = _ProbeSignal("alive-watch", False)
            driver.signal_until_terminated(set())
            self.assertIs(driver.state, driver.driving)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

        fixture = self.make_driver_with_tx(tx_state=State.BLOCKED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.select_tx(tx)
            driver.drive_kind = 'until-terminated-finish'
            tx.state = State.RAISED
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.raised)

            tx.state = State.RETURNED
            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.WAITING
            driver.success_signal = _ProbeSignal("success", True)
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.success)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.WAITING
            driver.success_signal = _ProbeSignal("no-success", False)
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.overshot)

            driver.select_tx(tx)
            driver.drive_kind = 'reenter'
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.success)

            tx.state = State.RAISED
            driver.select_tx(tx)
            driver.drive_kind = 'resume'
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.raised)

            tx.state = State.RETURNED
            driver.select_tx(tx)
            driver.drive_kind = 'until'
            driver.target = driver.terminated
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.success)

            driver.select_tx(tx)
            driver.drive_kind = 'other'
            driver.signal_tx_drive({tx})
            self.assertIs(driver.state, driver.success)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_park_and_impasse_signal_resolution_edges(self):
        fixture = self.make_driver_with_tx(tx_state=State.STALLED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.PAUSED
            driver.success_signal = _ProbeSignal("occupancy", False)
            driver.signals = frozenset({driver.success_signal})
            driver.state = driver.driving
            driver.signal_tx_drive({driver.success_signal})
            self.assertIs(driver.state, driver.driving)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.PAUSED
            driver.success_signal = _ProbeSignal("occupancy-high", True)
            driver.signal_tx_drive(set())
            self.assertIs(driver.state, driver.success)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.STALLED
            driver.success_signal = _ProbeSignal("state-signal", False)
            driver.signal_tx_drive({driver.thread_signal[State.STALLED]})
            self.assertIs(driver.state, driver.success)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.COMMIT
            driver.success_signal = _ProbeSignal("resample", True)
            driver.signal_tx_drive(set())
            self.assertIs(driver.state, driver.success)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            driver.target = State.COMMIT
            driver.success_signal = _ProbeSignal("still-waiting", False)
            driver.signals = frozenset({driver.success_signal})
            returned = driver.signal_tx_drive(set())
            self.assertEqual(returned, driver.signals)

            driver.select_tx(tx)
            driver.drive_kind = 'park'
            term = driver.base_thread_signal[Terminated]
            driver.signal_tx_drive({term})
            self.assertIs(driver.state, driver.terminated)

            driver.select_tx(tx)
            driver.drive_kind = 'until-impasse'
            driver.base_thread_signal[Terminated] = _ProbeSignal("until-impasse-term", True)
            driver.signal_until_impasse({driver.base_thread_signal[Terminated]})
            self.assertIs(driver.state, driver.terminated)

            driver.select_tx(tx)
            driver.drive_kind = 'until-impasse'
            driver.base_thread_signal[Terminated] = _ProbeSignal("until-impasse-live", False)
            returned = driver.signal_until_impasse(set())
            self.assertIs(returned, driver.signals)

            driver.being_driven = True
            with self.assertRaisesRegex(RuntimeError, "already being driven"):
                driver()
        finally:
            self.cleanup_driver_fixture(scenario, thread)


    def test_driver_deep_helpers_handle_children_and_scan_failures(self):
        scenario = Scenario()
        score = scenario._core
        class Tx:
            def __init__(self, state=State.BLOCKED, done=False):
                self.state = state
                self.done = done
        class D:
            success = score.Driver.success
            terminated = score.Driver.terminated
            nested = score.Driver.nested
            nested = score.Driver.nested
            raised = score.Driver.raised
            impasse = score.Driver.impasse
            overshot = score.Driver.overshot
            mutated = score.Driver.mutated
            def __init__(self, tx):
                self.tx = tx
                self.state = self.success
                self.ops = []
                self.child = Tx()
                self.surfaced = False
                self.scan_success = True
            def scan(self):
                self.ops.append('scan')
            def finish(self):
                self.ops.append(('finish', self.tx))
            def pause(self):
                self.ops.append(('pause', self.tx))
            def route(self, route):
                self._route_iterator = route(self)
            def _call_once(self):
                op = self.ops[-1]
                if op == 'scan':
                    if self.scan_success:
                        self.tx = self.root
                        self.state = self.success
                    else:
                        self.state = self.terminated
                    return
                name, tx = op
                self.state = self.success
                if name == 'finish':
                    if tx is self.root:
                        if not self.surfaced:
                            self.surfaced = True
                            self.tx = self.child
                        else:
                            tx.done = True
                            tx.state = State.RETURNED
                    else:
                        tx.done = True
                        tx.state = State.RETURNED
                elif name == 'pause':
                    if tx is self.root and not self.surfaced:
                        self.surfaced = True
                        self.tx = self.child
                    else:
                        tx.state = State.PAUSED
            def __call__(self):
                iterator = getattr(self, '_route_iterator', None)
                if iterator is not None:
                    self._route_iterator = None
                    while True:
                        try:
                            next(iterator)
                        except StopIteration:
                            return
                        self._call_once()
                else:
                    self._call_once()

        probe = D(Tx())
        probe.root = probe.tx
        probe.ops.append(('noop', probe.tx))
        probe()
        self.assertIs(probe.state, probe.success)

        with self.assertRaisesRegex(RuntimeError, "finish_deep requires"):
            score.driver_finish_deep(types.SimpleNamespace(tx=None))
        with self.assertRaisesRegex(RuntimeError, "pause_deep requires"):
            score.driver_pause_deep(types.SimpleNamespace(tx=None))

        root = Tx()
        d = D(root)
        d.root = root
        score.driver_finish_deep(d)
        self.assertTrue(root.done)
        self.assertTrue(d.child.done)
        self.assertIn('scan', d.ops)

        root = Tx()
        d = D(root)
        d.root = root
        d.scan_success = False
        score.driver_finish_deep(d)
        self.assertIn('scan', d.ops)
        self.assertFalse(root.done)

        root = Tx()
        d = D(root)
        d.root = root
        score.driver_pause_deep(d)
        self.assertIs(root.state, State.PAUSED)
        self.assertTrue(d.child.done)

        root = Tx()
        d = D(root)
        d.root = root
        d.scan_success = False
        score.driver_pause_deep(d)
        self.assertIn('scan', d.ops)
        self.assertIs(root.state, State.BLOCKED)

    def test_driver_deep_routes_handle_scan_and_child_branches(self):
        scenario = Scenario()
        score = scenario._core
        class Tx:
            def __init__(self, state=State.BLOCKED, done=False):
                self.state = state
                self.done = done
        class D:
            success = score.Driver.success
            terminated = score.Driver.terminated
            nested = score.Driver.nested
            def __init__(self, tx):
                self.tx = tx
                self.state = self.success
                self.ops = []
            def scan(self):
                self.ops.append('scan')
            def finish(self):
                self.ops.append('finish')
            def pause(self):
                self.ops.append('pause')

        root = Tx()
        child = Tx(done=True)
        d = D(root)
        gen = score.route_finish_deep(d)
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['finish'])
        d.state = d.success
        d.tx = child
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['finish', 'scan'])
        d.state = d.terminated
        with self.assertRaises(StopIteration):
            next(gen)

        root = Tx()
        child = Tx()
        d = D(root)
        gen = score.route_pause_deep(d)
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['pause'])
        d.state = d.success
        d.tx = child
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['pause', 'finish'])
        d.state = d.terminated
        with self.assertRaises(StopIteration):
            next(gen)

        root = Tx(done=True)
        d = D(root)
        gen = score.route_pause_deep(d)
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['scan'])
        d.state = d.success
        d.tx = root
        self.assertEqual(next(gen), None)
        self.assertEqual(d.ops, ['scan', 'pause'])
        d.state = d.terminated
        with self.assertRaises(StopIteration):
            next(gen)

        root = Tx()
        child = Tx(done=True)
        d = D(root)
        gen = score.route_finish_deep(d)
        self.assertEqual(next(gen), None)
        d.state = d.success
        d.tx = child
        self.assertEqual(next(gen), None)
        d.state = d.success
        d.tx = root
        self.assertEqual(next(gen), None)

        root = Tx(done=True)
        d = D(root)
        gen = score.route_pause_deep(d)
        self.assertEqual(next(gen), None)
        d.state = d.terminated
        with self.assertRaises(StopIteration):
            next(gen)

        root = Tx()
        child = Tx()
        d = D(root)
        gen = score.route_pause_deep(d)
        self.assertEqual(next(gen), None)
        d.state = d.success
        d.tx = child
        self.assertEqual(next(gen), None)
        child.done = True
        d.state = d.success
        self.assertEqual(next(gen), None)

        root = Tx()
        d = D(root)
        gen = score.route_pause_deep(d)
        self.assertEqual(next(gen), None)
        d.state = d.terminated
        with self.assertRaises(StopIteration):
            next(gen)

        d = D(None)
        gen = score.route_pause_deep(d)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_public_reenter_resume_wrappers_guard_scenario_entry(self):
        scenario = Scenario()
        thread = threading.Thread(target=lambda: None, name="wrapper-guard")
        driver = scenario.Driver(thread)
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            driver.reenter()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            driver.resume()


class TestLockAssignCoreErrorCoverage(unittest.TestCase):
    class FakeDispatch:
        def __init__(self):
            self.items = []
        def add(self, driver):
            self.items.append(driver)
        def __iter__(self):
            return iter(self.items)

    class FakeAssignDriver:
        def __init__(self, score, thread, state, method, tx_state, *, result=None):
            self.thread = thread
            self.state = state
            self.mode = state
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.success = score.Driver.success
            self.raised = score.Driver.raised
            self.overshot = score.Driver.overshot
            self.mutated = score.Driver.mutated
            self.tx = types.SimpleNamespace(
                method=method,
                state=tx_state,
                result=result,
                done=False,
            )
            self.finish_calls = 0
            self.pause_calls = 0
            self.drive_calls = 0
            self.on_call = None
        def scan(self):
            self.scan_calls = getattr(self, 'scan_calls', 0) + 1
        def finish(self):
            self.finish_calls += 1
            self.tx.done = True
            self.tx.state = State.RETURNED
            self.state = self.success
            self.mode = self.success
        def pause(self):
            self.pause_calls += 1
            self.tx.state = State.PAUSED
            self.state = self.success
            self.mode = self.success
        def route(self, route):
            self._route_iterator = route(self)
        def _call_once(self):
            if self.on_call is not None:
                return self.on_call(self)
            self.drive_calls += 1
        def __call__(self):
            iterator = getattr(self, '_route_iterator', None)
            if iterator is not None:
                self._route_iterator = None
                while True:
                    try:
                        next(iterator)
                    except StopIteration:
                        return
                    self._call_once()
            else:
                self._call_once()

    def with_fake_driver(self, scenario, drivers, func):
        score = scenario._core
        original_driver = score.Driver
        original_dispatch = score.Dispatch
        queue_ = list(drivers)
        dispatches = []
        def make_driver(thread, base=None):
            self.assertTrue(queue_, "unexpected Driver construction")
            driver = queue_.pop(0)
            self.assertIs(driver.thread, thread)
            return driver
        def make_dispatch():
            dispatch = self.FakeDispatch()
            dispatches.append(dispatch)
            return dispatch
        try:
            score.Driver = make_driver
            score.Dispatch = make_dispatch
            return func()
        finally:
            score.Driver = original_driver
            score.Dispatch = original_dispatch

    def test_fake_assign_driver_default_methods(self):
        scenario = Scenario()
        lock = scenario.Lock()
        driver = self.FakeAssignDriver(
            scenario._core, threading.Thread(target=lambda: None),
            scenario._core.Driver.success, lock.acquire, State.BLOCKED)
        driver.pause()
        self.assertEqual(driver.pause_calls, 1)
        driver()
        self.assertEqual(driver.drive_calls, 1)
        self.assertIs(driver.state, driver.success)

    def test_assign_reports_base_terminated_before_acquire(self):
        scenario = Scenario()
        lock = scenario.Lock()
        acquirer = threading.Thread(target=lambda: None, name="assign-base-ended")
        driver = self.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.terminated,
            lock.acquire, State.BLOCKED)
        def call_assign():
            return lock._core.assign(acquirer, None, False, thread_base=object())
        with self.assertRaisesRegex(RuntimeError, "base tx ended"):
            self.with_fake_driver(scenario, [driver], call_assign)

    def test_assign_reports_wrong_method_and_wrong_state(self):
        scenario = Scenario()
        lock = scenario.Lock()
        acquirer = threading.Thread(target=lambda: None, name="assign-wrong")

        wrong_method = self.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.success,
            lock.release, State.BLOCKED)
        with self.assertRaisesRegex(RuntimeError, "expected 'assign-wrong' to call"):
            self.with_fake_driver(
                scenario, [wrong_method],
                lambda: lock._core.assign(acquirer, None, False))

        wrong_state = self.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.success,
            lock.acquire, State.COMMIT)
        with self.assertRaisesRegex(RuntimeError, "to be BLOCKED"):
            self.with_fake_driver(
                scenario, [wrong_state],
                lambda: lock._core.assign(acquirer, None, False))

    def test_assign_surfaces_releaser_raised_and_acquirer_false(self):
        scenario = Scenario()
        lock = scenario.Lock()
        releaser = threading.Thread(target=lambda: None, name="assign-releaser")
        acquirer = threading.Thread(target=lambda: None, name="assign-acquirer")
        original_actual_held = lock._core.actual_held
        try:
            lock._core.actual_held = lambda: True
            error = RuntimeError("release exploded")
            r = self.FakeAssignDriver(
                scenario._core, releaser, scenario._core.Driver.success,
                lock.release, State.BLOCKED)
            a = self.FakeAssignDriver(
                scenario._core, acquirer, scenario._core.Driver.success,
                lock.acquire, State.BLOCKED)
            def releaser_call(driver):
                driver.drive_calls += 1
                driver.state = driver.success
                driver.tx.done = True
                driver.tx.state = State.RAISED
                driver.tx.result = error
            r.on_call = releaser_call
            with self.assertRaisesRegex(RuntimeError, "raised"):
                self.with_fake_driver(scenario, [r, a], lambda: lock._core.assign(releaser, acquirer, False))
            self.assertEqual(r.finish_calls, 1)
        finally:
            lock._core.actual_held = original_actual_held

        scenario = Scenario()
        lock = scenario.Lock()
        acquirer = threading.Thread(target=lambda: None, name="assign-timeout")
        d = self.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.success,
            lock.acquire, State.BLOCKED)
        def acquirer_call(self):
            self.drive_calls += 1
            self.state = self.success
            self.tx.done = True
            self.tx.state = State.RETURNED
            self.tx.result = False
        d.on_call = acquirer_call
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.with_fake_driver(scenario, [d], lambda: lock._core.assign(acquirer, None, False))
        self.assertEqual(d.finish_calls, 1)


class TestRelayAdditionalCoverage(unittest.TestCase):
    def test_relay_check_reports_terminated_with_and_without_base(self):
        scenario = Scenario()
        lock = scenario.Lock()
        thread = threading.Thread(target=lambda: None, name="relay-ended")

        class Driverish:
            pass

        d = Driverish()
        d.thread = thread
        d.state = scenario._core.Driver.terminated
        d.terminated = scenario._core.Driver.terminated
        d.impasse = scenario._core.Driver.impasse
        d.base_tx = object()
        with self.assertRaisesRegex(RuntimeError, "base tx ended"):
            lock._core._relay_check(d, "acquire")

        d.base_tx = None
        with self.assertRaisesRegex(RuntimeError, "terminated before"):
            lock._core._relay_check(d, "acquire")

    def test_relay_cleanup_skips_done_driver(self):
        scenario = Scenario()
        lock = scenario.Lock()
        thread = threading.Thread(target=lambda: None, name="relay-done")

        class DoneDriver:
            done = True
            def __init__(self):
                self.thread = thread
                self.closed = 0
            def scan(self):
                raise RuntimeError("relay boom")
            def close(self):
                self.closed += 1

        driver = DoneDriver()
        relay = lock._core._relay_generator([driver], pause=False)
        with self.assertRaisesRegex(RuntimeError, "relay boom"):
            next(relay)
        self.assertEqual(driver.closed, 0)
        driver.close()
        self.assertEqual(driver.closed, 1)

    def test_relay_pause_leaves_acquirer_paused_until_unpaused(self):
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)
        order = []

        def holder():
            lock.acquire()
            order.append("holder acquired")
            lock.release()
            order.append("holder released")

        def acquirer():
            lock.acquire()
            order.append("acquirer acquired")
            lock.release()
            order.append("acquirer released")

        with scenario:
            h = scenario.thread(holder)
            a = scenario.thread(acquirer)
            api.assign(h)
            relay = api.relay(h, a, pause=True)
            self.assertIs(next(relay), a)
            tx = scenario.transaction(a)
            self.assertEqual(tx.method, lock.acquire)
            self.assertIs(tx.state, State.PAUSED)
            tx.unpause()
            with self.assertRaises(StopIteration):
                next(relay)
            api.unblock(lock.release, a)

        self.assertEqual(order, [
            "holder acquired", "holder released",
            "acquirer acquired", "acquirer released",
        ])


class TestSemaphoreAllocateCoreErrorCoverage(unittest.TestCase):
    class FakeAllocateDriver(FakeDriver):
        def __init__(self, score, name, state, method, tx_state, *, result=None, base_tx=None):
            super().__init__(score, name)
            self.thread = threading.Thread(target=lambda: None, name=name)
            self.state = state
            self.mode = state
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.success = score.Driver.success
            self.raised = score.Driver.raised
            self.overshot = score.Driver.overshot
            self.mutated = score.Driver.mutated
            self.base_tx = base_tx
            self.tx = types.SimpleNamespace(
                method=method,
                state=tx_state,
                result=result,
                done=False,
                timed_out=False,
                timeout=None,
                kwargs={'blocking': True},
                n=1,
                thread=self.thread,
            )
            self.finish_calls = 0
            self.pause_calls = 0
        def scan(self):
            self.calls.append(("scan",))
        def _call_once(self):
            self.calls.append(("call",))
        def finish(self):
            self.finish_calls += 1
            self.tx.done = True
            self.tx.state = State.RETURNED
            self.state = self.success
            self.mode = self.success
        def pause(self):
            self.pause_calls += 1
            self.tx.state = State.PAUSED
            self.state = self.success
            self.mode = self.success

    def with_fake_allocate_drivers(self, scenario, drivers, func):
        score = scenario._core
        original_driver = score.Driver
        queue_ = list(drivers)
        def make_driver(thread, base=None):
            self.assertTrue(queue_, "unexpected Driver construction")
            driver = queue_.pop(0)
            self.assertIs(driver.thread, thread)
            return driver
        try:
            score.Driver = make_driver
            return func()
        finally:
            score.Driver = original_driver

    def test_fake_allocate_driver_default_methods(self):
        scenario = Scenario()
        sem = scenario.Semaphore()
        driver = self.FakeAllocateDriver(
            scenario._core, "alloc-default", scenario._core.Driver.success,
            sem.acquire, State.BLOCKED)
        driver()
        driver.finish()
        driver.pause()
        self.assertEqual(driver.calls, [("call",)])
        self.assertEqual(driver.finish_calls, 1)
        self.assertEqual(driver.pause_calls, 1)

    def test_allocate_reports_base_terminated_wrong_method_and_wrong_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        core = sem._core

        d = self.FakeAllocateDriver(
            scenario._core, "alloc-base-ended", scenario._core.Driver.terminated,
            sem.acquire, State.BLOCKED, base_tx=object())
        with self.assertRaisesRegex(RuntimeError, "base tx ended"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(core.allocate([(d.thread, object())])))

        d = self.FakeAllocateDriver(
            scenario._core, "alloc-wrong-method", scenario._core.Driver.success,
            object(), State.BLOCKED)
        with self.assertRaisesRegex(ValueError, "calling acquire or release"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(core.allocate([(d.thread, None)])))

        d = self.FakeAllocateDriver(
            scenario._core, "alloc-wrong-state", scenario._core.Driver.success,
            sem.acquire, State.COMMIT)
        with self.assertRaisesRegex(RuntimeError, "must be at BLOCKED"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(core.allocate([(d.thread, None)])))

    def test_allocate_drive_reports_false_result(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-timeout", scenario._core.Driver.success,
            sem.acquire, State.BLOCKED, result=False)
        d.tx.timed_out = True
        d.tx.kwargs = {'blocking': False}
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_cleanup_skips_done_driver(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-done-wrong-method", scenario._core.Driver.success,
            object(), State.BLOCKED)
        d.done = True
        with self.assertRaisesRegex(ValueError, "calling acquire or release"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.closed, 0)

    def test_allocate_reports_unexpected_start_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        strange = types.SimpleNamespace(name='STRANGE')
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-strange-start", strange,
            sem.acquire, State.BLOCKED)
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.closed, 1)

    def test_allocate_bounded_release_with_unknown_initial_value_can_run(self):
        scenario = Scenario()
        sem = scenario.BoundedSemaphore(1)
        sem.acquire()
        self.assertEqual(sem._core.value, 0)
        sem._core.actual._initial_value = None
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-release-unknown-bound", scenario._core.Driver.success,
            sem.release, State.BLOCKED)
        result = self.with_fake_allocate_drivers(
            scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(result, [])
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_zero_timeout_acquire_can_run(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-zero-timeout", scenario._core.Driver.success,
            sem.acquire, State.BLOCKED, result=True)
        d.tx.timeout = 0
        result = self.with_fake_allocate_drivers(
            scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(result, [d.thread])
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_pause_retries_after_nested_pause(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-pause-nested", scenario._core.Driver.success,
            sem.acquire, State.BLOCKED, result=True)
        d.nested = scenario._core.Driver.nested
        def nested_then_success():
            d.pause_calls += 1
            if d.pause_calls == 1:
                d.state = d.nested
                return
            d.tx.state = State.PAUSED
            d.state = d.success
        d.pause = nested_then_success
        result = self.with_fake_allocate_drivers(
            scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)], pause=True)))
        self.assertEqual(result, [d.thread])
        self.assertEqual(d.pause_calls, 2)

    def test_allocate_pause_reports_unexpected_finish_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-pause-strange", scenario._core.Driver.success,
            sem.acquire, State.BLOCKED, result=True)
        d.strange = types.SimpleNamespace(name='STRANGE')
        def strange_pause():
            d.pause_calls += 1
            d.state = d.strange
        d.pause = strange_pause
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)], pause=True)))
        self.assertEqual(d.pause_calls, 1)

    def test_allocate_finish_reports_unexpected_finish_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-finish-strange", scenario._core.Driver.success,
            sem.release, State.BLOCKED)
        d.strange = types.SimpleNamespace(name='STRANGE')
        def strange_finish():
            d.finish_calls += 1
            d.tx.done = True
            d.state = d.strange
        d.finish = strange_finish
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.finish_calls, 1)

    def test_route_pause_deep_retries_after_nested_pause(self):
        scenario = Scenario()
        score = scenario._core
        root = types.SimpleNamespace(done=False, state=State.BLOCKED)
        class RouteDriver:
            success = score.Driver.success
            nested = score.Driver.nested
            def __init__(self):
                self.tx = root
                self.state = self.success
                self.pause_calls = 0
            def pause(self):
                self.pause_calls += 1
                if self.pause_calls == 1:
                    self.state = self.nested
                else:
                    root.state = State.PAUSED
                    self.state = self.success
        d = RouteDriver()
        route = score.route_pause_deep(d)
        next(route)
        self.assertIs(d.state, d.nested)
        next(route)
        self.assertIs(d.state, d.success)
        with self.assertRaises(StopIteration):
            next(route)
        self.assertEqual(d.pause_calls, 2)
        self.assertIs(root.state, State.PAUSED)

class TestFinalPrimitiveCoverageEdges(unittest.TestCase):
    def test_dispatch_rejects_recursion_and_rewaits_routed_driver(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        dispatch.driving = True
        with self.assertRaisesRegex(RuntimeError, "recursively"):
            next(dispatch)
        dispatch.driving = False

        first = object()
        second = object()

        class RoutedDriver(FakeDriver):
            def __init__(self, score):
                super().__init__(score, "routed-dispatch", signals={first})
                self.route_iterator = object()
            def slot_empty(self):
                return True
            def proceed(self):
                self.calls.append(("proceed",))
                self.route_iterator = None
                self.signals = frozenset({second})
                return self.signals

        driver = RoutedDriver(score)
        driver.register(dispatch)
        dispatch.drivers[driver] = frozenset({first})
        waits = [frozenset({first}), frozenset({second})]
        original_wait = score.wait
        try:
            score.wait = lambda signals: waits.pop(0)
            yielded = next(dispatch)
        finally:
            score.wait = original_wait
        self.assertIs(yielded, driver)
        self.assertIn(("proceed",), driver.calls)
        self.assertFalse(waits)

    def test_dispatch_ignores_driver_removed_while_waiting(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        signal = object()
        driver = FakeDriver(score, "removed", signals={signal})
        dispatch.drivers[driver] = frozenset({signal})
        original_wait = score.wait
        def wait(signals):
            self.assertEqual(set(signals), {signal})
            dispatch.drivers.pop(driver)
            return {signal}
        try:
            score.wait = wait
            with self.assertRaises(StopIteration):
                next(dispatch)
        finally:
            score.wait = original_wait
        self.assertNotIn(("signal", frozenset({signal})), driver.calls)

    def test_assign_reports_terminated_without_base_before_acquire(self):
        scenario = Scenario()
        lock = scenario.Lock()
        acquirer = threading.Thread(target=lambda: None, name="assign-dead")
        driver = TestLockAssignCoreErrorCoverage.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.terminated,
            lock.acquire, State.BLOCKED)
        with self.assertRaisesRegex(RuntimeError, "terminated before reaching its acquire"):
            TestLockAssignCoreErrorCoverage().with_fake_driver(
                scenario, [driver], lambda: lock._core.assign(acquirer, None, False))

    def test_allocate_reports_terminated_without_base(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = TestSemaphoreAllocateCoreErrorCoverage.FakeAllocateDriver(
            scenario._core, "alloc-dead", scenario._core.Driver.terminated,
            sem.acquire, State.BLOCKED, base_tx=None)
        with self.assertRaisesRegex(RuntimeError, "terminated before pushing a tx"):
            TestSemaphoreAllocateCoreErrorCoverage().with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))

    def test_condition_cycle_drive_waiter_reports_no_predicate_outcome(self):
        scenario = Scenario()
        cond = scenario.Condition()
        core = cond._core
        cycle = object.__new__(core.Cycle)
        cycle.core = core
        cycle.scheduler = primitives_module._do_nothing
        cycle.caller = "cycle"
        cycle.ul_acquire = core.underlying.primitive.acquire
        cycle.wait_for_methods = (core.primitive.wait_for, core.raw.wait_for)
        cycle.wait_for_drivers = set()
        cycle.ensure_ul_free = lambda: None
        import time
        thread = threading.Thread(target=lambda: None, name="wf-no-outcome")
        tx = core.methods[cond.wait_for](
            cond.wait_for, time.monotonic(), regulated=False,
            predicate=lambda: False)
        tx.thread = thread
        tx.state = State.COMMIT
        class Driverish:
            pass
        d = Driverish()
        d.thread = thread
        d.tx = tx
        d.success = scenario._core.Driver.success
        d.terminated = scenario._core.Driver.terminated
        d.state = scenario._core.Driver.success
        d.callback_signal = None
        d.signaled = frozenset()
        d.motivation = frozenset()
        d.arm_blanket_pause = lambda tx: True
        d.release_blanket_pause = lambda tx: True
        d.reenter = lambda: setattr(d, '_staged', 'reenter')
        def call_no_outcome(self):
            d.state = scenario._core.Driver.success
            d.callback_signal = None
        make_routeable_fake(d, lambda: call_no_outcome(d))
        with self.assertRaisesRegex(RuntimeError, "predicate neither waited nor succeeded"):
            cycle.drive_waiter(d)

    def make_drive_waiter_case(self, *, method='wait_for',
                               reenter_state=None, reenter_callback='predicate',
                               parent_after_reenter=None, resume_state=None,
                               wait_state=None, wait_motivation=None):
        import time
        scenario = Scenario()
        cond = scenario.Condition()
        core = cond._core
        score = scenario._core
        cycle = object.__new__(core.Cycle)
        cycle.core = core
        cycle.scheduler = primitives_module._do_nothing
        cycle.caller = "cycle"
        cycle.ul_acquire = core.underlying.primitive.acquire
        cycle.wait_for_methods = (core.primitive.wait_for, core.raw.wait_for)
        cycle.wait_for_drivers = set()
        cycle.ensure_ul_free = lambda: None
        thread = threading.Thread(target=lambda: None, name="wf-route-case")
        if method == 'wait':
            tx = core.methods[cond.wait](
                cond.wait, time.monotonic(), regulated=False)
        else:
            tx = core.methods[cond.wait_for](
                cond.wait_for, time.monotonic(), regulated=False,
                predicate=lambda: False)
        tx.thread = thread
        tx.state = State.COMMIT
        class Driverish:
            pass
        d = Driverish()
        d.thread = thread
        d.tx = tx
        d.success = score.Driver.success
        d.terminated = score.Driver.terminated
        d.impasse = score.Driver.impasse
        d.state = score.Driver.success
        d.callback_signal = None
        d.motivation = frozenset()
        d.waiting_calls = 0
        d._staged = None
        d.reenter = lambda: setattr(d, '_staged', 'reenter')
        d.resume = lambda: setattr(d, '_staged', 'resume')
        d.wait = lambda *signals: setattr(d, '_staged', 'wait')
        d.scan = lambda tx=None: setattr(d, '_staged', 'scan')
        def waiting():
            d.waiting_calls += 1
            d._staged = 'waiting'
        def arm_blanket_pause(tx_arg=None):
            tx_arg = d.tx if tx_arg is None else tx_arg
            tx_arg.blanket_pause = True
            return True
        def release_blanket_pause(tx_arg=None):
            tx_arg = d.tx if tx_arg is None else tx_arg
            tx_arg.blanket_pause = False
            return True
        d.arm_blanket_pause = arm_blanket_pause
        d.release_blanket_pause = release_blanket_pause
        d.waiting = waiting
        predicate = primitives_module.Predicate(tx.api)
        def call(self):
            staged = d._staged
            d._staged = None
            if staged == 'reenter':
                d.state = score.Driver.success if reenter_state is None else reenter_state
                if reenter_callback == 'predicate':
                    d.callback_signal = predicate
                else:
                    d.callback_signal = None
                if parent_after_reenter is not None:
                    tx.state = parent_after_reenter
            elif staged == 'resume':
                d.state = score.Driver.success if resume_state is None else resume_state
            elif staged == 'wait':
                d.state = score.Driver.success if wait_state is None else wait_state
                d.motivation = frozenset() if wait_motivation is None else wait_motivation(tx.api)
            elif staged == 'scan':
                child = core.methods[cond.wait](
                    cond.wait, time.monotonic(), regulated=False)
                child.thread = thread
                child.parent = tx
                child.state = State.BLOCKED
                d.tx = child
                d.state = score.Driver.success
            else:
                assert staged == 'waiting'
                d.tx.state = State.WAITING
                d.state = score.Driver.success
        make_routeable_fake(d, lambda: call(d))
        tx.blanket_pause = False
        tx.scheduler_pause = False
        return cycle, d, tx

    def test_condition_cycle_drive_waiter_plain_wait_waits(self):
        cycle, d, tx = self.make_drive_waiter_case(method='wait')
        self.assertEqual(cycle.drive_waiter(d), 'waiting')
        self.assertEqual(d.waiting_calls, 1)
        self.assertIs(d.tx.state, State.WAITING)

    def test_condition_cycle_drive_waiter_ready_branches(self):
        cycle, d, tx = self.make_drive_waiter_case(
            reenter_callback=None, parent_after_reenter=State.PAUSED)
        self.assertEqual(cycle.drive_waiter(d), 'ready')
        self.assertTrue(tx.blanket_pause)

        cycle, d, tx = self.make_drive_waiter_case(
            wait_motivation=lambda wf: frozenset({wf}))
        self.assertEqual(cycle.drive_waiter(d), 'ready')
        self.assertFalse(tx.blanket_pause)

        cycle, d, tx = self.make_drive_waiter_case(
            wait_motivation=lambda wf: frozenset({Nested(wf)}))
        self.assertEqual(cycle.drive_waiter(d), 'waiting')
        self.assertFalse(tx.blanket_pause)
        self.assertEqual(d.waiting_calls, 1)
        self.assertIs(d.tx.state, State.WAITING)

    def test_condition_cycle_drive_waiter_route_errors(self):
        score = Scenario()._core
        cases = [
            dict(reenter_state=score.Driver.terminated,
                 error="terminated during wait_for predicate"),
            dict(reenter_state=score.Driver.impasse,
                 error="predicate drive stopped"),
            dict(resume_state=score.Driver.terminated,
                 error="terminated during wait_for predicate"),
            dict(resume_state=score.Driver.impasse,
                 error="predicate resume stopped"),
            dict(wait_state=score.Driver.terminated,
                 error="terminated while settling"),
            dict(wait_motivation=lambda wf: frozenset(),
                 error="no recognizable settling signal"),
        ]
        for case in cases:
            error = case.pop('error')
            with self.subTest(error=error):
                cycle, d, tx = self.make_drive_waiter_case(**case)
                with self.assertRaisesRegex(RuntimeError, error):
                    cycle.drive_waiter(d)


class TestConditionCycleReenteredCoverage(unittest.TestCase):
    def make_cycle_and_driver(self):
        import time
        scenario = Scenario()
        cond = scenario.Condition()
        core = cond._core
        score = scenario._core
        wait_for_tx = core.methods[cond.wait_for](
            cond.wait_for, time.monotonic(), regulated=False,
            predicate=lambda: False)
        wait_tx = core.methods[cond.wait](
            cond.wait, time.monotonic(), regulated=False)
        wait_tx.parent = wait_for_tx
        wait_for_tx.child = wait_tx
        thread = threading.Thread(target=lambda: None, name="wf-reentered")
        wait_tx.thread = thread
        wait_for_tx.thread = thread
        cycle = object.__new__(core.Cycle)
        cycle.core = core
        cycle.scheduler_calls = []
        cycle.scheduler = lambda tx: cycle.scheduler_calls.append(tx)
        cycle.caller = "cycle"
        cycle.ul_acquire = core.underlying.primitive.acquire
        cycle.ul_release = (core.underlying.primitive.release, core.underlying.raw.release)
        cycle.wait_for_methods = (core.primitive.wait_for, core.raw.wait_for)
        cycle.wait_for_drivers = set()
        cycle.previous = None
        cycle.ensure_ul_free = lambda: None
        class Driverish:
            pass
        d = Driverish()
        d.thread = thread
        d.tx = wait_tx
        d.state = score.Driver.success
        d.success = score.Driver.success
        d.terminated = score.Driver.terminated
        d.raised = score.Driver.raised
        d.impasse = score.Driver.impasse
        d.overshot = score.Driver.overshot
        d.mutated = score.Driver.mutated
        d.signaled = frozenset()
        d.motivation = frozenset()
        d.finish_calls = 0
        d.pause_calls = 0
        d.waiting_calls = 0
        d.scan_calls = 0
        d.resume_calls = 0
        d.call_count = 0
        d._staged = None
        d._wait_signals = None
        d._asserted_signals = None
        wait_for_tx.blanket_pause = False
        wait_for_tx.scheduler_pause = False
        def arm_blanket_pause(tx):
            tx.blanket_pause = True
            return True
        def release_blanket_pause(tx):
            tx.blanket_pause = False
            return True
        def handoff_blanket_pause_to_scheduler_pause(tx=None):
            tx = d.tx if tx is None else tx
            tx.scheduler_pause = True
            tx.pause = True
            tx.blanket_pause = False
            return True
        def finish():
            d.finish_calls += 1
            d._staged = 'finish'
        def pause():
            d.pause_calls += 1
            d._staged = 'pause'
        def scan():
            d.scan_calls += 1
            d._staged = 'scan'
        def waiting():
            d.waiting_calls += 1
            d._staged = 'waiting'
        def resume():
            d.resume_calls += 1
            d._staged = 'resume'
        def wait(*signals):
            d._staged = 'wait'
            terminated = Terminated(d.thread)
            d._asserted_signals = frozenset(signals)
            d._wait_signals = tuple(signals) + (terminated,)
        def call():
            d.call_count += 1
            if d._staged == 'wait':
                fired = frozenset(score.wait(d._wait_signals))
                d.signaled = frozenset(fired)
                explicit = fired & d._asserted_signals
                if explicit:
                    d.motivation = explicit
                    d.state = d.success
                elif Terminated(d.thread) in fired:  # pragma: no cover - defensive fake path
                    d.motivation = frozenset({Terminated(d.thread)})
                    d.state = d.terminated
                else:  # pragma: no cover - defensive fake path
                    d.motivation = frozenset()
                    d.state = d.success
            elif d._staged == 'resume':
                signal = primitives_module.Predicate(wait_for_tx.api)
                not_signal = Not(signal)
                fired = frozenset(score.wait((not_signal, Terminated(d.thread))))
                d.signaled = frozenset(fired)
                if not_signal in fired:
                    d.motivation = frozenset({not_signal})
                    d.state = d.success
                elif Terminated(d.thread) in fired:
                    d.motivation = frozenset({Terminated(d.thread)})
                    d.state = d.terminated
                else:  # pragma: no cover - defensive fake path
                    d.motivation = frozenset()
                    d.state = d.success
            elif d._staged == 'finish':
                wait_tx.state = State.RETURNED
            elif d._staged == 'pause':
                wait_for_tx.state = State.PAUSED
            elif d._staged == 'waiting':
                wait_tx.state = State.WAITING
            if d._staged not in ('wait', 'resume'):
                d.state = d.success
            d._staged = None
            d._wait_signals = None
            d._asserted_signals = None
        d.arm_blanket_pause = arm_blanket_pause
        d.release_blanket_pause = release_blanket_pause
        d.handoff_blanket_pause_to_scheduler_pause = handoff_blanket_pause_to_scheduler_pause
        d.finish = finish
        d.pause = pause
        d.scan = scan
        d.waiting = waiting
        d.resume = resume
        d.wait = wait
        make_routeable_fake(d, call)
        cycle.ready = [d]
        cycle.wait_for_drivers.add(d)
        wait_tx.state = State.STALLED
        return scenario, score, wait_for_tx.api, wait_tx, cycle, d

    def run_act_one_with_waits(self, verb, wait_results):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = list(wait_results)
        calls = []
        def wait(signals):
            calls.append(tuple(signals))
            self.assertTrue(results, "unexpected wait call")
            result = results.pop(0)
            return result(wf, d.thread) if callable(result) else result
        score.wait = wait
        score.lock.acquire()
        try:
            cycle.act_one(d, verb)
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        return wf, cycle, d, calls


    def test_driverish_fake_directives_cover_all_staged_branches(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        d.pause(); d()
        self.assertEqual(d.pause_calls, 1)
        self.assertIs(wf._core.state, State.PAUSED)
        d.scan(); d()
        self.assertEqual(d.scan_calls, 1)
        d.waiting(); d()
        self.assertEqual(d.waiting_calls, 1)
        self.assertIs(wait_tx.state, State.WAITING)

    def test_condition_cycle_wait_reruns_scheduler_then_thread_terminates(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {primitives_module.Predicate(wf)},
            lambda wf, thread: {Not(primitives_module.Predicate(wf))},
            lambda wf, thread: {Terminated(thread)},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "thread terminated"):
                cycle.act_one(d, 'wait')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertEqual(cycle.scheduler_calls, [wf])

    def test_condition_cycle_wait_termination_during_resume_reports_error(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {primitives_module.Predicate(wf)},
            lambda wf, thread: {Terminated(thread)},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "thread terminated"):
                cycle.act_one(d, 'wait')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        self.assertEqual(cycle.scheduler_calls, [wf])

    def test_condition_cycle_wake_termination_during_resume_returns_previous(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'wake',
            [
                lambda wf, thread: {primitives_module.Predicate(wf)},
                lambda wf, thread: {Terminated(thread)},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)

    def test_condition_cycle_wake_reentered_exits_on_parent_transaction(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'wake',
            [
                lambda wf, thread: {primitives_module.Predicate(wf)},
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {wf},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)
        self.assertEqual(d.finish_calls, 1)

    def test_condition_cycle_pause_reentered_exits_on_paused_parent(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'pause',
            [
                lambda wf, thread: {primitives_module.Predicate(wf)},
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {primitives_module.Paused(wf)},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)
        self.assertEqual(d.pause_calls, 0)

    def test_condition_cycle_wake_reentered_returns_on_thread_termination(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'wake',
            [
                lambda wf, thread: {primitives_module.Predicate(wf)},
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {Terminated(thread)},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)

    def test_condition_cycle_pause_reentered_complains_if_predicate_waited_again(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {Nested(wf)},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, r"pause\(\) expected"):
                cycle.act_one(d, 'pause')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        self.assertIsNone(cycle.previous)

    def test_condition_cycle_wake_reentered_complains_if_predicate_waited_again(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {Nested(wf)},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, r"wake\(\) expected"):
                cycle.act_one(d, 'wake')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        self.assertIsNone(cycle.previous)


class TestRemainingCycleErrorLineCoverage(unittest.TestCase):
    class QueueDispatch:
        def __init__(self):
            self.items = []
        def add(self, driver):
            self.items.append(driver)
        def __iter__(self):
            while self.items:
                yield self.items.pop(0)

    class Tx:
        def __init__(self, method, state):
            self.method = method
            self.state = state
            self.result = None
            self.done = False
        def validate(self, **kwargs):
            return None

    class CycleDriver:
        def __init__(self, score, thread, method, tx_state):
            self.thread = thread
            self.state = score.Driver.success
            self.mode = self.state
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.success = score.Driver.success
            self.active = self.success
            self.done = False
            self.closed = 0
            self.tx = TestRemainingCycleErrorLineCoverage.Tx(method, tx_state)
            self.base_tx = None
            self.wait_calls = 0
            self.blanket_pause_calls = 0
        def close(self):
            self.closed += 1
            self.done = True
        def scan(self):
            pass
        def wait(self):
            self.wait_calls += 1
            self.state = self.success
            self.mode = self.success
            self.tx.state = State.WAITING
        def waiting(self):
            return self.wait()
        def stall(self):
            self.state = self.success
            self.mode = self.success
            self.tx.state = State.STALLED
        def pause_internal(self):
            self.blanket_pause_calls += 1
            self.state = self.success
            self.mode = self.success
            self.tx.state = State.PAUSED

        def __call__(self):
            return None

    def test_cycle_driver_test_double_methods(self):
        scenario = Scenario()
        lock = scenario.Lock()
        driver = self.CycleDriver(
            scenario._core, threading.Thread(target=lambda: None),
            lock.acquire, State.BLOCKED)
        driver.stall()
        self.assertIs(driver.state, driver.success)
        self.assertIs(driver.tx.state, State.STALLED)
        driver.pause_internal()
        self.assertEqual(driver.blanket_pause_calls, 1)
        self.assertIs(driver.tx.state, State.PAUSED)
        self.assertIsNone(driver())

    def with_fake_cycle_drivers(self, scenario, drivers, func):
        score = scenario._core
        original_driver = score.Driver
        original_dispatch = score.Dispatch
        queue_ = list(drivers)
        def make_driver(thread, base=None):
            self.assertTrue(queue_, "unexpected Driver construction")
            driver = queue_.pop(0)
            self.assertIs(driver.thread, thread)
            return driver
        try:
            score.Driver = make_driver
            score.Dispatch = self.QueueDispatch
            return func()
        finally:
            score.Driver = original_driver
            score.Dispatch = original_dispatch

    def test_condition_cycle_cleanup_skips_done_driver(self):
        scenario = Scenario()
        cond = scenario.Condition()
        waiter = threading.Thread(target=lambda: None, name="cond-not-waiting-done")
        waker = threading.Thread(target=lambda: None, name="cond-notify-done")
        waiter_driver = self.CycleDriver(scenario._core, waiter, cond.wait, State.BLOCKED)
        waker_driver = self.CycleDriver(scenario._core, waker, cond.notify, State.BLOCKED)
        waker_driver.done = True
        with self.assertRaisesRegex(ValueError, "all waiters must already be in WAITING"):
            self.with_fake_cycle_drivers(
                scenario, [waiter_driver, waker_driver],
                lambda: cond._core.Cycle([waiter, waker]))
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(waker_driver.closed, 0)

    def test_condition_cycle_rejects_notify_before_waiters_are_waiting(self):
        scenario = Scenario()
        cond = scenario.Condition()
        waiter = threading.Thread(target=lambda: None, name="cond-not-waiting")
        waker = threading.Thread(target=lambda: None, name="cond-notify")
        waiter_driver = self.CycleDriver(scenario._core, waiter, cond.wait, State.BLOCKED)
        waker_driver = self.CycleDriver(scenario._core, waker, cond.notify, State.BLOCKED)
        with self.assertRaisesRegex(ValueError, "all waiters must already be in WAITING"):
            self.with_fake_cycle_drivers(
                scenario, [waiter_driver, waker_driver],
                lambda: cond._core.Cycle([waiter, waker]))
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(waker_driver.closed, 1)

    def test_event_cycle_cleanup_skips_done_driver(self):
        scenario = Scenario()
        event = scenario.Event()
        waiter = threading.Thread(target=lambda: None, name="event-waiter-done")
        setter = threading.Thread(target=lambda: None, name="event-setter-done")
        waiter_driver = self.CycleDriver(scenario._core, waiter, event.wait, State.BLOCKED)
        setter_driver = self.CycleDriver(scenario._core, setter, event.set, State.BLOCKED)
        setter_driver.done = True
        original_is_set = event._core.actual.is_set
        original_waiter_count = event._core.actual_waiter_count
        calls = []
        def is_set():
            calls.append("is_set")
            return len(calls) > 1
        try:
            event._core.actual.is_set = is_set
            event._core.actual_waiter_count = lambda: 1
            with self.assertRaisesRegex(RuntimeError, "Event was set unexpectedly"):
                self.with_fake_cycle_drivers(
                    scenario, [waiter_driver, setter_driver],
                    lambda: event._core.Cycle([waiter, setter]))
        finally:
            event._core.actual.is_set = original_is_set
            event._core.actual_waiter_count = original_waiter_count
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(setter_driver.closed, 0)

    def test_event_cycle_reports_event_set_during_construction(self):
        scenario = Scenario()
        event = scenario.Event()
        waiter = threading.Thread(target=lambda: None, name="event-waiter")
        setter = threading.Thread(target=lambda: None, name="event-setter")
        waiter_driver = self.CycleDriver(scenario._core, waiter, event.wait, State.BLOCKED)
        setter_driver = self.CycleDriver(scenario._core, setter, event.set, State.BLOCKED)
        original_is_set = event._core.actual.is_set
        original_waiter_count = event._core.actual_waiter_count
        calls = []
        def is_set():
            calls.append("is_set")
            return len(calls) > 1
        try:
            event._core.actual.is_set = is_set
            event._core.actual_waiter_count = lambda: 1
            with self.assertRaisesRegex(RuntimeError, "Event was set unexpectedly"):
                self.with_fake_cycle_drivers(
                    scenario, [waiter_driver, setter_driver],
                    lambda: event._core.Cycle([waiter, setter]))
        finally:
            event._core.actual.is_set = original_is_set
            event._core.actual_waiter_count = original_waiter_count
        self.assertEqual(calls, ["is_set", "is_set"])
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(setter_driver.closed, 1)

    def test_barrier_cycle_reports_extra_waiters(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        waiter = threading.Thread(target=lambda: None, name="barrier-waiter")
        opener = threading.Thread(target=lambda: None, name="barrier-opener")
        waiter_driver = self.CycleDriver(scenario._core, waiter, barrier.wait, State.WAITING)
        opener_driver = self.CycleDriver(scenario._core, opener, barrier.wait, State.BLOCKED)
        original_waiters = barrier._core.actual._cond._waiters
        barrier._core.actual._cond._waiters = [object(), object()]
        try:
            with self.assertRaisesRegex(RuntimeError, "extra threads"):
                self.with_fake_cycle_drivers(
                    scenario, [waiter_driver, opener_driver],
                    lambda: barrier._core.Cycle([waiter, opener]))
        finally:
            barrier._core.actual._cond._waiters = original_waiters
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(opener_driver.closed, 1)

    def test_barrier_cycle_reraises_runtime_error_when_no_waiter_raised(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        waiter = threading.Thread(target=lambda: None, name="barrier-failing-waiter")
        opener = threading.Thread(target=lambda: None, name="barrier-opener")
        waiter_driver = self.CycleDriver(scenario._core, waiter, barrier.wait, State.BLOCKED)
        opener_driver = self.CycleDriver(scenario._core, opener, barrier.wait, State.BLOCKED)
        def failing_wait():
            raise RuntimeError("parking failed")
        waiter_driver.waiting = failing_wait
        original_waiters = barrier._core.actual._cond._waiters
        barrier._core.actual._cond._waiters = []
        try:
            with self.assertRaisesRegex(RuntimeError, "parking failed"):
                self.with_fake_cycle_drivers(
                    scenario, [waiter_driver, opener_driver],
                    lambda: barrier._core.Cycle([waiter, opener]))
        finally:
            barrier._core.actual._cond._waiters = original_waiters
        self.assertEqual(waiter_driver.closed, 1)
        self.assertEqual(opener_driver.closed, 1)


class TestLastPrimitivesBranchCoverage(unittest.TestCase):
    def test_wait_transaction_cleanup_leaves_nonempty_waiter_bucket(self):
        scenario = Scenario()
        score = scenario._core
        class NeverSignal(primitives_module.Signaling):
            def sample(self, scenario):
                return False
            def __repr__(self):
                return "NeverSignal()"
        signal = NeverSignal()
        self.assertEqual(repr(signal), "NeverSignal()")
        sentinel = object()
        score.waiters[signal].add(sentinel)
        wtx = score.WaitTransaction({signal})
        score.lock.acquire()
        try:
            self.assertEqual(wtx.wait(timeout=0.001), set())
        finally:
            if score.lock.locked():
                score.lock.release()
        self.assertEqual(score.waiters[signal], {sentinel})

    def test_driver_release_blanket_pause_is_idempotent(self):
        score = Scenario()._core
        driver = score.Driver(threading.Thread(target=lambda: None, name="release-internal-pause"))
        self.assertTrue(driver.release_blanket_pause())
        class Tx:
            blanket_pause = True
            def release_blanket_pause(self):
                self.blanket_pause = False
        tx = Tx()
        driver.tx = tx
        driver.set_owns_pause(tx, True)
        self.assertTrue(driver.release_blanket_pause(tx))
        self.assertFalse(tx.blanket_pause)
        self.assertTrue(driver.release_blanket_pause(tx))


    def test_transaction_call_with_post_commit_state_skips_commit_block(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.locked](
            lock.locked, time.monotonic(), regulated=False)
        tx.state = State.COMMITTED
        tx.raised = False
        tx.timed_out = False
        scenario._core.lock.acquire()
        try:
            result = tx()
        finally:
            if scenario._core.lock.locked():
                scenario._core.lock.release()
        self.assertIs(result, None)
        self.assertIs(tx.state, State.RETURNED)
        self.assertTrue(tx.succeeded)

    def test_condition_cycle_wait_predicate_branch_with_default_scheduler(self):
        helper = TestConditionCycleReenteredCoverage()
        scenario, score, wf, wait_tx, cycle, d = helper.make_cycle_and_driver()
        cycle.scheduler = primitives_module._do_nothing
        original_wait = score.wait
        results = [
            lambda wf, thread: {primitives_module.Predicate(wf)},
            lambda wf, thread: {Not(primitives_module.Predicate(wf))},
            lambda wf, thread: {Terminated(thread)},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "thread terminated"):
                cycle.act_one(d, 'wait')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        self.assertEqual(cycle.scheduler_calls, [])

    def test_condition_cycle_wake_reentered_with_default_scheduler(self):
        helper = TestConditionCycleReenteredCoverage()
        scenario, score, wf, wait_tx, cycle, d = helper.make_cycle_and_driver()
        cycle.scheduler = primitives_module._do_nothing
        original_wait = score.wait
        results = [
            lambda wf, thread: {primitives_module.Predicate(wf)},
            lambda wf, thread: {Not(primitives_module.Predicate(wf))},
            lambda wf, thread: {wf},
        ]
        def wait(signals):
            self.assertTrue(results, "unexpected wait call")
            return results.pop(0)(wf, d.thread)
        score.wait = wait
        score.lock.acquire()
        try:
            cycle.act_one(d, 'wake')
        finally:
            score.wait = original_wait
            if score.lock.locked():
                score.lock.release()
        self.assertFalse(results)
        self.assertEqual(cycle.scheduler_calls, [])
        self.assertIs(cycle.previous, d)


class TestSimulatedMissingStdlibFeatureCoverage(unittest.TestCase):
    def load_primitives_with_missing_features(self):
        import importlib.util
        import pathlib
        import queue as queue_module
        import threading as threading_module

        class NoAliasLock:
            def acquire(self, *args, **kwargs): return True
            def release(self): return None
            def locked(self): return False

        rlock_provides_locked = hasattr(threading_module.RLock(), 'locked')

        class NoPrivateRLock:
            def acquire(self, *args, **kwargs): return True
            def release(self): return None

        if rlock_provides_locked:
            def no_private_rlock_locked(self):
                return False
            NoPrivateRLock.locked = no_private_rlock_locked

        class QueueWithoutShutdown:
            pass

        lock_probe = NoAliasLock()
        self.assertTrue(lock_probe.acquire())
        self.assertIsNone(lock_probe.release())
        self.assertFalse(lock_probe.locked())
        rlock_probe = NoPrivateRLock()
        self.assertTrue(rlock_probe.acquire())
        self.assertIsNone(rlock_probe.release())
        if rlock_provides_locked:
            self.assertFalse(rlock_probe.locked())
        else:
            self.assertFalse(hasattr(rlock_probe, 'locked'))

        path = pathlib.Path(primitives_module.__file__)
        original_lock = threading_module.Lock
        original_rlock = threading_module.RLock
        original_queue = queue_module.Queue
        had_simplequeue = hasattr(queue_module, 'SimpleQueue')
        original_simplequeue = getattr(queue_module, 'SimpleQueue', None)
        try:
            threading_module.Lock = lambda: NoAliasLock()
            threading_module.RLock = lambda: NoPrivateRLock()
            queue_module.Queue = QueueWithoutShutdown
            if had_simplequeue:
                delattr(queue_module, 'SimpleQueue')
            spec = importlib.util.spec_from_file_location(
                "blanket_primitives_missing_stdlib_features_for_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            threading_module.Lock = original_lock
            threading_module.RLock = original_rlock
            queue_module.Queue = original_queue
            if had_simplequeue:
                queue_module.SimpleQueue = original_simplequeue
        self.assertFalse(module._lock_provides_legacy_aliases)
        self.assertFalse(module._queue_provides_simplequeue)
        self.assertFalse(module._queue_provides_shutdown)
        return module

    def test_missing_stdlib_features_are_not_exposed(self):
        module = self.load_primitives_with_missing_features()
        scenario = module.Scenario()
        lock = scenario.Lock()
        raw_lock = scenario.raw(lock)
        self.assertFalse(hasattr(lock, 'acquire_lock'))
        self.assertFalse(hasattr(lock, 'release_lock'))
        self.assertFalse(hasattr(lock, 'locked_lock'))
        self.assertFalse(hasattr(raw_lock, 'acquire_lock'))
        self.assertFalse(hasattr(raw_lock, 'release_lock'))
        self.assertFalse(hasattr(raw_lock, 'locked_lock'))

        rlock = scenario.RLock()
        self.assertFalse(hasattr(rlock, '_recursion_count'))

        self.assertFalse(hasattr(scenario, 'SimpleQueue'))
        self.assertFalse(hasattr(scenario.Queue(), 'shutdown'))
        self.assertFalse(hasattr(scenario.raw(scenario.Queue()), 'shutdown'))


class TestSimulatedOldSemaphoreReleaseCoverage(unittest.TestCase):
    def load_primitives_without_release_n(self):
        import importlib.util
        import inspect
        import pathlib
        import threading as threading_module
        path = pathlib.Path(primitives_module.__file__)
        original_signature = inspect.signature
        class SignatureWithoutN:
            parameters = {}
        def fake_signature(obj, *args, **kwargs):
            if obj is threading_module.Semaphore.release:
                return SignatureWithoutN()
            return original_signature(obj, *args, **kwargs)
        sample_callable = lambda: None
        self.assertEqual(fake_signature(sample_callable), original_signature(sample_callable))
        inspect.signature = fake_signature
        try:
            spec = importlib.util.spec_from_file_location(
                "blanket_primitives_no_release_n_for_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            inspect.signature = original_signature
        self.assertFalse(module._semaphore_release_accepts_n)
        return module

    def test_raw_semaphore_release_old_signature_and_tx_repr(self):
        import inspect
        import time
        module = self.load_primitives_without_release_n()
        scenario = module.Scenario()
        sem = scenario.Semaphore(1)
        raw = scenario.raw(sem)
        self.assertNotIn('n', inspect.signature(raw.release).parameters)
        self.assertIsNone(raw.release())
        self.assertEqual(sem._core.actual._value, 2)
        tx = sem._core.methods[sem.release](sem.release, time.monotonic(), regulated=True)
        self.assertIn("Semaphore.release", repr(tx))

    def test_raw_bounded_semaphore_release_old_signature(self):
        import inspect
        module = self.load_primitives_without_release_n()
        scenario = module.Scenario()
        bsem = scenario.BoundedSemaphore(2)
        bsem.acquire()
        raw = scenario.raw(bsem)
        self.assertNotIn('n', inspect.signature(raw.release).parameters)
        self.assertIsNone(raw.release())
        self.assertEqual(bsem._core.actual._value, 2)

class TestQueueDeliverInternalCoverage(unittest.TestCase):
    class FakeCore:
        def __init__(self, *, size=1, can_run=True):
            self.size = size
            self.can_run = can_run

        def deliver_role(self, tx):
            return getattr(tx, 'role', None)

        def deliver_size(self):
            return self.size

        def deliver_can_run(self, role, size):
            return self.can_run

        def deliver_adjust_size(self, role, size):
            return size

    class HashThread:
        def __init__(self, name='worker'):
            self.name = name

    class FakeTx:
        def __init__(self, *, role='get', state=State.BLOCKED, result=None):
            self.role = role
            self.state = state
            self.result = result
            self.done = False
            self.method = types.SimpleNamespace(__name__='fake_queue_call')
            self.api = object()

    class FakeDriver:
        undirected = types.SimpleNamespace(name='UNDIRECTED')
        success = types.SimpleNamespace(name='SUCCESS')
        active = types.SimpleNamespace(name='ACTIVE')
        terminated = types.SimpleNamespace(name='TERMINATED')
        impasse = types.SimpleNamespace(name='IMPASSE')
        raised = types.SimpleNamespace(name='RAISED')
        overshot = types.SimpleNamespace(name='OVERSHOT')
        mutated = types.SimpleNamespace(name='MUTATED')
        strange = types.SimpleNamespace(name='STRANGE')

        def __init__(self, thread, base_tx, *, start_state=None, end_state=None, tx=None):
            self.thread = thread
            self.base_tx = base_tx
            self.start_state = self.success if start_state is None else start_state
            self.end_state = self.success if end_state is None else end_state
            self.tx = (types.SimpleNamespace(
                role='get',
                state=State.BLOCKED,
                result=None,
                done=False,
                method=types.SimpleNamespace(__name__='fake_queue_call'),
                api=object()) if tx is None else tx)
            self.state = self.undirected
            self.mode = self.undirected
            self.calls = 0
            self.finished_called = False
            self.closed = False

        @property
        def done(self):
            return self.state in (self.terminated, self.impasse, self.raised)

        def scan(self):
            pass

        def route(self, route):
            self._route_iterator = route(self)
        def _call_once(self):
            self.calls += 1
            self.state = self.start_state if self.calls == 1 else self.end_state
            self.mode = self.state
        def __call__(self):
            iterator = getattr(self, '_route_iterator', None)
            if iterator is not None:
                self._route_iterator = None
                while True:
                    try:
                        next(iterator)
                    except StopIteration:
                        return
                    self._call_once()
            else:
                self._call_once()

        def finish(self):
            self.finished_called = True
            self.tx.done = True
            if self.tx.state is not State.RAISED:
                self.tx.state = State.RETURNED

        def close(self):
            self.closed = True

    def drive_with_driver(self, driver, *, core=None, pairs=None):
        scenario = Scenario()
        score = scenario._core
        core = self.FakeCore() if core is None else core
        thread = self.HashThread('worker')
        pairs = [(thread, None)] if pairs is None else pairs
        score.Driver = lambda thread, base_tx: driver
        return score.drive_queue_deliver(core, pairs)


    def test_fake_driver_finish_preserves_raised_state(self):
        tx = self.FakeTx(state=State.RAISED)
        driver = self.FakeDriver(self.HashThread('worker'), None, tx=tx)
        driver.finish()
        self.assertTrue(driver.finished_called)
        self.assertIs(tx.state, State.RAISED)
        self.assertTrue(tx.done)

    def test_deliver_empty_participant_list_rejected(self):
        scenario = Scenario()
        with self.assertRaises(ValueError):
            scenario._core.drive_queue_deliver(self.FakeCore(), [])

    def test_deliver_driver_impasse_rejected(self):
        driver = self.FakeDriver(
            self.HashThread('worker'), None,
            start_state=self.FakeDriver.impasse)
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver)
        self.assertTrue(driver.done)

    def test_deliver_driver_terminated_without_base_rejected(self):
        driver = self.FakeDriver(
            self.HashThread('worker'), None,
            start_state=self.FakeDriver.terminated)
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver)
        self.assertTrue(driver.done)

    def test_deliver_driver_terminated_with_base_rejected(self):
        driver = self.FakeDriver(
            self.HashThread('worker'), object(),
            start_state=self.FakeDriver.terminated)
        pairs = [(self.HashThread('worker'), object())]
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver, pairs=pairs)
        self.assertTrue(driver.done)

    def test_deliver_unexpected_start_state_rejected(self):
        driver = self.FakeDriver(
            self.HashThread('worker'), None,
            start_state=self.FakeDriver.strange)
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver)
        self.assertTrue(driver.closed)

    def test_deliver_non_blocked_tx_rejected(self):
        tx = self.FakeTx(state=State.COMMIT)
        driver = self.FakeDriver(self.HashThread('worker'), None, tx=tx)
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver)
        self.assertTrue(driver.closed)

    def test_deliver_raised_tx_propagates_result(self):
        err = ValueError('boom')
        tx = self.FakeTx(state=State.BLOCKED, result=err)
        driver = self.FakeDriver(self.HashThread('worker'), None, tx=tx)
        # Special methods are looked up on the class, so patch the class
        # method for this one instance and restore it immediately below.
        cls = type(driver)
        old_call = cls.__call__
        def patched(self):
            old_call(self)
            if self is driver and self.calls == 2:
                tx.state = State.RAISED
        cls.__call__ = patched
        try:
            with self.assertRaises(ValueError):
                self.drive_with_driver(driver)
        finally:
            cls.__call__ = old_call
        self.assertTrue(driver.finished_called)

    def test_deliver_unexpected_finish_state_rejected(self):
        driver = self.FakeDriver(
            self.HashThread('worker'), None,
            end_state=self.FakeDriver.active)
        with self.assertRaises(RuntimeError):
            self.drive_with_driver(driver)
        self.assertTrue(driver.closed)

    def test_deliver_role_helpers_reject_foreign_and_nontraffic_txs(self):
        scenario = Scenario()
        q = scenario.Queue()
        other = scenario.Queue()
        self.assertIsNone(q._core.deliver_role(types.SimpleNamespace(core=other._core)))
        self.assertEqual(self.FakeCore().deliver_adjust_size('get', 3), 3)
        sq = scenario.SimpleQueue()
        self.assertIsNone(sq._core.deliver_role(types.SimpleNamespace(core=q._core)))
        self.assertIsNone(sq._core.deliver_role(types.SimpleNamespace(core=sq._core)))


class TestPrimitiveAPIThreadSpecs(unittest.TestCase):

    def make_base_child(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core
        thread = threading.Thread(target=lambda: None, name="api-thread-spec")
        base = core.methods[lock.acquire](lock.acquire, time.monotonic(), regulated=False)
        child = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        other = core.methods[lock.locked](lock.locked, time.monotonic(), regulated=False)
        for tx in (base, child, other):
            tx.thread = thread
        child.parent = base
        base.child = child
        return scenario, lock, core, thread, base, child, other

    def test_threads_to_txs_accepts_existing_base_child(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.COMMIT
        score.transactions[thread] = child
        try:
            threads, txs = core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.transactions.pop(thread, None)
        self.assertEqual(threads, ((thread, base.api),))
        self.assertEqual(txs, [child])

    def test_threads_to_txs_waits_for_base_child(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.COMMIT
        base.child = None
        score.transactions[thread] = base
        old_wait = score.wait
        calls = []

        def fake_wait(items):
            calls.append(items)
            child.parent = base
            base.child = child
            score.transactions[thread] = child
            return {Nested(base.api)}

        score.wait = fake_wait
        try:
            threads, txs = core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.wait = old_wait
            score.transactions.pop(thread, None)
        self.assertEqual(threads, ((thread, base.api),))
        self.assertEqual(txs, [child])
        self.assertEqual(len(calls), 1)

    def test_threads_to_txs_base_ended_before_child(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.RETURNED
        score.transactions[thread] = other
        try:
            with self.assertRaisesRegex(RuntimeError, "base tx ended"):
                core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.transactions.pop(thread, None)

    def test_threads_to_txs_base_parked_before_child(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.BLOCKED
        score.transactions[thread] = base
        try:
            with self.assertRaisesRegex(RuntimeError, "base tx is blanket-parked"):
                core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.transactions.pop(thread, None)

    def test_threads_to_txs_wait_reports_base_ending(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.COMMIT
        score.transactions[thread] = base
        old_wait = score.wait
        score.wait = lambda items: {base.api}
        try:
            with self.assertRaisesRegex(RuntimeError, "base tx ended"):
                core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.wait = old_wait
            score.transactions.pop(thread, None)

    def test_threads_to_txs_wait_reports_thread_termination(self):
        scenario, lock, core, thread, base, child, other = self.make_base_child()
        score = scenario._core
        base.state = State.COMMIT
        score.transactions[thread] = base
        old_wait = score.wait
        score.wait = lambda items: {Terminated(thread)}
        try:
            with self.assertRaisesRegex(ValueError, "has exited"):
                core.threads_to_txs(((thread, base.api),), caller="unblock")
        finally:
            score.wait = old_wait
            score.transactions.pop(thread, None)


def run_tests():
    blankettestlib.run(name="blanket.primitives.internal_coverage", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
