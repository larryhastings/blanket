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

from blanket import Scenario, Not, Call, Use, State, Terminated, Nested, CompetingDriversError
from blanket import primitives as primitives_module

NEVER = 1e9


class FakeDriver:
    def __init__(self, score, name, *, signals=(), done=False):
        self.score = score
        self.thread = types.SimpleNamespace(name=name)
        self.state = score.Driver.idle
        self.owner = None
        self.signals = frozenset(signals)
        self.done = done
        self.closed = 0
        self.calls = []
        self.reactivated = 0

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

    def __call__(self):
        self.calls.append(("call",))

    def drive(self):
        self.calls.append(("drive",))
        self.done = False

    def reactivate(self):
        self.reactivated += 1
        self.done = False
        self.calls.append(("reactivate",))

    def signal(self, signals):
        self.calls.append(("signal", frozenset(signals)))
        self.signals = frozenset()
        return self.signals


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
        self.assertEqual(d.signal({"signal"}), frozenset())
        self.assertIn(("signal", frozenset({"signal"})), d.calls)

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

    def test_dispatch_drains_done_driver_by_reactivating(self):
        score = Scenario()._core
        dispatch = score.Dispatch()
        d = FakeDriver(score, "done", done=True)
        dispatch.add(d)

        yielded = next(dispatch)
        self.assertIs(yielded, d)
        self.assertEqual(d.reactivated, 1)
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
    def test_core_driver_claim_slot_and_reactivate_errors(self):
        scenario = Scenario()
        score = scenario._core
        thread = threading.Thread(target=lambda: None, name="driver-errors")
        d1 = score.Driver(thread)
        d2 = score.Driver(thread)
        score.drivers[thread] = d1
        with self.assertRaises(CompetingDriversError):
            d2.claim_slot()

        d1.state = d1.active
        with self.assertRaisesRegex(RuntimeError, "already in"):
            d1.reactivate()

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
        with self.assertRaisesRegex(RuntimeError, "advanced past PAUSED"):
            plain_tx.unpause()
        with self.assertRaisesRegex(RuntimeError, "advanced past PAUSED"):
            plain_tx.set_pause(True)
        with self.assertRaisesRegex(RuntimeError, "not in a scheduler-controlled"):
            plain_tx.unstick()

        pause_tx = core.methods[lock.release](lock.release, time.monotonic(), regulated=False)
        pause_tx.set_pause(True)
        self.assertTrue(pause_tx.pause)
        self.assertEqual(pause_tx.pausing, 1)
        pause_tx.set_pause(False)
        self.assertFalse(pause_tx.pause)
        self.assertEqual(pause_tx.pausing, 0)
        pause_tx.pausing = 1
        pause_tx.unpausing()
        self.assertEqual(pause_tx.pausing, 0)

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
        cycle.core = types.SimpleNamespace(
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
                self.state = state
                self.terminated = object()
                self.finished = object()
                self.tx = tx
                self.called = 0
                self.finished_calls = 0

            def __call__(self):
                self.called += 1

            def finish(self):
                self.finished_calls += 1
                held["value"] = False
                self.state = self.finished

        prev = Previous(object(), types.SimpleNamespace(method=release_method))
        prev.state = prev.terminated
        cycle.previous = prev
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            cycle.ensure_ul_free()
        self.assertEqual(prev.called, 1)

        prev = Previous(object(), None)
        cycle.previous = prev
        with self.assertRaisesRegex(RuntimeError, "not at lock.release"):
            cycle.ensure_ul_free()

        held["value"] = True
        prev = Previous(object(), types.SimpleNamespace(method=release_method))
        cycle.previous = prev
        cycle.ensure_ul_free()
        self.assertEqual(prev.called, 2)
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
                self.finished = object()
                self.state = self.active
                self.tx = types.SimpleNamespace(
                    method=notify_method,
                    n=notify_n,
                    state=State.BLOCKED,
                    result=raised,
                    validate=lambda **kwargs: None,
                )
                self.finish_calls = 0
                self.drive_calls = 0

            def finish(self):
                self.finish_calls += 1

            def __call__(self):
                self.drive_calls += 1
                self.state = self.finished
                if raised is not None:
                    self.tx.state = State.RAISED

            def reactivate(self):
                self.state = self.active

        waker = Waker()
        cycle.incoming = [waker]
        return cycle, waker

    def test_drive_waker_rejects_finite_notify_with_extra_waiters(self):
        cycle, waker = self.make_cycle(
            actual_waiters=2, managed_waiters=1, notify_n=1)
        with self.assertRaisesRegex(ValueError, "extra waiters"):
            cycle.drive_waker()
        waker.state = waker.finished
        waker.reactivate()
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
                self.pausing = 1
                self.state = State.PAUSED
                self.result = None
            def unpausing(self):
                self.pausing -= 1

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
        self.assertEqual(d1.tx.pausing, 1)
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

    def test_transaction_unpausing_from_paused_unparks(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.release](
            lock.release, time.monotonic(), regulated=False)
        released = []
        tx.state = State.PAUSED
        tx.pausing = 1
        blocker = threading.Lock()
        blocker.acquire()
        tx.blocker = blocker
        tx.unpausing()
        self.assertEqual(tx.pausing, 0)
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
    def make_driver_with_tx(self, *, base=False, tx_state=State.BLOCKED):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        score = scenario._core
        core = lock._core
        thread = threading.Thread(target=lambda: None, name="driver-direct")
        parent = None
        if base:
            parent = core.methods[lock.locked](
                lock.locked, time.monotonic(), regulated=False)
        tx = core.methods[lock.acquire](
            lock.acquire, time.monotonic(), regulated=False)
        tx.state = tx_state
        if parent is not None:
            tx.parent = parent
            parent.child = tx
        score.transactions[thread] = tx
        score.transaction_apis[thread] = tx.api
        score.entered = True
        driver = score.Driver(thread, parent if parent is not None else None)
        return scenario, lock, thread, parent, tx, driver

    def cleanup_driver_fixture(self, scenario, thread):
        scenario._core.transactions.pop(thread, None)
        scenario._core.transaction_apis.pop(thread, None)
        scenario._core.drivers.pop(thread, None)
        scenario._core.entered = False

    def test_base_driver_idle_nested_signal_surfaces_child(self):
        fixture = self.make_driver_with_tx(base=True)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.initialize()
            self.assertIs(driver.state, driver.active)
            driver.to(driver.idle)
            self.assertIn(driver._base_nested, driver.signals)
            driver.signal({driver._base_nested})
            self.assertIs(driver.state, driver.active)
            self.assertIs(driver.tx, tx)
            self.assertIs(driver.base, tx)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_active_driver_terminated_signal_becomes_terminated(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.initialize()
            self.assertIs(driver.state, driver.active)
            driver.signal({driver.thread_signal[Terminated]})
            self.assertIs(driver.state, driver.terminated)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_active_driver_tx_terminal_signal_raises(self):
        fixture = self.make_driver_with_tx(tx_state=State.RETURNED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.initialize()
            with self.assertRaisesRegex(RuntimeError, "tx terminated while Driver was ACTIVE"):
                driver.signal({tx})
            self.assertIs(driver.state, driver.raised)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_pursue_rejects_terminal_cached_tx(self):
        fixture = self.make_driver_with_tx(tx_state=State.RETURNED)
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.initialize()
            self.assertIs(driver.state, driver.active)
            with self.assertRaisesRegex(RuntimeError, "tx currently in"):
                driver.finish()
            self.assertIs(driver.state, driver.active)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_pursue_refreshes_stale_active_tx_to_finished(self):
        fixture = self.make_driver_with_tx()
        scenario, lock, thread, base_tx, tx, driver = fixture
        try:
            driver.initialize()
            self.assertIs(driver.state, driver.active)
            driver.cache_tx = lambda: setattr(driver, "tx", None)
            self.assertIsNone(driver.finish())
            self.assertIs(driver.state, driver.finished)
        finally:
            self.cleanup_driver_fixture(scenario, thread)


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
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.active = score.Driver.active
            self.finished = score.Driver.finished
            self.parked = score.Driver.parked
            self.tx = types.SimpleNamespace(
                method=method,
                state=tx_state,
                result=result,
            )
            self.finish_calls = 0
            self.pause_calls = 0
            self.drive_calls = 0
            self.on_call = None
        def finish(self, autoskip=False):
            self.finish_calls += 1
            self.state = self.finished
        def pause(self, autoskip=False):
            self.pause_calls += 1
            self.state = self.parked
        def __call__(self):
            if self.on_call is not None:
                return self.on_call(self)
            self.drive_calls += 1
            self.state = self.finished

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
            scenario._core.Driver.active, lock.acquire, State.BLOCKED)
        driver.pause()
        self.assertEqual(driver.pause_calls, 1)
        driver()
        self.assertEqual(driver.drive_calls, 1)
        self.assertIs(driver.state, driver.finished)

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
            scenario._core, acquirer, scenario._core.Driver.active,
            lock.release, State.BLOCKED)
        with self.assertRaisesRegex(RuntimeError, "expected 'assign-wrong' to call"):
            self.with_fake_driver(
                scenario, [wrong_method],
                lambda: lock._core.assign(acquirer, None, False))

        wrong_state = self.FakeAssignDriver(
            scenario._core, acquirer, scenario._core.Driver.active,
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
                scenario._core, releaser, scenario._core.Driver.active,
                lock.release, State.BLOCKED)
            a = self.FakeAssignDriver(
                scenario._core, acquirer, scenario._core.Driver.active,
                lock.acquire, State.BLOCKED)
            def releaser_call(driver):
                driver.drive_calls += 1
                driver.state = driver.finished
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
            scenario._core, acquirer, scenario._core.Driver.active,
            lock.acquire, State.BLOCKED)
        def acquirer_call(self):
            self.drive_calls += 1
            self.state = self.finished
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
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.active = score.Driver.active
            self.finished = score.Driver.finished
            self.parked = score.Driver.parked
            self.base_tx = base_tx
            self.tx = types.SimpleNamespace(
                method=method,
                state=tx_state,
                result=result,
                timed_out=False,
                timeout=None,
                kwargs={'blocking': True},
                n=1,
                thread=self.thread,
            )
            self.finish_calls = 0
            self.pause_calls = 0
        def __call__(self):
            self.calls.append(("call",))
        def finish(self, autoskip=False):
            self.finish_calls += 1
            self.state = self.finished
        def pause(self, autoskip=False):
            self.pause_calls += 1
            self.state = self.parked

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
            scenario._core, "alloc-default", scenario._core.Driver.active,
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
            scenario._core, "alloc-wrong-method", scenario._core.Driver.active,
            object(), State.BLOCKED)
        with self.assertRaisesRegex(ValueError, "calling acquire or release"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(core.allocate([(d.thread, None)])))

        d = self.FakeAllocateDriver(
            scenario._core, "alloc-wrong-state", scenario._core.Driver.active,
            sem.acquire, State.COMMIT)
        with self.assertRaisesRegex(RuntimeError, "must be at BLOCKED"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(core.allocate([(d.thread, None)])))

    def test_allocate_drive_reports_false_result(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-timeout", scenario._core.Driver.active,
            sem.acquire, State.BLOCKED, result=False)
        d.tx.timed_out = True
        d.tx.kwargs = {'blocking': False}
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_reports_unexpected_start_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        strange = types.SimpleNamespace(name='STRANGE')
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-strange-start", strange,
            sem.acquire, State.BLOCKED)
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver state"):
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
            scenario._core, "alloc-release-unknown-bound", scenario._core.Driver.active,
            sem.release, State.BLOCKED)
        result = self.with_fake_allocate_drivers(
            scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(result, [])
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_zero_timeout_acquire_can_run(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-zero-timeout", scenario._core.Driver.active,
            sem.acquire, State.BLOCKED, result=True)
        d.tx.timeout = 0
        result = self.with_fake_allocate_drivers(
            scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(result, [d.thread])
        self.assertEqual(d.finish_calls, 1)

    def test_allocate_pause_reports_unexpected_finish_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-pause-strange", scenario._core.Driver.active,
            sem.acquire, State.BLOCKED, result=True)
        def strange_pause(autoskip=False):
            d.pause_calls += 1
            d.state = d.finished
        d.pause = strange_pause
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver state"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)], pause=True)))
        self.assertEqual(d.pause_calls, 1)

    def test_allocate_finish_reports_unexpected_finish_state(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        d = self.FakeAllocateDriver(
            scenario._core, "alloc-finish-strange", scenario._core.Driver.active,
            sem.release, State.BLOCKED)
        def strange_finish(autoskip=False):
            d.finish_calls += 1
            d.state = d.active
        d.finish = strange_finish
        with self.assertRaisesRegex(RuntimeError, "unexpected Driver state"):
            self.with_fake_allocate_drivers(
                scenario, [d], lambda: list(sem._core.allocate([(d.thread, None)])))
        self.assertEqual(d.finish_calls, 1)

class TestFinalPrimitiveCoverageEdges(unittest.TestCase):
    def make_driver_fixture(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        score = scenario._core
        core = lock._core
        thread = threading.Thread(target=lambda: None, name="driver-final")
        parent = core.methods[lock.acquire](lock.acquire, time.monotonic(), regulated=False)
        child = core.methods[lock.locked](lock.locked, time.monotonic(), regulated=False)
        parent.state = State.BLOCKED
        child.state = State.BLOCKED
        child.parent = parent
        parent.child = child
        score.transactions[thread] = parent
        score.transaction_apis[thread] = parent.api
        score.entered = True
        driver = score.Driver(thread)
        driver.initialize()
        self.assertIs(driver.tx, parent)
        return scenario, lock, thread, parent, child, driver

    def cleanup_driver_fixture(self, scenario, thread):
        score = scenario._core
        score.transactions.pop(thread, None)
        score.transaction_apis.pop(thread, None)
        score.drivers.pop(thread, None)
        score.entered = False

    def test_driver_delegate_parking_nested_signal_rotates_to_child(self):
        scenario, lock, thread, parent, child, driver = self.make_driver_fixture()
        try:
            parent.is_delegate = lambda candidate: candidate is child
            driver.target = State.WAITING
            driver.to(driver.parking)
            scenario._core.transactions[thread] = child
            scenario._core.transaction_apis[thread] = child.api
            signals = {driver.thread_signal[Nested]}
            driver.signal(signals)
            self.assertIs(driver.state, driver.parking)
            self.assertIs(driver.tx, child)
            self.assertIs(driver.base, child)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_stacked_parking_parent_terminal_overshoots(self):
        scenario, lock, thread, parent, child, driver = self.make_driver_fixture()
        try:
            driver.tx = child
            driver.state = driver.parking
            driver.target = State.WAITING
            driver.stack.append((parent, driver.parking, State.WAITING, parent))
            driver.cache_tx = lambda: setattr(driver, "tx", None)
            parent.state = State.RETURNED
            with self.assertRaisesRegex(RuntimeError, "overshot target WAITING"):
                driver.signal({child})
            self.assertIs(driver.state, driver.raised)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_stacked_repop_parking_overshoots(self):
        scenario, lock, thread, parent, child, driver = self.make_driver_fixture()
        try:
            grandparent = parent
            parent2 = types.SimpleNamespace(state=State.RETURNED)
            child.parent = parent
            driver.tx = child
            driver.state = driver.skipping
            driver.target = State.WAITING
            driver.stack.extend([
                (grandparent, driver.parking, State.WAITING, grandparent),
                (parent, driver.skipping, State.WAITING, parent),
            ])
            survivor = types.SimpleNamespace(api=object(), parent=parent2)
            driver.cache_tx = lambda: setattr(driver, "tx", survivor)
            with self.assertRaisesRegex(RuntimeError, "overshot target WAITING"):
                driver.signal({child})
            self.assertIs(driver.state, driver.raised)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

    def test_driver_parking_canary_overshoots(self):
        scenario, lock, thread, parent, child, driver = self.make_driver_fixture()
        try:
            driver.target = State.WAITING
            driver.to(driver.parking)
            with self.assertRaisesRegex(RuntimeError, "reached COMMITTED"):
                driver.signal({driver.thread_signal[State.COMMITTED]})
            self.assertIs(driver.state, driver.raised)
        finally:
            self.cleanup_driver_fixture(scenario, thread)

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
        thread = threading.Thread(target=lambda: None, name="wf-no-outcome")
        tx = types.SimpleNamespace(
            method=core.primitive.wait_for,
            state=State.COMMIT,
            api=object(),
            validate=lambda **kwargs: None,
        )
        class Driverish:
            pass
        d = Driverish()
        d.thread = thread
        d.tx = tx
        d.state = scenario._core.Driver.active
        d.listen_predicate = False
        d.reentered = scenario._core.Driver.reentered
        d.pausing = lambda: None
        d.__call__ = lambda self=d: None
        d.__class__.__call__ = lambda self: None
        with self.assertRaisesRegex(RuntimeError, "predicate neither waited nor succeeded"):
            cycle.drive_waiter(d)
        self.assertTrue(d.listen_predicate)


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
        d.state = score.Driver.parked
        d.reentered = score.Driver.reentered
        d.listen_predicate = False
        d.finish_calls = 0
        d.pause_calls = 0
        d.call_count = 0
        def finish():
            d.finish_calls += 1
        def pause():
            d.pause_calls += 1
        def call():
            d.call_count += 1
            d.state = d.reentered
        d.finish = finish
        d.pause = pause
        d.__class__.__call__ = lambda self: call()
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
        self.assertFalse(d.listen_predicate)

    def test_condition_cycle_wake_reentered_exits_on_parent_transaction(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'wake',
            [
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {wf},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)
        self.assertFalse(d.listen_predicate)
        self.assertEqual(d.finish_calls, 1)

    def test_condition_cycle_pause_reentered_exits_on_paused_parent(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'pause',
            [
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {primitives_module.Paused(wf)},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)
        self.assertFalse(d.listen_predicate)
        self.assertEqual(d.pause_calls, 1)

    def test_condition_cycle_wake_reentered_returns_on_thread_termination(self):
        wf, cycle, d, calls = self.run_act_one_with_waits(
            'wake',
            [
                lambda wf, thread: {Not(primitives_module.Predicate(wf))},
                lambda wf, thread: {Terminated(thread)},
            ])
        self.assertEqual(cycle.scheduler_calls, [wf])
        self.assertIs(cycle.previous, d)
        self.assertFalse(d.listen_predicate)

    def test_condition_cycle_pause_reentered_complains_if_predicate_waited_again(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {Not(primitives_module.Predicate(wf))},
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
        self.assertTrue(d.listen_predicate)
        self.assertIsNone(cycle.previous)

    def test_condition_cycle_wake_reentered_complains_if_predicate_waited_again(self):
        scenario, score, wf, wait_tx, cycle, d = self.make_cycle_and_driver()
        original_wait = score.wait
        results = [
            lambda wf, thread: {Not(primitives_module.Predicate(wf))},
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
        self.assertTrue(d.listen_predicate)
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
        def validate(self, **kwargs):
            return None

    class CycleDriver:
        def __init__(self, score, thread, method, tx_state):
            self.thread = thread
            self.state = score.Driver.active
            self.impasse = score.Driver.impasse
            self.terminated = score.Driver.terminated
            self.active = score.Driver.active
            self.parked = score.Driver.parked
            self.done = False
            self.closed = 0
            self.tx = TestRemainingCycleErrorLineCoverage.Tx(method, tx_state)
            self.base_tx = None
            self.wait_calls = 0
            self.pausing_calls = 0
        def close(self):
            self.closed += 1
            self.done = True
        def wait(self):
            self.wait_calls += 1
            self.state = self.parked
            self.tx.state = State.WAITING
        def stall(self):
            self.state = self.parked
            self.tx.state = State.STALLED
        def pausing(self):
            self.pausing_calls += 1
            self.state = self.parked
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
        self.assertIs(driver.state, driver.parked)
        self.assertIs(driver.tx.state, State.STALLED)
        driver.pausing()
        self.assertEqual(driver.pausing_calls, 1)
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
        waiter_driver.wait = failing_wait
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

    def test_driver_stacked_repop_falls_off_stack_to_active_survivor(self):
        fixture = TestFinalPrimitiveCoverageEdges().make_driver_fixture()
        scenario, lock, thread, parent, child, driver = fixture
        try:
            other = types.SimpleNamespace(state=State.RETURNED)
            survivor = types.SimpleNamespace(api=object(), parent=other)
            driver.tx = child
            driver.state = driver.skipping
            driver.target = State.WAITING
            hidden_parent = types.SimpleNamespace(state=State.RETURNED)
            driver.stack.extend([
                (hidden_parent, driver.skipping, State.WAITING, object()),
                (parent, driver.skipping, State.WAITING, parent),
            ])
            driver.cache_tx = lambda: setattr(driver, "tx", survivor)
            driver.signal({child})
            self.assertIs(driver.state, driver.active)
            self.assertIs(driver.tx, survivor)
            self.assertIs(driver.base, survivor)
        finally:
            TestFinalPrimitiveCoverageEdges().cleanup_driver_fixture(scenario, thread)

    def test_transaction_call_with_post_commit_state_skips_commit_block(self):
        import time
        scenario = Scenario()
        lock = scenario.Lock()
        tx = lock._core.methods[lock.locked](
            lock.locked, time.monotonic(), regulated=False)
        tx.state = State.COMMITTED
        tx.raised = False
        tx.timed_out = False
        tx.pausing = 0
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
            self.method = types.SimpleNamespace(__name__='fake_queue_call')
            self.api = object()

    class FakeDriver:
        idle = types.SimpleNamespace(name='IDLE')
        active = types.SimpleNamespace(name='ACTIVE')
        finished = types.SimpleNamespace(name='FINISHED')
        terminated = types.SimpleNamespace(name='TERMINATED')
        impasse = types.SimpleNamespace(name='IMPASSE')
        strange = types.SimpleNamespace(name='STRANGE')

        def __init__(self, thread, base_tx, *, start_state=None, end_state=None, tx=None):
            self.thread = thread
            self.base_tx = base_tx
            self.start_state = self.active if start_state is None else start_state
            self.end_state = self.finished if end_state is None else end_state
            self.tx = (types.SimpleNamespace(
                role='get',
                state=State.BLOCKED,
                result=None,
                method=types.SimpleNamespace(__name__='fake_queue_call'),
                api=object()) if tx is None else tx)
            self.state = self.idle
            self.calls = 0
            self.finished_called = False
            self.closed = False

        @property
        def done(self):
            return self.state in (self.finished, self.terminated, self.impasse)

        def __call__(self):
            self.calls += 1
            self.state = self.start_state if self.calls == 1 else self.end_state

        def finish(self, autoskip=False):
            self.finished_called = True
            self.autoskip = autoskip

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
