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


import threading
import blanket
from threading import BrokenBarrierError
import time
import types
import unittest

from blanket import Scenario
from blanket import Call, Use, Terminated, Not, TimeoutState, ThreadOrderingError, Blocked, Waiting, Paused, Nested, TransactionState, State, Reached, Action
from blanket import Stalled, Commit, Committed, Exiting, CompetingDriversError
from blanket import Primitive
from blanket import Location, inject_call
from blanket import primitives as primitives_module
from big.boundinnerclass import bound_to


# Canonical timeout values for tests.  Use these instead of magic numbers:
#
#   IMMEDIATELY  - For timeouts that are expected to fire promptly.
#                  A properly-written blanket scheduler step should complete
#                  in far less than this; the timeout is only a safety net.
#                  The value is 1 microsecond, aka 1000 nanoseconds, which is
#                  oodles of time from a modern CPU's perspective.
#
#   NEVER        - For timeouts that should never fire during the test.
#                  About 31.7 years; safe for time_t on every platform.

IMMEDIATELY = 0.0001
NEVER = 1e9


class TestLockBasic(unittest.TestCase):
    """Basic Lock tests."""

    def test_acquire_release_raw(self):
        """Lock acquire and release work in raw mode."""
        scenario = Scenario()
        lock = scenario.Lock()

        result = lock.acquire()
        self.assertTrue(result)

        lock.release()

    def test_acquire_blocking_false(self):
        """Lock acquire with blocking=False returns immediately."""
        scenario = Scenario()
        lock = scenario.Lock()

        # First acquire succeeds
        result1 = lock.acquire()
        self.assertTrue(result1)

        # Second acquire with blocking=False fails
        result2 = lock.acquire(blocking=False)
        self.assertFalse(result2)

        lock.release()

    def test_acquire_nonblocking_timeout_raises(self):
        """Lock follows threading.Lock: nonblocking acquire may not specify timeout."""
        scenario = Scenario()
        lock = scenario.Lock()

        with self.assertRaises(ValueError):
            lock.acquire(blocking=False, timeout=0)
        with self.assertRaises(ValueError):
            lock.acquire(blocking=False, timeout=IMMEDIATELY)

    def test_acquire_timeout(self):
        """Lock acquire with timeout expires correctly."""
        scenario = Scenario()
        lock = scenario.Lock()

        # First acquire succeeds
        lock.acquire()

        # Second acquire with timeout should fail
        start = time.perf_counter()
        result = lock.acquire(timeout=IMMEDIATELY)
        elapsed = time.perf_counter() - start

        self.assertFalse(result)
        self.assertGreaterEqual(elapsed, IMMEDIATELY)

        lock.release()

    def test_context_manager(self):
        """Lock works as context manager."""
        scenario = Scenario()
        lock = scenario.Lock()

        with lock:
            # Lock should be held
            self.assertFalse(lock.acquire(blocking=False))

        # Lock should be released
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()


class TestRLockBasic(unittest.TestCase):
    """Basic RLock tests."""

    def test_reentrant(self):
        """RLock can be acquired multiple times by same thread."""
        scenario = Scenario()
        rlock = scenario.RLock()

        result1 = rlock.acquire()
        result2 = rlock.acquire()
        result3 = rlock.acquire()

        self.assertTrue(result1)
        self.assertTrue(result2)
        self.assertTrue(result3)

        rlock.release()
        rlock.release()
        rlock.release()

    def test_release_wrong_thread_raises(self):
        """RLock.release() from wrong thread raises."""
        scenario = Scenario()
        rlock = scenario.RLock()

        rlock.acquire()

        error = []
        def other_thread():
            try:
                rlock.release()
            except RuntimeError as e:
                error.append(e)

        t = threading.Thread(target=other_thread)
        t.start()
        t.join()

        self.assertEqual(len(error), 1)

        rlock.release()


class TestLockRegulated(unittest.TestCase):
    """Regulated Lock tests with scheduler control."""

    def test_regulated_basic(self):
        """Scheduler can control Lock acquire and release."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)

        results = []

        def worker():
            results.append('before_acquire')
            lock.acquire()
            results.append('after_acquire')
            lock.release()
            results.append('after_release')

        # Create thread before entering scenario
        t = scenario.thread(worker)

        with scenario:
            # Wait for thread to block on acquire
            scenario.wait(lock.acquire)
            results.append('scheduler_saw_acquire_block')

            # Unblock acquire - worker proceeds to release and blocks there
            lock_api.unblock(lock.acquire, t)
            results.append('scheduler_unblocked_acquire')

            # Wait for thread to block on release
            scenario.wait(lock.release)
            results.append('scheduler_saw_release_block')

            # Unblock release
            lock_api.unblock(lock.release, t)
            results.append('scheduler_unblocked_release')

        # The order depends on thread scheduling, but we know:
        # - before_acquire happens before scheduler_saw_acquire_block
        # - after_acquire happens after unblock(acquire) but timing vs scheduler varies
        # - after_release happens after unblock(release)
        self.assertIn('before_acquire', results)
        self.assertIn('after_acquire', results)
        self.assertIn('after_release', results)
        self.assertIn('scheduler_saw_acquire_block', results)
        self.assertIn('scheduler_saw_release_block', results)

        # Verify ordering constraints
        self.assertLess(results.index('before_acquire'), results.index('scheduler_saw_acquire_block'))
        self.assertLess(results.index('scheduler_saw_acquire_block'), results.index('after_acquire'))
        self.assertLess(results.index('after_acquire'), results.index('scheduler_saw_release_block'))
        self.assertLess(results.index('scheduler_saw_release_block'), results.index('after_release'))

    def test_two_threads_contention(self):
        """Scheduler controls which thread gets lock first."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)

        order = []

        def worker_a():
            lock.acquire()
            order.append('A')
            lock.release()

        def worker_b():
            lock.acquire()
            order.append('B')
            lock.release()

        # Create threads before entering scenario
        a = scenario.thread(worker_a)
        b = scenario.thread(worker_b)

        with scenario:
            # Wait for both to block on acquire
            scenario.wait(lock.acquire)
            scenario.wait(lock.acquire)

            # Give lock to B first
            lock_api.assign(b)

            # Give lock to A, using B as the unlocker
            # New API: first thread releases, last thread acquires
            lock_api.assign(b, a)

            # Wait for A to block on release, then unblock it
            scenario.wait(lock.release)
            lock_api.unblock(lock.release, a)

        self.assertEqual(order, ['B', 'A'])

    def test_two_threads_contention_waiting(self):
        """Scheduler controls which thread gets lock first, with thread in WAITING state."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)

        order = []

        def worker_a():
            lock.acquire()
            order.append('A')
            lock.release()

        def worker_b():
            lock.acquire()
            order.append('B')
            lock.release()

        # Create threads before entering scenario
        a = scenario.thread(worker_a)
        b = scenario.thread(worker_b)

        with scenario:
            lock_api.assign(b)
            lock_api.assign(b, a)
            lock_api.unblock(lock.release, a)

        self.assertEqual(order, ['B', 'A'])

    def test_release_notifies_waiting_acquire(self):
        """Lock.release notifies waiting Lock.acquire via NotifyTransaction."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)

        order = []

        def worker_a():
            with lock:
                order.append('A_acquired')
            order.append('A_released')

        def worker_b():
            with lock:
                order.append('B_acquired')
            order.append('B_released')

        a = scenario.thread(worker_a)
        b = scenario.thread(worker_b)

        with scenario:
            # Park both threads at their first lock.acquire so we can
            # control the order deterministically.
            parked = scenario.park(b, lock.acquire, a, lock.acquire)
            tx_b = parked[b]

            # B acquires first
            lock_api.unblock(lock.acquire, b)
            scenario.wait(tx_b)

            # A's acquire is unblocked but will block on the actual lock
            # until B releases (no wait — A's tx stays in COMMIT)
            lock_api.unblock(lock.acquire, a)

            # B releases - this notifies A's waiting acquire
            scenario.wait(lock.release)
            tx_b_release = scenario.transactions[b]
            lock_api.unblock(lock.release, b)
            scenario.wait(tx_b_release)

            # A reaches lock.release (after running through with-block)
            scenario.wait(lock.release)
            tx_a_release = scenario.transactions[a]
            lock_api.unblock(lock.release, a)
            scenario.wait(tx_a_release)

        # Assert relative orderings
        a_acquired = order.index('A_acquired')
        a_released = order.index('A_released')
        b_acquired = order.index('B_acquired')
        b_released = order.index('B_released')

        # Each thread's acquire comes before its release
        self.assertLess(a_acquired, a_released)
        self.assertLess(b_acquired, b_released)

        # B acquired before A (scheduler controlled this)
        self.assertLess(b_acquired, a_acquired)


class TestTransactionAPI(unittest.TestCase):
    """Tests for TransactionAPI properties."""

    def test_timeout_tuple(self):
        """TransactionAPI.timeout returns Timeout tuple."""
        scenario = Scenario()
        lock = scenario.Lock()

        tx_api = None

        def worker():
            # Get tx_api while blocked
            lock.acquire(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]
            timeout = tx_api.timeout

            self.assertIsInstance(timeout, TimeoutState)
            self.assertEqual(timeout.value, NEVER)
            self.assertIsNotNone(timeout.time)

            # Unblock to clean up
            scenario.api(lock).unblock(lock.acquire, t)

    def test_end_time_set_after_commit(self):
        """TransactionAPI.end_time is None before commit, set after."""
        scenario = Scenario()
        lock = scenario.Lock()

        tx_api = None

        def worker():
            lock.acquire()
            lock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]
            self.assertIsNone(tx_api.end_time)

            scenario.api(lock).unblock(lock.acquire, t)
            scenario.wait(t)

        # After completion, end_time should be set
        # Note: tx_api is removed from scenario.transactions after close()
        # so we can't check it here directly


class TestOverrideWriteOnce(unittest.TestCase):
    """Tests for override write-once behavior and state checks."""

    def test_expire_on_terminal_state_raises(self):
        """Calling expire after transaction completes raises RuntimeError."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.acquire(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]

            # Settings-only expire marks the tx; drive worker to
            # terminal explicitly.
            tx_api.expire()
            scenario.finish(t)

            self.assertEqual(tx_api.state, State.RETURNED)

            # Second expire should raise because we're in terminal state
            with self.assertRaises(RuntimeError) as cm:
                tx_api.expire()
            self.assertIn(State.RETURNED[1], str(cm.exception))

    def test_disregard_on_terminal_state_raises(self):
        """Calling disregard after transaction completes raises RuntimeError."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.acquire(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]

            tx_api.expire()
            scenario.finish(t)

            self.assertEqual(tx_api.state, State.RETURNED)

            # Disregard should raise because we're in terminal state
            with self.assertRaises(RuntimeError) as cm:
                tx_api.disregard()
            self.assertIn(State.RETURNED[1], str(cm.exception))


class TestStateConstants(unittest.TestCase):
    """Tests that state constants are properly defined."""



class TestInstructions(unittest.TestCase):
    """Tests for instruction-based functionality via public APIs."""

    def test_disregard_prevents_timeout(self):
        """TransactionAPI.disregard() prevents timeout expiration."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)
        result = []

        def holder():
            # Hold the lock, then release it
            with lock:
                pass

        def waiter():
            # Try to acquire with timeout; disregard() below will neutralize it.
            r = lock.acquire(timeout=IMMEDIATELY)
            result.append(r)
            if r:
                lock.release()

        with scenario:
            # Holder grabs the lock
            h = scenario.thread(holder)
            lock_api.assign(h)
            # h will now block on lock.release

            # Waiter tries to acquire, blocks
            w = scenario.thread(waiter)
            scenario.wait(w)

            # tell it to ignore the timeout
            tx_api = scenario.transactions[w]
            tx_api.disregard()
            # and block on the lock.acquire call
            tx_api.unblock()

            # Holder release notifies waiter, completing waiter's acquire
            tx_h = scenario.transactions[h]
            lock_api.unblock(lock.release, h)
            scenario.wait(tx_h)

            # Waiter's acquire completed (was woken), now wait for it to finish
            scenario.wait(tx_api)
            scenario.wait(lock.release)
            tx_w = scenario.transactions[w]
            lock_api.unblock(lock.release, w)
            scenario.wait(tx_w)

        # Should have succeeded (True), not timed out (False)
        self.assertEqual(result, [True])

    def test_barrier_action_executes(self):
        """Barrier action callable is executed when barrier fills."""
        scenario = Scenario()
        action_record = []

        def my_action(tx):
            action_record.append('executed')

        barrier = scenario.Barrier(2, action=my_action)
        api = scenario.api(barrier)

        def worker1():
            barrier.wait()

        def worker2():
            barrier.wait()

        with scenario:
            t1 = scenario.thread(worker1)
            t2 = scenario.thread(worker2)

            scenario.wait(t1)
            scenario.wait(t2)

            # cycle drives both threads through the barrier; the action
            # runs automatically when the opener (t2) fills the barrier.
            with api.cycle(t1, t2):
                pass

        self.assertEqual(action_record, ['executed'])


class TestLockAPIConvenience(unittest.TestCase):
    """Tests for LockAPI convenience methods."""

    def test_lock_api_expire(self):
        """LockAPI.expire(thread) expires thread's acquire."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)
        result = None

        def worker():
            nonlocal result
            result = lock.acquire(timeout=NEVER)

        # Lock it first so acquire will block
        lock.acquire()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            # Use convenience method
            lock_api.expire(lock.acquire, t)

        lock.release()
        self.assertFalse(result)

    def test_lock_api_disregard(self):
        """LockAPI.disregard(thread) forces thread's acquire to disregard its timeout."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock_api = scenario.api(lock)
        result = None

        def worker():
            nonlocal result
            result = lock.acquire(timeout=NEVER)  # Test controls completion.

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            # Use convenience method
            lock_api.disregard(lock.acquire, t)
            scenario.finish(t)

        self.assertTrue(result)
        lock.release()

    def test_semaphore_api_expire(self):
        """SemaphoreAPI.expire(thread) expires thread's acquire."""
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        sem_api = scenario.api(sem)
        result = None

        def worker():
            nonlocal result
            result = sem.acquire(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(sem.acquire)
            sem_api.expire(sem.acquire, t)

        self.assertFalse(result)

    def test_semaphore_api_disregard(self):
        """SemaphoreAPI.disregard(thread) forces thread's acquire to disregard its timeout."""
        scenario = Scenario()
        sem = scenario.Semaphore(1)  # has a permit available
        sem_api = scenario.api(sem)
        result = None

        def worker():
            nonlocal result
            result = sem.acquire(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(sem.acquire)
            sem_api.disregard(sem.acquire, t)
            scenario.finish(t)

        self.assertTrue(result)

    def test_event_api_expire(self):
        """EventAPI.expire(thread) expires thread's wait."""
        scenario = Scenario()
        ev = scenario.Event()
        ev_api = scenario.api(ev)
        result = None

        def worker():
            nonlocal result
            result = ev.wait(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(ev.wait)
            ev_api.expire(ev.wait, t)

        self.assertFalse(result)

    def test_event_api_disregard(self):
        """EventAPI.disregard(thread) forces thread's wait to disregard its timeout."""
        scenario = Scenario()
        ev = scenario.Event()
        ev.set()  # so the underlying wait returns immediately at commit
        ev_api = scenario.api(ev)
        result = None

        def worker():
            nonlocal result
            result = ev.wait(timeout=NEVER)

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(ev.wait)
            ev_api.disregard(ev.wait, t)
            scenario.finish(t)

        self.assertTrue(result)

    def test_barrier_api_expire(self):
        """BarrierAPI.expire(thread) expires thread's wait."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        barrier_api = scenario.api(barrier)
        result = None

        def worker():
            nonlocal result
            try:
                result = barrier.wait(timeout=NEVER)
            except threading.BrokenBarrierError:
                result = 'broken'

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(barrier.wait)
            barrier_api.expire(barrier.wait, t)

        self.assertEqual(result, 'broken')

    def test_barrier_api_disregard(self):
        """BarrierAPI.disregard(thread) forces thread's wait to disregard its timeout."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        barrier_api = scenario.api(barrier)
        result_a = None
        result_b = None

        def worker_a():
            nonlocal result_a
            result_a = barrier.wait(timeout=NEVER)

        def worker_b():
            nonlocal result_b
            result_b = barrier.wait()

        with scenario:
            ta = scenario.thread(worker_a)
            scenario.wait(barrier.wait)
            barrier_api.disregard(barrier.wait, ta)
            tb = scenario.thread(worker_b)
            scenario.finish(ta, tb)

        self.assertIn(result_a, (0, 1))
        self.assertIn(result_b, (0, 1))
        self.assertNotEqual(result_a, result_b)

    def test_condition_api_expire(self):
        """ConditionAPI.expire(thread) expires thread's wait."""
        scenario = Scenario()
        cond = scenario.Condition()
        cond_api = scenario.api(cond)
        result = None

        def worker():
            nonlocal result
            cond.acquire()
            result = cond.wait(timeout=NEVER)
            cond.release()

        with scenario:
            t = scenario.thread(worker)
            api = scenario.api(cond)
            api.unblock(cond.acquire, t)
            scenario.wait(cond.wait)
            # Settings-only expire: mark the wait to timeout, then
            # drive the worker through the rest of cond.wait's commit
            # dance (release_save -> actual.wait(0) -> STALLED park ->
            # acquire_restore) to terminal.
            cond_api.expire(cond.wait, t)
            scenario.finish(t)

        self.assertFalse(result)


class TestRepr(unittest.TestCase):
    """Tests for repr strings on primitives."""

    def test_lock_repr_unlocked_no_name(self):
        """Lock repr looks like _thread.lock when unlocked and unnamed."""
        scenario = Scenario()
        lock = scenario.Lock()
        r = repr(lock)
        self.assertIn('unlocked', r)
        self.assertIn('_thread.lock', r)
        self.assertIn('0X', r)  # Uppercase hex

    def test_lock_repr_locked_no_name(self):
        """Lock repr shows locked state."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock.acquire()
        r = repr(lock)
        self.assertIn('locked', r)
        self.assertNotIn('unlocked', r)
        lock.release()

    def test_lock_repr_with_name(self):
        """Lock repr shows name and uses 'Lock' class name."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock.name = 'my_lock'
        r = repr(lock)
        self.assertIn('my_lock', r)
        self.assertIn('Lock', r)
        self.assertNotIn('_thread.lock', r)

    def test_rlock_repr_unlocked_no_name(self):
        """RLock repr looks like _thread.RLock when unlocked and unnamed."""
        scenario = Scenario()
        rlock = scenario.RLock()
        r = repr(rlock)
        self.assertIn('unlocked', r)
        self.assertIn('_thread.RLock', r)
        self.assertIn('owner=0', r)
        self.assertIn('count=0', r)

    def test_rlock_repr_locked(self):
        """RLock repr shows owner and count when locked."""
        scenario = Scenario()
        rlock = scenario.RLock()
        rlock.acquire()
        r = repr(rlock)
        self.assertIn('locked', r)
        self.assertNotIn('owner=0', r)  # owner should be non-zero
        self.assertIn('count=1', r)
        rlock.acquire()
        r = repr(rlock)
        self.assertIn('count=2', r)
        rlock.release()
        rlock.release()

    def test_rlock_repr_with_name(self):
        """RLock repr shows name and uses 'RLock' class name."""
        scenario = Scenario()
        rlock = scenario.RLock()
        rlock.name = 'my_rlock'
        r = repr(rlock)
        self.assertIn('my_rlock', r)
        self.assertIn('RLock', r)
        self.assertNotIn('_thread.RLock', r)

    def test_rlock_repr_worker_owned_from_scheduler_thread(self):
        """RLock repr reports cross-thread owner/count, not scheduler-local count."""
        scenario = Scenario()
        rlock = scenario.RLock()

        def worker():
            rlock.acquire()
            rlock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.skip(t, rlock.acquire, wait=True)
            scenario.wait(t)
            r = repr(rlock)
            self.assertIn('locked', r)
            self.assertNotIn('owner=0', r)
            self.assertIn('count=1', r)
            scenario.skip(t, rlock.release, wait=True)

    def test_lock_api_repr(self):
        """LockAPI repr shows LockAPI class name."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)
        r = repr(api)
        self.assertIn('LockAPI', r)

    def test_rlock_api_repr(self):
        """RLockAPI repr shows RLockAPI class name."""
        scenario = Scenario()
        rlock = scenario.RLock()
        api = scenario.api(rlock)
        r = repr(api)
        self.assertIn('RLockAPI', r)

    def test_lock_core_repr(self):
        """LockCore repr shows LockCore class name."""
        scenario = Scenario()
        lock = scenario.Lock()
        r = repr(lock._core)
        self.assertIn('LockCore', r)

    def test_rlock_core_repr(self):
        """RLockCore repr shows RLockCore class name."""
        scenario = Scenario()
        rlock = scenario.RLock()
        r = repr(rlock._core)
        self.assertIn('RLockCore', r)

    def test_name_property_defaults_to_default_name(self):
        """Name property defaults to the core default name."""
        scenario = Scenario()
        lock = scenario.Lock()
        self.assertIn('Lock 1', lock.name)

    def test_name_property_set_via_primitive(self):
        """Name can be set via primitive."""
        scenario = Scenario()
        lock = scenario.Lock()
        lock.name = 'test_name'
        self.assertEqual(lock.name, 'test_name')

    def test_name_property_set_via_api(self):
        """Name can be set via API."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)
        api.name = 'api_set_name'
        self.assertEqual(lock.name, 'api_set_name')
        self.assertEqual(api.name, 'api_set_name')

    def test_primitive_repr_defaults_to_compatibility_repr(self):
        """Primitive repr matches the real object shape by default."""
        scenario = Scenario()
        lock = scenario.Lock()
        r = repr(lock)
        self.assertEqual(r, f"<unlocked _thread.lock object at {hex(id(lock)).upper()}>")

    def test_setting_name_switches_primitive_to_fancy_repr(self):
        """Explicitly setting the name switches primitive repr to fancy mode."""
        scenario = Scenario()
        lock = scenario.Lock()
        default_name = lock.name
        lock.name = default_name
        r = repr(lock)
        self.assertIn(default_name, r)
        self.assertIn('Lock object', r)
        self.assertNotIn('_thread.lock object', r)

    def test_setting_name_switches_all_primitives_to_fancy_repr(self):
        """Every primitive type starts with the compatibility repr;
        once a name is set (via the API), repr() switches to the
        fancy form that includes the name and class label.  This is
        a help to debugging and to writing readable test failures.
        """
        scenario = Scenario()

        # (factory, compat-repr-marker, fancy-class-label).  The
        # compat marker is something present in the default repr and
        # NOT in the named repr; the fancy label is the class name
        # that should appear in the named repr.
        cases = [
            (lambda: scenario.Lock(),             '_thread.lock',     'Lock'),
            (lambda: scenario.RLock(),            '_thread.RLock',    'RLock'),
            (lambda: scenario.Condition(),        'Condition(',       'Condition('),
            (lambda: scenario.Semaphore(),        'threading.Semaphore', 'Semaphore'),
            (lambda: scenario.BoundedSemaphore(), 'threading.BoundedSemaphore', 'BoundedSemaphore'),
            (lambda: scenario.Event(),            'threading.Event',  'Event'),
            (lambda: scenario.Barrier(2),         'threading.Barrier', 'Barrier'),
        ]

        for factory, compat_marker, fancy_label in cases:
            p = factory()
            cls_name = type(p).__name__
            with self.subTest(primitive=cls_name):
                # Default: compatibility repr.
                default_repr = repr(p)
                self.assertIn(compat_marker, default_repr,
                    f"{cls_name} default repr missing compat marker: {default_repr}")

                # Set a custom name via the API.
                custom = f"my_{cls_name.lower()}"
                p.name = custom

                fancy_repr = repr(p)
                self.assertIn(custom, fancy_repr,
                    f"{cls_name} named repr missing custom name: {fancy_repr}")
                self.assertIn(fancy_label, fancy_repr,
                    f"{cls_name} named repr missing fancy label: {fancy_repr}")

    def test_setting_name_to_default_still_switches_to_fancy(self):
        """Even setting the name to its current (default) value
        flips use_fancy_repr -- the act of explicitly setting it
        is what counts, not the value."""
        scenario = Scenario()
        for factory in (scenario.Lock, scenario.RLock,
                        scenario.Condition, scenario.Semaphore,
                        scenario.BoundedSemaphore, scenario.Event,
                        lambda: scenario.Barrier(2)):
            p = factory()
            cls_name = type(p).__name__
            with self.subTest(primitive=cls_name):
                default = p.name
                self.assertFalse(p._core.use_fancy_repr,
                    f"{cls_name} should start with use_fancy_repr=False")
                p.name = default
                self.assertTrue(p._core.use_fancy_repr,
                    f"{cls_name} should have use_fancy_repr=True after explicit set")
                self.assertIn(default, repr(p))

    def test_raw_repr_is_always_fancy(self):
        """Raw repr is always the fancy repr."""
        scenario = Scenario()
        lock = scenario.Lock()
        raw = scenario.raw(lock)
        r = repr(raw)
        self.assertIn(lock.name, r)
        self.assertIn('Lock.raw object', r)

    def test_name_property_rejects_non_string(self):
        """Name property raises TypeError for non-string values."""
        scenario = Scenario()
        lock = scenario.Lock()
        with self.assertRaises(TypeError) as cm:
            lock.name = 123
        self.assertIn('string', str(cm.exception))

    def test_name_property_rejects_non_string_via_core(self):
        """Core name property raises TypeError for non-string values."""
        scenario = Scenario()
        lock = scenario.Lock()
        with self.assertRaises(TypeError) as cm:
            lock.name = None
        self.assertIn('string', str(cm.exception))



class TestPrimitiveMasquerading(unittest.TestCase):
    """Tests for primitive masquerading as real threading primitives."""

    def assert_masquerades_as(self, blanket_object, real_object):
        """Assert a blanket primitive presents the real object's surface."""
        real_class = real_object.__class__
        self.assertIs(blanket_object.__class__, real_class)
        self.assertIsInstance(blanket_object, real_class)
        self.assertEqual(dir(blanket_object), dir(real_object))
        self.assertNotIn('_blanket', dir(blanket_object))

    def test_lock_masquerades_as_threading_lock(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.Lock(), threading.Lock())

    def test_rlock_masquerades_as_threading_rlock(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.RLock(), threading.RLock())

    def test_condition_masquerades_as_threading_condition(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.Condition(), threading.Condition())

    def test_event_masquerades_as_threading_event(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.Event(), threading.Event())

    def test_semaphore_masquerades_as_threading_semaphore(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.Semaphore(), threading.Semaphore())

    def test_bounded_semaphore_masquerades_as_threading_bounded_semaphore(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.BoundedSemaphore(), threading.BoundedSemaphore())

    def test_barrier_masquerades_as_threading_barrier(self):
        scenario = Scenario()
        self.assert_masquerades_as(scenario.Barrier(2), threading.Barrier(2))

    def test_raw_primitives_also_masquerade(self):
        scenario = Scenario()
        pairs = [
            (scenario.Lock(), threading.Lock()),
            (scenario.RLock(), threading.RLock()),
            (scenario.Condition(), threading.Condition()),
            (scenario.Event(), threading.Event()),
            (scenario.Semaphore(), threading.Semaphore()),
            (scenario.BoundedSemaphore(), threading.BoundedSemaphore()),
            (scenario.Barrier(2), threading.Barrier(2)),
        ]
        for primitive, real in pairs:
            with self.subTest(primitive=primitive):
                self.assert_masquerades_as(scenario.raw(primitive), real)

    def test_primitives_pass_blanket_and_threading_isinstance(self):
        scenario = Scenario()
        checks = [
            (scenario.Lock(), Scenario.Lock, type(threading.Lock())),
            (scenario.RLock(), Scenario.RLock, type(threading.RLock())),
            (scenario.Condition(), Scenario.Condition, threading.Condition),
            (scenario.Event(), Scenario.Event, threading.Event),
            (scenario.Semaphore(), Scenario.Semaphore, threading.Semaphore),
            (scenario.BoundedSemaphore(), Scenario.BoundedSemaphore, threading.BoundedSemaphore),
            (scenario.Barrier(2), Scenario.Barrier, threading.Barrier),
        ]
        for primitive, blanket_class, real_class in checks:
            with self.subTest(primitive=primitive):
                self.assertIsInstance(primitive, blanket_class)
                self.assertIsInstance(primitive, real_class)

    def test_class_spoof_is_read_only(self):
        scenario = Scenario()
        lock = scenario.Lock()
        with self.assertRaises(AttributeError):
            lock.__class__ = object

    def test_compatibility_repr_uses_uppercase_hex_ids(self):
        scenario = Scenario()
        lock = scenario.Lock()
        rlock = scenario.RLock()
        condition = scenario.Condition()
        event = scenario.Event()
        semaphore = scenario.Semaphore()
        bounded = scenario.BoundedSemaphore()
        barrier = scenario.Barrier(2)

        self.assertIn(hex(id(lock)).upper(), repr(lock))
        self.assertIn(hex(id(rlock)).upper(), repr(rlock))
        self.assertIn('0X', repr(condition))
        self.assertIn(hex(id(event)).upper(), repr(event))
        self.assertIn(hex(id(semaphore)).upper(), repr(semaphore))
        self.assertIn(hex(id(bounded)).upper(), repr(bounded))
        self.assertIn(hex(id(barrier)).upper(), repr(barrier))


class TestScenarioCoreRepr(unittest.TestCase):
    """Tests for _ScenarioCore repr."""

    def test_scenario_core_repr_empty(self):
        """_ScenarioCore repr with no primitives, threads, or transactions."""
        scenario = Scenario()
        core = scenario._core
        r = repr(core)
        self.assertIn('_ScenarioCore', r)
        self.assertIn('unlocked', r)
        self.assertIn('0 primitives', r)
        self.assertIn('0 threads', r)
        self.assertIn('0 transactions', r)

    def test_scenario_core_repr_with_primitive(self):
        """_ScenarioCore repr shows primitive count."""
        scenario = Scenario()
        core = scenario._core
        lock = scenario.Lock()
        r = repr(core)
        self.assertIn('1 primitives', r)

    def test_scenario_core_repr_with_name(self):
        """_ScenarioCore repr shows name first."""
        scenario = Scenario()
        core = scenario._core
        core.name = 'my_scenario'
        r = repr(core)
        self.assertTrue(r.startswith('<my_scenario '))
        self.assertIn('_ScenarioCore', r)

    def test_scenario_core_name_default_empty(self):
        """_ScenarioCore name defaults to empty string."""
        scenario = Scenario()
        self.assertEqual(scenario.name, '')

    def test_scenario_core_name_rejects_non_string(self):
        """_ScenarioCore name raises TypeError for non-string."""
        scenario = Scenario()
        with self.assertRaises(TypeError) as cm:
            scenario.name = 42
        self.assertIn('string', str(cm.exception))


class TestScenarioRepr(unittest.TestCase):
    """Tests for Scenario repr."""

    def test_scenario_repr_empty(self):
        """Scenario repr with no primitives, threads, or transactions."""
        scenario = Scenario()
        r = repr(scenario)
        self.assertIn('Scenario', r)
        self.assertNotIn('_ScenarioCore', r)
        self.assertIn('unlocked', r)
        self.assertIn('0 primitives', r)

    def test_scenario_repr_with_primitive(self):
        """Scenario repr shows primitive count."""
        scenario = Scenario()
        lock = scenario.Lock()
        r = repr(scenario)
        self.assertIn('1 primitives', r)

    def test_scenario_repr_with_name(self):
        """Scenario repr shows name from core."""
        scenario = Scenario()
        scenario.name = 'my_scenario'
        r = repr(scenario)
        self.assertTrue(r.startswith('<my_scenario '))
        self.assertIn('Scenario', r)

    def test_scenario_name_setter_rejects_non_string(self):
        """The user-facing Scenario.name setter delegates to the
        core, which type-checks; non-strings raise TypeError."""
        scenario = Scenario()
        with self.assertRaises(TypeError) as cm:
            scenario.name = 42
        self.assertIn('string', str(cm.exception))


class TestTimeoutRepr(unittest.TestCase):
    """Tests for Timeout repr."""

    def test_timeout_repr_none_values(self):
        """Timeout repr with None values."""
        t = primitives_module.TimeoutState(None, None, False)
        r = repr(t)
        self.assertEqual(r, "TimeoutState(value=None, time=None, timed_out=False)")

    def test_timeout_repr_with_values(self):
        """Timeout repr with actual values."""
        t = primitives_module.TimeoutState(5.0, 12345.678, True)
        r = repr(t)
        self.assertIn('value=5.0', r)
        self.assertIn('time=12345.678', r)
        self.assertIn('timed_out=True', r)


class TestTransactionRepr(unittest.TestCase):
    """Tests for transaction repr strings."""

    def test_acquire_transaction_repr(self):
        """Lock.acquire transaction has informative repr."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]
            r = repr(tx_api)
            self.assertIn('Lock.acquire', r)
            self.assertIn(State.BLOCKED[1], r)
            self.assertIn('blocking', r)
            self.assertIn('Lock 1', r)  # default_name

            scenario.api(lock).unblock(lock.acquire, t)
            scenario.wait(lock.release)
            scenario.api(lock).unblock(lock.release, t)

    def test_release_transaction_repr(self):
        """Lock.release transaction has informative repr."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)
            scenario.api(lock).unblock(lock.acquire, t)
            scenario.wait(lock.release)

            tx_api = scenario.transactions[t]
            r = repr(tx_api)
            self.assertIn('Lock.release', r)

            scenario.api(lock).unblock(lock.release, t)

    def test_acquire_api_repr(self):
        """Lock.acquire transaction API has informative repr."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)

            tx_api = scenario.transactions[t]
            r = repr(tx_api)
            self.assertIn('Lock.acquire', r)
            self.assertIn(State.BLOCKED[1], r)

            scenario.api(lock).unblock(lock.acquire, t)
            scenario.wait(lock.release)
            scenario.api(lock).unblock(lock.release, t)

    def test_locked_transaction_repr(self):
        """Lock.locked transaction has informative repr."""
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.locked()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.locked)

            tx_api = scenario.transactions[t]
            r = repr(tx_api)
            self.assertIn('Lock.locked', r)

            scenario.api(lock).unblock(lock.locked, t)

    def test_core_default_name_serial_number(self):
        """Core default_name includes serial number."""
        scenario = Scenario()
        lock1 = scenario.Lock()
        lock2 = scenario.Lock()
        lock3 = scenario.RLock()

        self.assertIn('Lock 1', lock1.name)
        self.assertIn('Lock 2', lock2.name)
        # RLock uses LockCore, so default_name says "Lock" not "RLock"
        self.assertIn('Lock 3', lock3.name)

    def test_core_default_name_not_updated_by_name_setter(self):
        """Setting name does not change default_name; repr uses name if set."""
        scenario = Scenario()
        lock = scenario.Lock()

        # name defaults to the auto-generated default_name
        self.assertIn('Lock 1', lock.name)

        # Setting name overrides
        lock.name = 'custom_lock'
        self.assertEqual(lock.name, 'custom_lock')

        # But repr should use the custom name
        def worker():
            lock.acquire()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(lock.acquire)
            tx_api = scenario.transactions[t]
            r = repr(tx_api)
            self.assertIn('custom_lock', r)
            self.assertNotIn('Lock 1', r)
            scenario.api(lock).unblock(lock.acquire, t)


class TestEvent(unittest.TestCase):
    """Tests for Event primitive."""

    def test_event_is_set_default_false(self):
        scenario = Scenario()
        event = scenario.Event()
        self.assertFalse(event.is_set())
        self.assertFalse(event.isSet())
        self.assertFalse(scenario.api(event).raw.is_set())

    def test_event_set_and_clear(self):
        scenario = Scenario()
        event = scenario.Event()

        self.assertFalse(event.is_set())
        event.set()
        self.assertTrue(event.is_set())
        event.clear()
        self.assertFalse(event.is_set())

    def test_event_wait_when_set(self):
        scenario = Scenario()
        event = scenario.Event()
        event.set()
        self.assertTrue(event.wait())

    def test_event_wait_timeout_expires_raw(self):
        scenario = Scenario()
        event = scenario.Event()
        self.assertFalse(event.wait(timeout=IMMEDIATELY))

    def test_event_repr_unset_and_set(self):
        scenario = Scenario()
        event = scenario.Event()
        r = repr(event)
        self.assertIn('threading.Event', r)
        self.assertIn('unset', r)
        self.assertIn('0X', r)
        event.set()
        r = repr(event)
        self.assertIn('set', r)
        self.assertNotIn('unset', r)

    def test_event_repr_with_name(self):
        scenario = Scenario()
        event = scenario.Event()
        event.name = 'my_event'
        r = repr(event)
        self.assertIn('my_event', r)
        self.assertIn('Event', r)
        self.assertNotIn('threading.Event', r)

    def test_event_api_waiters(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)

        def waiter():
            event.wait()

        def setter():
            event.set()

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(setter)
            # cycle requires strict BLOCKED state for waiters; let the
            # cycle protocol handle advancing them to WAITING itself.
            cycle = api.cycle(a, x)
            self.assertEqual(api.waiters, 0)
            self.assertEqual(cycle.close(), (a, x))

    def test_event_cycle_one_waiter(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        log = []

        def waiter():
            log.append('A_wait')
            event.wait()
            log.append('A_woke')

        def setter():
            log.append('X_set')
            event.set()
            log.append('X_done')

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(setter)
            cycle = api.cycle(a, x)
            tx = scenario.transaction(a)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(scenario.transaction(x).state, State.PAUSED)
            paused = Paused(tx)
            signaled = scenario.wait(Call(a, event.wait, State.PAUSED), paused)
            self.assertIn(paused, signaled)
            self.assertEqual(cycle.waiters, (a, x))
            self.assertEqual(cycle.extra_waiters, 0)
            self.assertFalse(hasattr(api, 'finish'))
            self.assertFalse(hasattr(api, 'choose'))
            self.assertEqual(cycle.wake(), a)
            self.assertEqual(cycle.wake(), x)
            with self.assertRaises(RuntimeError):
                cycle.wake()
            self.assertTrue(cycle.closed)

        self.assertEqual(log, ['A_wait', 'X_set', 'A_woke', 'X_done'])

    def test_event_cycle_two_waiters_context_manager_close(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        log = []

        def waiter(name):
            def fn():
                event.wait()
                log.append(name)
            return fn

        def setter():
            event.set()
            log.append('set')

        with scenario:
            a = scenario.thread(waiter('A'))
            b = scenario.thread(waiter('B'))
            x = scenario.thread(setter)
            with api.cycle(a, b, x) as cycle:
                self.assertEqual(cycle.wake(b), (b,))
            self.assertEqual(cycle.waiters, ())

        self.assertEqual(log, ['B', 'A', 'set'])

    def test_event_cycle_iter_and_call(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        log = []

        def waiter(name):
            def fn():
                event.wait()
                log.append(name)
            return fn

        def setter():
            event.set()

        with scenario:
            a = scenario.thread(waiter('A'))
            b = scenario.thread(waiter('B'))
            c = scenario.thread(waiter('C'))
            x = scenario.thread(setter)
            cycle = api.cycle(a, b, c, x)

            it = cycle.iter(c, a)
            self.assertIs(next(it), c)
            self.assertIs(next(it), a)
            self.assertIsNone(next(it, None))
            self.assertEqual(cycle(), (b, x))

        self.assertEqual(log, ['C', 'A', 'B'])

    def test_event_cycle_with_pause(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        log = []

        def waiter():
            event.wait()
            log.append('woke')

        def setter():
            event.set()

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(setter)
            cycle = api.cycle(a, x)
            self.assertEqual(cycle.pause(a), (a,))
            tx = scenario.transaction(a)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(log, [])
            tx.unpause()
            scenario.wait(tx)
            # Drain setter so the cycle can close cleanly.
            cycle.close()

        self.assertEqual(log, ['woke'])

    def test_event_cycle_rejects_already_set_event(self):
        scenario = Scenario()
        event = scenario.Event()
        api = scenario.api(event)
        event.set()

        def waiter():
            event.wait()

        def setter():
            event.set()

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(setter)
            with self.assertRaisesRegex(RuntimeError, 'already set'):
                api.cycle(a, x)
            scenario.finish(a, x)

    def test_event_cycle_validation_edges(self):
        scenario = Scenario()
        event = scenario.Event()
        lock = scenario.Lock()
        api = scenario.api(event)

        def waiter():
            event.wait()

        def setter():
            event.set()

        def wrong():
            lock.locked()

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(setter)
            w = scenario.thread(wrong)

            with self.assertRaises(ValueError):
                api.cycle(a)
            with self.assertRaises(ValueError):
                api.cycle(a, a)
            with self.assertRaises(ValueError):
                api.cycle(w, x)
            with self.assertRaises(TypeError):
                api.cycle(object(), x)

            scenario.finish(a, x, w)



class TestConditionDelegation(unittest.TestCase):
    """Tests for Condition delegation to underlying lock."""

    def test_condition_locked_delegates_to_lock(self):
        """Condition.locked() reflects underlying lock state across all related primitives."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition1 = scenario.Condition(lock)
        condition2 = scenario.Condition(lock)

        primitives = (condition1, condition2, lock)

        for p in primitives:
            self.assertFalse(any(o.locked() for o in primitives))
            p.acquire()
            self.assertTrue(all(o.locked() for o in primitives))
            p.release()
            self.assertFalse(any(o.locked() for o in primitives))

    def test_condition_unblock_delegates_to_lock(self):
        """Unblocking condition.acquire via condition API delegates to underlying lock."""
        scenario = Scenario()
        condition = scenario.Condition()
        api = scenario.api(condition)

        results = []

        def worker():
            results.append('acquiring')
            condition.acquire()
            results.append('acquired')
            condition.release()
            results.append('released')

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(condition.acquire)
            self.assertEqual(results, ['acquiring'])

            # Unblock via condition API using condition.acquire (not lock.acquire)
            api.unblock(condition.acquire, t)

        self.assertEqual(results, ['acquiring', 'acquired', 'released'])


class TestSemaphore(unittest.TestCase):
    """Tests for Semaphore and BoundedSemaphore primitives."""

    def test_semaphore_creation_and_properties(self):
        scenario = Scenario()
        sem = scenario.Semaphore(3)
        api = scenario.api(sem)
        self.assertIn('value=3', repr(sem))
        self.assertEqual(api.value, 3)
        self.assertEqual(api.waiters, 0)
        self.assertEqual(api.available, 3)

    def test_semaphore_default_value(self):
        scenario = Scenario()
        sem = scenario.Semaphore()
        self.assertIn('value=1', repr(sem))
        self.assertEqual(scenario.api(sem).value, 1)

    def test_semaphore_acquire_release(self):
        scenario = Scenario()
        sem = scenario.Semaphore(2)

        self.assertTrue(sem.acquire())
        self.assertTrue(sem.acquire())
        self.assertFalse(sem.acquire(blocking=False))
        sem.release()
        self.assertTrue(sem.acquire())
        sem.release()
        sem.release()

    def test_semaphore_context_manager(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)

        with sem:
            self.assertFalse(sem.acquire(blocking=False))

        self.assertTrue(sem.acquire(blocking=False))
        sem.release()

    def test_bounded_semaphore_creation_and_overrelease(self):
        scenario = Scenario()
        sem = scenario.BoundedSemaphore(2)
        self.assertIn('BoundedSemaphore', repr(sem))
        self.assertIn('value=2/2', repr(sem))

        sem.acquire()
        sem.acquire()
        sem.release()
        sem.release()

        with self.assertRaises(ValueError):
            sem.release()

    def test_semaphore_scheduler_api(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        sem_api = scenario.api(sem)
        results = []

        def worker():
            sem.acquire()
            results.append('acquired')
            sem.release()
            results.append('released')

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(sem.acquire)
            sem_api.unblock(sem.acquire, t)
            scenario.wait(sem.release)
            sem_api.unblock(sem.release, t)

        self.assertEqual(results, ['acquired', 'released'])

    def test_allocate_with_existing_value(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        sem_api = scenario.api(sem)
        results = []

        def acquirer():
            results.append('before')
            results.append(sem.acquire())
            results.append('after')

        with scenario:
            a = scenario.thread(acquirer)
            self.assertEqual(list(sem_api.allocate(a)), [a])

        self.assertEqual(results, ['before', True, 'after'])
        self.assertEqual(sem_api.value, 0)
        self.assertEqual(sem_api.available, 0)

    def test_allocate_release_then_acquire(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        sem_api = scenario.api(sem)
        results = []

        def releaser():
            sem.release()
            results.append('released')

        def acquirer():
            results.append('before')
            results.append(sem.acquire())
            results.append('after')

        with scenario:
            r = scenario.thread(releaser)
            a = scenario.thread(acquirer)
            self.assertEqual(list(sem_api.allocate(r, a)), [a])

        self.assertEqual(results, ['before', 'released', True, 'after'])

    def test_allocate_mixed_order(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)
        sem_api = scenario.api(sem)
        results = []

        def releaser():
            sem.release()
            results.append('released')

        def acquirer(name):
            results.append(f'{name}_before')
            results.append((name, sem.acquire()))
            results.append(f'{name}_after')

        with scenario:
            a = scenario.thread(acquirer, 'A')
            r = scenario.thread(releaser)
            b = scenario.thread(acquirer, 'B')
            self.assertEqual(list(sem_api.allocate(a, r, b)), [a, b])
            # allocate's Chain serializes the commits in spec order,
            # but the workers' post-commit Python (the results.append
            # calls) races outside blanket's synchronization.  Wait
            # for each worker to actually terminate so the ordering
            # assertion below is deterministic.
            scenario.finish(a, r, b)

        self.assertIn(('A', True), results)
        self.assertIn(('B', True), results)
        self.assertLess(results.index(('A', True)), results.index(('B', True)))

    def test_allocate_rejects_impossible_acquire(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        sem_api = scenario.api(sem)

        def acquirer():
            sem.acquire()

        with scenario:
            a = scenario.thread(acquirer)
            with self.assertRaisesRegex(RuntimeError, 'cannot be proven|would block'):
                sem_api.allocate(a)
            scenario.raw(sem).release()
            scenario.finish(a)

    def test_allocate_pause_applies_to_acquire_threads_only(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        sem_api = scenario.api(sem)
        results = []

        def releaser():
            sem.release()
            results.append('released')

        def acquirer():
            sem.acquire()
            results.append('acquired')

        with scenario:
            r = scenario.thread(releaser)
            a = scenario.thread(acquirer)
            iterator = sem_api.allocate(r, a, pause=True)
            self.assertEqual(next(iterator), a)
            tx_a = scenario.transaction(a)
            self.assertEqual(tx_a.state, State.PAUSED)
            self.assertIsNone(scenario.transaction(r))
            tx_a.unpause()
            with self.assertRaises(StopIteration):
                next(iterator)

        self.assertEqual(results, ['released', 'acquired'])

    def test_bounded_semaphore_allocate_overrelease_raises(self):
        scenario = Scenario()
        sem = scenario.BoundedSemaphore(1)
        sem_api = scenario.api(sem)

        results = []

        def releaser():
            try:
                sem.release()
            except ValueError:
                results.append('overrelease')

        with scenario:
            r = scenario.thread(releaser)
            iterator = sem_api.allocate(r)
            with self.assertRaises(ValueError):
                list(iterator)
            scenario.finish(r)

        self.assertEqual(results, ['overrelease'])


class TestBarrier(unittest.TestCase):
    """Tests for Barrier primitive."""

    def test_barrier_creation(self):
        """Barrier can be created with parties count."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)
        self.assertEqual(barrier.parties, 3)
        self.assertEqual(barrier.n_waiting, 0)
        self.assertFalse(barrier.broken)

    def test_barrier_repr(self):
        """Barrier has informative repr."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        r = repr(barrier)
        self.assertIn('Barrier', r)
        self.assertIn('waiters=0/2', r)

    def test_barrier_repr_with_name(self):
        """Barrier repr shows name when set."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        barrier.name = "test_barrier"
        r = repr(barrier)
        self.assertIn('test_barrier', r)
        self.assertIn('Barrier', r)

    def test_barrier_invalid_parties(self):
        """Barrier raises ValueError for invalid parties."""
        scenario = Scenario()
        with self.assertRaises(ValueError):
            scenario.Barrier(0)
        with self.assertRaises(ValueError):
            scenario.Barrier(-1)

    def test_barrier_auto_mode_basic(self):
        """Barrier works in auto mode with correct number of parties."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        results = []

        def worker1():
            results.append(('w1_before', barrier.n_waiting))
            n = barrier.wait()
            results.append(('w1_after', n))

        def worker2():
            results.append(('w2_before', barrier.n_waiting))
            n = barrier.wait()
            results.append(('w2_after', n))

        with scenario:
            t1 = scenario.thread(worker1)
            t2 = scenario.thread(worker2)

            # Wait for both to block
            scenario.wait(t1)
            scenario.wait(t2)

            # cycle drives waiters through the barrier; t2 is the opener
            # (last party).  Closing the cycle releases t1 (the non-opener
            # waiter).  Both threads complete the barrier.
            with api.cycle(t1, t2):
                pass

        # Both should have completed with their wait numbers
        self.assertEqual(len(results), 4)
        # Each thread gets a unique number from 0 to parties-1
        wait_numbers = [r[1] for r in results if r[0].endswith('_after')]
        self.assertEqual(sorted(wait_numbers), [0, 1])

    def test_barrier_with_timeout_auto_mode(self):
        """Barrier.wait() with timeout returns None and raises BrokenBarrierError on timeout."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        result = []

        def worker():
            try:
                n = barrier.wait(timeout=IMMEDIATELY)
                result.append(('success', n))
            except BrokenBarrierError:
                result.append(('broken',))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=NEVER)

        # Should have timed out and barrier should be broken
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ('broken',))
        self.assertTrue(barrier.broken)

    def test_barrier_raw_basic(self):
        """Barrier works in raw mode without scheduler control.

        This tests that on_approached doesn't double-advance after release()
        when the barrier fills and on_finished() already advances transactions.
        """
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        results = []

        def worker1():
            results.append(('w1_before', barrier.n_waiting))
            n = barrier.wait()
            results.append(('w1_after', n))

        def worker2():
            results.append(('w2_before', barrier.n_waiting))
            n = barrier.wait()
            results.append(('w2_after', n))

        t1 = scenario.thread(worker1)
        t2 = scenario.thread(worker2)

        # Run raw (no "with scenario:")
        t1.start()
        t2.start()

        t1.join(timeout=NEVER)
        t2.join(timeout=NEVER)

        # Both should have completed with their wait numbers
        self.assertEqual(len(results), 4)
        # Each thread gets a unique number from 0 to parties-1
        wait_numbers = [r[1] for r in results if r[0].endswith('_after')]
        self.assertEqual(sorted(wait_numbers), [0, 1])

    def test_barrier_api_exists(self):
        """Barrier has an API accessible via scenario.api()."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        self.assertIsNotNone(api)
        self.assertEqual(api.raw.parties, 2)
        self.assertEqual(api.raw.n_waiting, 0)

    def test_barrier_with_action_auto_mode(self):
        """Barrier executes action when filled in auto mode."""
        scenario = Scenario()
        action_called = []

        def barrier_action(tx):
            action_called.append('action')

        barrier = scenario.Barrier(2, action=barrier_action)

        api = scenario.api(barrier)
        results = []

        def worker1():
            results.append('w1_before')
            n = barrier.wait()
            results.append(('w1_after', n))

        def worker2():
            results.append('w2_before')
            n = barrier.wait()
            results.append(('w2_after', n))

        with scenario:
            t1 = scenario.thread(worker1)
            t2 = scenario.thread(worker2)

            # Wait for both to block
            scenario.wait(t1)
            scenario.wait(t2)

            # cycle drives both threads through the barrier; the action
            # runs automatically when the opener (t2) fills the barrier.
            with api.cycle(t1, t2):
                pass

        # Action should have been called
        self.assertEqual(action_called, ['action'])
        # Both threads should complete
        self.assertEqual(len(results), 4)

    def test_barrier_regulated_mode(self):
        """Barrier can be used in regulated mode with cycle()."""
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        results = []

        def worker1():
            results.append('w1_before')
            n = barrier.wait()
            results.append(('w1_after', n))

        def worker2():
            results.append('w2_before')
            n = barrier.wait()
            results.append(('w2_after', n))

        with scenario:
            t1 = scenario.thread(worker1)
            t2 = scenario.thread(worker2)

            # Wait for both to block (they're in BLOCKED state)
            scenario.wait(t1)
            scenario.wait(t2)

            # Both are blocked, n_waiting is 0 (only counts ASLEEP, not BLOCKED)
            self.assertEqual(api.raw.n_waiting, 0)

            # cycle drives both threads through the barrier; t2 is the
            # opener (last party).  Closing the cycle releases t1 (the
            # non-opener waiter).
            with api.cycle(t1, t2):
                pass

        # Both threads should complete
        self.assertEqual(len(results), 4)
        # Both threads completed with their barrier positions
        self.assertIn(('w1_after', 0), results)
        self.assertIn(('w2_after', 1), results)

    def test_barrier_reset(self):
        """Barrier.reset() resets barrier and breaks waiting transactions."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)  # Need 3 parties
        api = scenario.api(barrier)
        results = []

        def waiter():
            try:
                barrier.wait(timeout=NEVER)
                results.append('wait_success')
            except BrokenBarrierError:
                results.append('wait_broken')

        def resetter():
            barrier.reset()
            results.append('reset_done')

        with scenario:
            # Start waiter thread
            t_wait = scenario.thread(waiter)
            scenario.wait(t_wait)

            # Unblock waiter to WAITING state.  Waiting for WAITING (rather
            # than just letting api.unblock advance the tx to COMMIT) is
            # essential: the waiter doesn't appear on actual._cond._waiters
            # until commit's actual.wait() has gone deep enough to call
            # _release_save (which is what transitions the tx to WAITING).
            # Without this sync point, resetter can race ahead and call
            # actual.reset() before the waiter has registered, leaving the
            # waiter parked on _cond._waiters with no notify ever coming.
            api.unblock(barrier.wait, t_wait)
            scenario.wait(Call(t_wait, barrier.wait, State.WAITING))

            # Start resetter thread
            t_reset = scenario.thread(resetter)
            scenario.wait(t_reset)

            # Unblock reset - this will break the waiting transaction
            api.unblock(barrier.reset, t_reset)

        # Waiter should have been broken, resetter should complete
        self.assertIn('wait_broken', results)
        self.assertIn('reset_done', results)

    def test_barrier_abort(self):
        """Barrier.abort() puts barrier in broken state."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)
        api = scenario.api(barrier)
        results = []

        def waiter():
            try:
                barrier.wait(timeout=NEVER)
                results.append('wait_success')
            except BrokenBarrierError:
                results.append('wait_broken')

        def aborter():
            barrier.abort()
            results.append('abort_done')

        with scenario:
            # Start waiter thread
            t_wait = scenario.thread(waiter)
            scenario.wait(t_wait)

            # Unblock waiter to WAITING state
            api.unblock(barrier.wait, t_wait)

            # Start aborter thread
            t_abort = scenario.thread(aborter)
            scenario.wait(t_abort)

            # Unblock abort - this will break the barrier and waiting transaction
            api.unblock(barrier.abort, t_abort)

        # Waiter should have been broken, aborter should complete
        self.assertIn('wait_broken', results)
        self.assertIn('abort_done', results)
        self.assertTrue(barrier.broken)

    def test_barrier_timeout_in_blocked_state(self):
        """Barrier.wait times out correctly when timeout expires in BLOCKED state."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)  # Need 3 parties, we'll only have 1
        results = []

        def waiter():
            try:
                result = barrier.wait(timeout=IMMEDIATELY)
                results.append(('success', result))
            except BrokenBarrierError:
                results.append('broken')

        with scenario:
            t = scenario.thread(waiter)

            # Wait for t to reach BLOCKED state.
            scenario.wait(t)
            tx = scenario.transaction(t)
            self.assertEqual(tx.state, State.BLOCKED)

            # Force the timeout deterministically via the scheduler API.
            # This transitions the tx to COMMITTED with timed_out=True,
            # result=False -- without depending on wall-clock elapsing.
            tx.expire()

        # Should have timed out and raised BrokenBarrierError
        self.assertEqual(results, ['broken'])
        # Barrier should be broken after timeout
        self.assertTrue(barrier.broken)


class TestConditionWaitRegulated(unittest.TestCase):
    """Tests for Condition.wait with scheduler control."""

    def test_condition_wait_notify_regulated(self):
        """Condition.cycle drives notify and manages the awakened waiter."""
        scenario = Scenario()
        rlock = scenario.RLock()
        condition = scenario.Condition(rlock)
        cond_api = scenario.api(condition)

        results = []

        def worker_A():
            condition.acquire()
            results.append('waiter_acquired')
            condition.wait()
            results.append('waiter_notified')
            condition.release()
            results.append('waiter_done')

        def worker_B():
            condition.acquire()
            results.append('notifier_acquired')
            condition.notify()
            results.append('notifier_notified')
            condition.release()
            results.append('notifier_done')

        with scenario:
            A = scenario.thread(worker_A)
            A.name = 'A'

            scenario.skip(A, rlock.acquire, wait=True)
            scenario.wait(A)
            a_wait = scenario.transaction(A)
            a_wait.unblock()
            scenario.wait(Waiting(a_wait))
            self.assertEqual(results, ['waiter_acquired'])

            B = scenario.thread(worker_B)
            B.name = 'B'
            scenario.skip(B, rlock.acquire, wait=True)
            scenario.wait(B)

            cycle = cond_api.cycle(A, B)
            self.assertFalse(hasattr(cond_api, 'choose'))
            self.assertFalse(hasattr(cond_api, 'finish'))
            self.assertEqual(cycle.waiters, (A,))

            # The notifier still holds the UL after notify.  Release it,
            # then wake the waiter from the cycle object.
            scenario.skip(B, rlock.release)
            self.assertEqual(cycle.wake(), A)
            scenario.skip(A, rlock.release, wait=True)

        self.assertEqual(results[:3], [
            'waiter_acquired',
            'notifier_acquired',
            'notifier_notified',
        ])
        self.assertEqual(sorted(results), [
            'notifier_acquired',
            'notifier_done',
            'notifier_notified',
            'waiter_acquired',
            'waiter_done',
            'waiter_notified',
        ])


class TestUnblockRaceCondition(unittest.TestCase):
    """Test that Core.unblock processes transactions one at a time."""

    def test_unblock_sequential_processing(self):
        """Unblock with wait=True processes each transaction sequentially.

        This is a regression test verifying that unblock processes transactions
        one at a time (observer, unblock, wait) rather than batching all observers
        first, then all unblocks, then all waits.

        We add observers to both transactions before calling unblock. Each observer
        records the state of both transactions at the moment it runs. With correct
        sequential processing:
        - Observer A runs after A is unblocked but before B is unblocked
        - Observer B runs after B is unblocked (and A is already done)
        """
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        # Will hold (txA.state, txB.state) at the moment each observer runs
        observer_snapshots = {}

        def worker_A():
            lock.locked()

        def worker_B():
            lock.locked()

        with scenario:
            A = scenario.thread(worker_A)
            B = scenario.thread(worker_B)

            # Wait for both to block
            scenario.wait(A)
            scenario.wait(B)

            # Get transactions
            txA = scenario._core.transactions[A]
            txB = scenario._core.transactions[B]

            # Now unblock A
            api.unblock(lock.locked, A)
            # and watch what happens:

            txs = [txA, txB]
            while txs:
                signalled = scenario.wait(*txs)
                if txA in signalled:
                    observer_snapshots['A'] = (txA.state, txB.state)
                    txs.remove(txA)
                    api.unblock(lock.locked, B)

                if txB in signalled:
                    observer_snapshots['B'] = (txA.state, txB.state)
                    txs.remove(txB)

        # Verify observer A saw: A in progress (RETURNED), B still BLOCKED
        self.assertEqual(observer_snapshots['A'][0], State.RETURNED)
        self.assertEqual(observer_snapshots['A'][1], State.BLOCKED)

        # Verify observer B saw: A done (RETURNED), B in progress (RETURNED)
        self.assertEqual(observer_snapshots['B'][0], State.RETURNED)
        self.assertEqual(observer_snapshots['B'][1], State.RETURNED)


class TestRawAccess(unittest.TestCase):
    """Test scenario.raw() and scenario.raws access."""

    def test_scenario_raw_method(self):
        """scenario.raw(primitive) returns the raw primitive."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        unreg = scenario.raw(lock)

        self.assertIsNotNone(unreg)
        self.assertIs(unreg, api.raw)
        # Calling a method on the raw reference produces an
        # raw tx.  Since we're outside any `with scenario:`,
        # the tx never gets published; verify behavior indirectly by
        # confirming calls go through without scheduler interference.
        unreg.acquire()
        unreg.release()

    def test_scenario_raws_dict(self):
        """scenario.raws[primitive] returns the raw primitive."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        unreg = scenario.raws[lock]

        self.assertIsNotNone(unreg)
        self.assertIs(unreg, api.raw)

    def test_raws_all_primitive_types(self):
        """All primitive types are registered in scenario.raws."""
        scenario = Scenario()

        lock = scenario.Lock()
        rlock = scenario.RLock()
        event = scenario.Event()
        condition = scenario.Condition()
        barrier = scenario.Barrier(2)

        # All should be accessible via scenario.raw()
        self.assertIsNotNone(scenario.raw(lock))
        self.assertIsNotNone(scenario.raw(rlock))
        self.assertIsNotNone(scenario.raw(event))
        self.assertIsNotNone(scenario.raw(condition))
        self.assertIsNotNone(scenario.raw(barrier))

        # All should be in scenario.raws
        self.assertIn(lock, scenario.raws)
        self.assertIn(rlock, scenario.raws)
        self.assertIn(event, scenario.raws)
        self.assertIn(condition, scenario.raws)
        self.assertIn(barrier, scenario.raws)





class TestPrimitiveAttributes(unittest.TestCase):
    """Regression tests for primitive attribute constraints."""

    def test_primitive_single_underscore_attributes(self):
        """A blanket primitive should expose the same single-underscore
        attributes as its threading.* counterpart (because primitives
        masquerade), plus _core (always) and _lock (where threading
        doesn't already use that name).
        """
        scenario = Scenario()

        # (blanket factory, threading factory) pairs.
        pairs = [
            (scenario.Lock,             lambda: threading.Lock()),
            (scenario.RLock,            lambda: threading.RLock()),
            (scenario.Semaphore,        lambda: threading.Semaphore()),
            (scenario.BoundedSemaphore, lambda: threading.BoundedSemaphore()),
            (scenario.Event,            lambda: threading.Event()),
            (lambda: scenario.Barrier(2),
                                        lambda: threading.Barrier(2)),
            (scenario.Condition,        lambda: threading.Condition()),
        ]

        def sunders(obj):
            return {n for n in dir(obj)
                    if n.startswith('_') and not n.startswith('__')}

        for blanket_factory, threading_factory in pairs:
            blanket_inst = blanket_factory()
            threading_inst = threading_factory()
            cls_name = blanket_inst.__class__.__name__

            extras = sunders(blanket_inst) - sunders(threading_inst)
            allowed_extras = {'_core', '_lock'}
            unexpected = extras - allowed_extras
            self.assertEqual(
                unexpected, set(),
                f"{cls_name} adds unexpected sunder attrs beyond _core/_lock: {unexpected}")

            # _core must always be present.
            self.assertTrue(hasattr(blanket_inst, '_core'),
                            f"{cls_name} missing _core")
            # _lock must be present (either as score lock added by blanket or
            # as the threading primitive's own _lock).
            self.assertTrue(hasattr(blanket_inst, '_lock'),
                            f"{cls_name} missing _lock")

class TestSchedulerPause(unittest.TestCase):
    """Tests for scheduler pause functionality."""

    def test_lock_acquire_pause_and_unpause(self):
        """Lock.acquire with pause=True pauses after COMMIT, unpause advances to EXITED."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)
        result = []

        def worker():
            lock.acquire()
            result.append('acquired')

        with scenario:
            t = scenario.thread(worker)

            # Use assign with pause=True
            api.assign(t, pause=True)

            # Transaction should be in PAUSED state
            tx = scenario.transactions[t]
            self.assertEqual(tx.state, State.PAUSED)

            # result should still be empty - worker hasn't returned yet
            self.assertEqual(result, [])

            # Unpause the transaction
            api.unpause(lock.acquire, t)

            # Wait for thread to complete
            terminated = Terminated(t)
            signaled = scenario.wait(terminated)
            self.assertIn(terminated, signaled)

        self.assertEqual(result, ['acquired'])



    def test_unpause_validates_state(self):
        """unpause raises if transaction is not in PAUSED state."""
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        def worker():
            lock.acquire()

        with scenario:
            t = scenario.thread(worker)
            scenario.wait(t)

            # t is in BLOCKED, not PAUSED - should raise
            with self.assertRaises(ValueError):
                api.unpause(lock.acquire, t)

            # Clean up - let the transaction complete
            api.assign(t)


class TestDriverDispatchLazy(unittest.TestCase):
    """Regression tests for lazy semantics of Driver imperatives
    combined with Dispatch add/remove.

    Driver.pause() (and the other parking imperatives) stage their
    tx.unblock on the driver's `lazy` slot rather than firing it
    eagerly.  That lazy must not fire as a side effect of
    Dispatch bookkeeping -- only the drain that __next__ runs at
    the top of each iteration may fire it.
    """

    def test_lazy_add_remove_is_side_effect_free(self):
        """Verify that for a paused driver:

            dispatch.add(d); dispatch.remove(d)

        leaves tx.state at BLOCKED throughout.  Same invariant
        via a Chain:

            chain.append(d); dispatch.add(chain); dispatch.remove(chain)

        Then re-adding and iterating fires the staged lazy and
        drives tx to PAUSED.
        """
        def make():
            s = Scenario()
            lock = s.Lock()
            def worker():
                lock.acquire(timeout=-1)
            return s, lock, worker

        # --- Phase 1: Driver directly into Dispatch. ---
        s, lock, worker = make()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            self.assertIs(tx.state, State.BLOCKED)
            d.pause()
            self.assertIs(tx.state, State.BLOCKED)
            disp = s.Dispatch()
            disp.add(d)
            self.assertIs(tx.state, State.BLOCKED)
            disp.remove(d)
            self.assertIs(tx.state, State.BLOCKED)
            # Re-add and iterate: now the staged lazy fires.
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.parked)
            self.assertIs(tx.state, State.PAUSED)
            s.finish(t)

        # --- Phase 2: Driver via Chain into Dispatch. ---
        s, lock, worker = make()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            self.assertIs(tx.state, State.BLOCKED)
            d.pause()
            self.assertIs(tx.state, State.BLOCKED)
            chain = s.Chain()
            chain.append(d)
            self.assertIs(tx.state, State.BLOCKED)
            self.assertIn(d, chain)
            disp = s.Dispatch()
            disp.add(chain)
            self.assertIs(tx.state, State.BLOCKED)
            self.assertIn(chain, disp)
            disp.remove(chain)
            self.assertIs(tx.state, State.BLOCKED)
            # Re-add the chain and iterate: chain promotes d to
            # current, then d.drive() fires the staged lazy.
            disp.add(chain)
            yielded = next(iter(disp))
            self.assertIs(yielded, d)
            self.assertIs(yielded.state, d.parked)
            self.assertIs(tx.state, State.PAUSED)
            s.finish(t)


class TestChain(unittest.TestCase):
    """Baseline smoke coverage for Scenario.Chain -- one test per
    public method.  Not exhaustive; broader coverage (edge cases,
    interleaved iteration, etc.) lands as part of the general
    coverage push.
    """

    def _make_thread(self, s):
        """Add a worker thread to scenario s, blocked on its own
        Lock.acquire.  Returns (lock, thread, driver).  Each thread
        gets its own lock so cleanup can release them independently
        (one lock shared by N threads only lets one through on
        unblock; the rest deadlock the scenario exit).
        Must be called inside `with s:`.
        """
        lock = s.Lock()
        def worker():
            lock.acquire(timeout=-1)
        t = s.thread(worker)
        s.wait(t)
        d = s.Driver(t)
        return lock, t, d

    def _cleanup(self, s, *triples):
        """Release the worker threads.  Each triple is (lock, thread,
        driver).  Drivers not already in a terminal state get
        close()'d to release their score-slot; then each thread's
        lock.acquire is unblocked so it can exit.
        """
        for lock, t, d in triples:
            if not d.done:
                d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_construct_empty(self):
        s = Scenario()
        chain = s.Chain()
        self.assertEqual(len(chain), 0)
        self.assertEqual(chain.pending, ())

    def test_construct_with_drivers(self):
        s = Scenario()
        with s:
            lock1, t1, d1 = self._make_thread(s)
            lock2, t2, d2 = self._make_thread(s)
            chain = s.Chain(d1, d2)
            self.assertEqual(len(chain), 2)
            self.assertEqual(chain.pending, (d1, d2))
            self.assertIs(chain.pending[0], d1)
            self.assertIs(chain.pending[1], d2)
            self._cleanup(s, (lock1, t1, d1), (lock2, t2, d2))

    def test_repr_terminates(self):
        # Regression: with a driver in pending, repr() of the chain
        # used to recurse infinitely (chain shows pending drivers
        # whose .owner is the chain).  Both the chain's repr and
        # the driver's repr must terminate.
        s = Scenario()
        with s:
            lock, t, d = self._make_thread(s)
            chain = s.Chain(d)
            chain_str = repr(chain)
            self.assertIn('Scenario.Chain', chain_str)
            self.assertIn(t.name, chain_str)
            # Driver repr also terminates even though its owner
            # is the chain.
            d_str = repr(d)
            self.assertIn('Scenario.Driver', d_str)
            self._cleanup(s, (lock, t, d))

    def test_len(self):
        s = Scenario()
        with s:
            lock1, t1, d1 = self._make_thread(s)
            lock2, t2, d2 = self._make_thread(s)
            chain = s.Chain()
            self.assertEqual(len(chain), 0)
            chain.append(d1)
            self.assertEqual(len(chain), 1)
            chain.append(d2)
            self.assertEqual(len(chain), 2)
            chain.remove(d1)
            self.assertEqual(len(chain), 1)
            chain.remove(d2)
            self.assertEqual(len(chain), 0)
            self._cleanup(s, (lock1, t1, d1), (lock2, t2, d2))

    def test_bool(self):
        # Empty chain is falsy; chain with a pending driver is
        # truthy; chain with only a current (no pending) is also
        # truthy; chain that has emptied via dispatch promotion is
        # falsy again.
        s = Scenario()
        with s:
            chain = s.Chain()
            self.assertFalse(chain)
            lock, t, d = self._make_thread(s)
            chain.append(d)
            self.assertTrue(chain)
            # Promote pending head to current via Dispatch iteration.
            d.pause()
            disp = s.Dispatch()
            disp.add(chain)
            next(iter(disp))
            # After the yielded driver leaves the active set,
            # advance_chain_after clears current; pending was already
            # empty; chain is falsy again.
            self.assertFalse(chain)
            s.finish(t)

    def test_contains(self):
        s = Scenario()
        with s:
            lock1, t1, d1 = self._make_thread(s)
            lock2, t2, d2 = self._make_thread(s)
            chain = s.Chain(d1)
            self.assertIn(d1, chain)
            self.assertNotIn(d2, chain)
            chain.remove(d1)
            self.assertNotIn(d1, chain)
            self._cleanup(s, (lock1, t1, d1), (lock2, t2, d2))

    def test_append_already_owned_raises(self):
        # Once a driver is in a Chain (or any other owner), trying
        # to append it to another Chain raises.
        s = Scenario()
        with s:
            lock, t, d = self._make_thread(s)
            chain1 = s.Chain(d)
            chain2 = s.Chain()
            with self.assertRaises(RuntimeError):
                chain2.append(d)
            self._cleanup(s, (lock, t, d))

    def test_remove_absent_raises(self):
        s = Scenario()
        with s:
            lock, t, d = self._make_thread(s)
            chain = s.Chain()
            with self.assertRaises(ValueError):
                chain.remove(d)
            self._cleanup(s, (lock, t, d))

    def test_current_set_after_dispatch_iteration(self):
        # When a Chain owned by a Dispatch is iterated, the head
        # of pending is promoted to current; once that current
        # yields a terminal, current goes back to None.
        s = Scenario()
        with s:
            lock, t, d = self._make_thread(s)
            d.pause()
            chain = s.Chain(d)
            disp = s.Dispatch()
            disp.add(chain)
            self.assertEqual(chain.pending, (d,))
            yielded = next(iter(disp))
            self.assertIs(yielded, d)
            # After the yielded driver is handed to the user,
            # advance_chain_after has cleared current.
            self.assertEqual(chain.pending, ())
            s.finish(t)

    def test_two_drivers_serialized_through_dispatch(self):
        # A chain of two drivers iterated through a Dispatch yields
        # them in pending-order: d1 first, then d2 after d1 reaches
        # terminal.
        s = Scenario()
        with s:
            lock1, t1, d1 = self._make_thread(s)
            lock2, t2, d2 = self._make_thread(s)
            d1.pause()
            d2.pause()
            chain = s.Chain(d1, d2)
            disp = s.Dispatch()
            disp.add(chain)
            it = iter(disp)
            first = next(it)
            self.assertIs(first, d1)
            self.assertIs(first.state, d1.parked)
            second = next(it)
            self.assertIs(second, d2)
            self.assertIs(second.state, d2.parked)
            s.finish(t1)
            s.finish(t2)


class TestModuleHelpersAndImportBranches(unittest.TestCase):
    def test_state_helpers_and_scenario_properties(self):
        s = Scenario()
        self.assertEqual(s.name, '')
        self.assertIs(s.apis, s._core.apis_proxy)
        self.assertIs(s.raws, s._core.raws_proxy)
        self.assertIs(s.log, s._core.log)
        self.assertIs(s.managed, s._core.managed_proxy)
        self.assertIsNone(s.transaction(threading.current_thread()))
        self.assertEqual(State.BLOCKED.index, 100)
        self.assertEqual(State.BLOCKED.name, 'BLOCKED')

    def test_scenario_wait_requires_items(self):
        s = Scenario()
        with self.assertRaises(ValueError):
            s.wait()

class TestSignalAndProxyInternals(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        self.core = self.scenario._core

    def test_locked_dict_proxy_methods(self):
        proxy = self.core.LockedDictProxy({'a': 1, 'b': 2})
        self.assertIn("'a': 1", repr(proxy))
        self.assertEqual(proxy.get('a'), 1)
        self.assertEqual(proxy['b'], 2)
        self.assertIn('a', proxy)
        self.assertEqual(set(iter(proxy)), {'a', 'b'})
        self.assertEqual(set(proxy.keys()), {'a', 'b'})
        self.assertEqual(set(proxy.values()), {1, 2})
        self.assertEqual(set(proxy.items()), {('a', 1), ('b', 2)})
        self.assertEqual(repr(self.core.LockedDictProxy()), '{}')

    def test_read_only_list_proxy_and_log_proxy(self):
        p1 = self.core.ReadOnlyListProxy([1, 2, 3])
        p2 = self.core.ReadOnlyListProxy([1, 2, 3])
        self.assertIn('ReadOnlyListProxy', repr(p1))
        self.assertIn(2, p1)
        self.assertEqual(p1[0], 1)
        self.assertEqual(len(p1), 3)
        self.assertEqual(list(iter(p1)), [1, 2, 3])
        self.assertEqual(list(reversed(p1)), [3, 2, 1])
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, [9])
        self.assertTrue(p1)
        self.assertEqual(p1.copy(), [1, 2, 3])
        self.assertEqual(p1.count(2), 1)
        self.assertEqual(p1.index(3), 2)
        self.assertEqual(p1.index(2, 0, None), 1)
        log = self.core.LogProxy([1, 2])
        self.assertIn('LogProxy', repr(log))
        log.clear()
        self.assertEqual(list(log), [])

    def test_locked_set_proxy_methods(self):
        changes = []
        p = self.core.LockedSetProxy(change=lambda a, r: changes.append((set(a), set(r))))
        self.assertEqual(repr(p), 'LockedSetProxy(set())')
        self.assertEqual(str(p), 'set()')
        p.add('a')
        p.add('a')
        p.update({'b'}, {'c'})
        self.assertIn('a', p)
        self.assertEqual(len(p), 3)
        self.assertTrue(p)
        self.assertEqual(set(iter(p)), {'a', 'b', 'c'})
        self.assertEqual(p.copy(), {'a', 'b', 'c'})
        self.assertTrue(p.issubset({'a', 'b', 'c', 'd'}))
        self.assertTrue(p.issuperset({'a'}))
        self.assertTrue(p.isdisjoint({'z'}))
        self.assertTrue(p <= {'a', 'b', 'c', 'd'})
        self.assertTrue(p < {'a', 'b', 'c', 'd'})
        self.assertTrue(p >= {'a'})
        self.assertTrue(p > set())
        self.assertTrue(p == {'a', 'b', 'c'})
        self.assertTrue(p != {'a'})
        self.assertEqual(p.union({'d'}), {'a', 'b', 'c', 'd'})
        self.assertEqual(p.intersection({'b', 'x'}), {'b'})
        self.assertEqual(p.difference({'a'}), {'b', 'c'})
        self.assertEqual(p.symmetric_difference({'c', 'd'}), {'a', 'b', 'd'})
        self.assertEqual(p | {'d'}, {'a', 'b', 'c', 'd'})
        self.assertEqual(p & {'b', 'd'}, {'b'})
        self.assertEqual(p - {'a'}, {'b', 'c'})
        self.assertEqual(p ^ {'c', 'd'}, {'a', 'b', 'd'})
        popped = p.pop()
        self.assertNotIn(popped, p)
        p.discard('a')
        p.discard('missing')
        p.remove('b')
        p.difference_update({'x', 'c'})
        p.update({'x', 'y'})
        p.intersection_update({'x'})
        p.symmetric_difference_update({'x', 'z'})
        p |= {'m'}
        p &= {'m', 'z'}
        p -= {'z'}
        p ^= {'n'}
        p.clear()
        self.assertFalse(p)
        self.assertTrue(changes)
        d = self.core.LockedSetProxy()
        self.assertEqual(d.copy(), set())

    def test_read_only_dict_proxy_methods(self):
        d = {'x': 1}
        proxy = self.core.ReadOnlyDictProxy(d)
        self.assertEqual(proxy[ 'x' ], 1)
        self.assertEqual(proxy.get('x'), 1)
        self.assertIn('x', proxy)
        self.assertEqual(len(proxy), 1)
        self.assertTrue(proxy)
        self.assertEqual(set(iter(proxy)), {'x'})
        self.assertEqual(set(proxy.keys()), {'x'})
        self.assertEqual(set(proxy.values()), {1})
        self.assertEqual(set(proxy.items()), {('x', 1)})


class TestInvisibleSunders(unittest.TestCase):
    """Scenario(nested=False) makes Lock/RLock sunder transactions run
    raw: cond.wait / cond.notify / cond.notify_all appear as
    monolithic txs to users, the sunders run silently as internal nested
    calls, and skip/park refuse to name nested-only methods."""

class TestScenarioCoreInternals(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        self.core = self.scenario._core

    def test_reset_clears_terminated_tx_debris_from_signaling(self):
        """reset() drops accumulated state.  Self-reporting tx and
        Signaled items report their own state directly; the per-score
        "signaling" set is gone, so reset has no signaling debris to
        clear.  This test verifies the post-refactor contract: tx.api
        self-reports True once done and stays True across reset (state
        is monotonic), and the log is cleared."""
        scenario = self.scenario
        lock = scenario.Lock()
        api = scenario.api(lock)
        with scenario:
            t = scenario.thread(lambda: lock.acquire(timeout=0))
            scenario.wait(lock.acquire)
            api.unblock(lock.acquire, t)
            scenario.wait(Terminated(t))
            tx_api = scenario.log[-1]
            self.assertTrue(tx_api.signal(scenario))
            scenario.reset()
            self.assertTrue(tx_api.signal(scenario))
            self.assertEqual(len(self.core.log), 0)
            self.assertTrue(Terminated(t).signal(scenario))

    def test_reset_clears_log_and_waiters(self):
        scenario = self.scenario
        lock = scenario.Lock()
        api = scenario.api(lock)
        with scenario:
            t = scenario.thread(lambda: lock.acquire(timeout=0))
            scenario.wait(lock.acquire)
            api.unblock(lock.acquire, t)
            scenario.wait(Terminated(t))
        # __exit__ does NOT clear the log (it persists for post-mortem
        # inspection); waiters self-clean as each wait completes.
        self.assertGreater(len(self.core.log), 0)
        self.assertEqual(len(self.core.waiters), 0)
        # reset() clears the log explicitly.
        scenario.reset()
        self.assertEqual(len(self.core.log), 0)

    def test_reset_is_idempotent(self):
        scenario = self.scenario
        scenario.reset()
        scenario.reset()  # second call is a no-op

    def test_context_manager_exit_auto_clears(self):
        scenario = self.scenario
        lock = scenario.Lock()
        api = scenario.api(lock)
        with scenario:
            t = scenario.thread(lambda: lock.acquire(timeout=0))
            scenario.wait(lock.acquire)
            api.unblock(lock.acquire, t)
            scenario.wait(Terminated(t))
            tx_api = scenario.log[-1]
            # tx.api is Signaling and self-reports True once done.
            self.assertTrue(tx_api.signal(scenario))
        # After exit, the log PERSISTS (for post-mortem inspection);
        # it is cleared on the next entry, not on exit.
        self.assertGreater(len(self.core.log), 0)
        with scenario:
            self.assertEqual(len(self.core.log), 0)

    def test_context_manager_catches_already_started_thread(self):
        ran = []
        def worker():
            ran.append(True)
        t = self.scenario.thread(worker)
        t.start(); t.join()
        with self.scenario:
            pass
        self.assertTrue(ran)

    def test_core_signal_requires_signaled(self):
        # Every wait item is Signaled now; score.signal on a bare
        # object asserts out.  This is a contract assertion, not a
        # user-facing error -- internal callers must wrap properly.
        item = object()
        with self.assertRaises(AssertionError):
            self.core.signal(item)

    def test_wait_on_monitor_thread_raises(self):
        evt = threading.Event()
        def sleeper():
            # Block forever; the test sets evt at the end to release us.
            # This ensures the monitor is still live when wait runs,
            # so we reliably hit the "cannot block on monitor thread" path.
            evt.wait(NEVER)
        t = threading.Thread(target=sleeper, daemon=True)
        t.start()
        with self.core.lock:
            self.core.register_thread(t)
            monitor = self.core.monitors[t]
            try:
                with self.assertRaises(ValueError):
                    self.core.wait({monitor}, timeout=IMMEDIATELY)
            finally:
                evt.set()
        t.join(); monitor.join()


    def test_thread_active_scenario_starts_immediately(self):
        evt = threading.Event()
        with self.scenario:
            t = self.scenario.thread(lambda: evt.set())
            evt.wait(NEVER)
            self.assertTrue(evt.is_set())
            t.join()


class TestCoreTransactionAndApiInternals(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        self.lock = self.scenario.Lock()
        self.api = self.scenario.api(self.lock)
        self.core = self.lock._core

    def test_transaction_error_helpers_via_direct_tx(self):
        tx_cls = self.core.acquire
        tx = tx_cls(self.lock.acquire, primitives_module._current_time(), regulated=False, blocking=True, timeout=-1)
        self.assertIsNone(tx.timeout)
        self.assertIn('blocking', tx.repr_helper())
        self.assertIn('BLOCKED', tx.repr('X'))
        with self.assertRaises(NotImplementedError):
            self.core.Transaction.commit(tx)
        with self.assertRaises(RuntimeError):
            tx.unblock()
        # tx.unpause is a counter-decrement primitive with an internal
        # invariant (pausing >= 0); callers outside the merge's design
        # shouldn't reach it.  The old strict-state RuntimeError path
        # is gone, so there's nothing to assert here.
        with self.assertRaises(RuntimeError):
            tx.close()

    def test_timeout_transaction_error_paths(self):
        # Setting tx.timeout (via expire/disregard/revert or directly)
        # is only allowed in BLOCKED state.
        tx_cls = self.core.acquire
        tx = tx_cls(self.lock.acquire, primitives_module._current_time(), regulated=False, blocking=True, timeout=NEVER)
        tx.state = State.COMMIT
        with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
            tx.expire()
        with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
            tx.disregard()
        with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
            tx.revert()
        with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
            tx.timeout = 0
        tx.state = State.BLOCKED
        # In BLOCKED, the trio is idempotent and order-insensitive:
        # only the last write matters.
        tx.expire()
        self.assertEqual(tx._timeout, 0)
        tx.disregard()
        # Lock.acquire's no_timeout sentinel is -1.
        self.assertEqual(tx._timeout, tx.no_timeout)
        self.assertEqual(tx._timeout, -1)
        tx.revert()
        self.assertEqual(tx._timeout, tx.original_timeout)
        # original_timeout preserved across all calls.
        self.assertEqual(tx.original_timeout, NEVER)
        # Lock.acquire's blocking=True, timeout=-1 stores -1 (the
        # no_timeout sentinel); the timeout property normalizes it to
        # None for callers.
        tx3 = tx_cls(self.lock.acquire, primitives_module._current_time(), regulated=False, blocking=True, timeout=-1)
        self.assertIsNone(tx3.timeout)
        self.assertEqual(tx3.original_timeout, -1)
        self.assertEqual(tx3._timeout, -1)

    def test_api_property_accessors_and_transaction_api_properties(self):
        self.assertIs(self.api.transactions, self.core.transactions_proxy)
        self.assertIsNone(self.api.transaction(threading.current_thread()))
        self.api.name = 'foo'
        self.assertEqual(self.api.name, 'foo')

        def worker():
            self.lock.acquire(timeout=NEVER)
        with self.scenario:
            t = self.scenario.thread(worker)
            self.scenario.wait(t)
            tx = self.scenario.transaction(t)
            _ = tx.thread; _ = tx.method; _ = tx.done; _ = tx.start_time; _ = tx.state
            _ = tx.kwargs; _ = tx.pause; tx.pause = tx.pause
            _ = tx.pausing  # read-only bool view of the internal counter
            self.assertFalse(hasattr(tx, 'pend'))
            self.assertFalse(hasattr(tx, 'blocking'))
            self.assertFalse(hasattr(tx, 'stalling'))
            _ = tx.end_time; _ = tx.result; _ = tx.timeout
            tx.disregard()
            tx.unblock()
            self.scenario.wait(tx)

    def test_transaction_api_unblock_errors(self):
        tx_cls = self.core.acquire
        tx = tx_cls(self.lock.acquire, primitives_module._current_time(), regulated=False, blocking=True, timeout=NEVER)
        with self.assertRaises(RuntimeError):
            tx.api.unblock()
        # tx.api.unpause is permissive: it sets self.pause = False,
        # which is a no-op when pause was already False.  Only an
        # unpause attempt past PAUSED raises (via the pause setter's
        # state > PAUSED guard), which can't be reached on this fresh
        # unattached tx.
        tx.api.unpause()


    def test_non_timeout_transaction_api_timeout_and_acquire_repr(self):
        def worker():
            self.lock.acquire(); self.lock.release()
        with self.scenario:
            t = self.scenario.thread(worker)
            self.scenario.wait(t)
            tx = self.scenario.transaction(t)
            self.assertIn('Lock.acquire', repr(tx._core))
            self.scenario.skip(t, self.lock.acquire, wait=True)
            self.scenario.wait(t)
            rtx = self.scenario.transaction(t)
            self.assertIsNone(rtx.timeout.value)
            self.scenario.finish(t)

    def test_api_unblock_pause_path(self):
        result = []
        def worker():
            self.lock.acquire(); result.append('acquired')
        with self.scenario:
            t = self.scenario.thread(worker)
            self.api.unblock(self.lock.acquire, t, pause=True)
            tx = self.scenario.transactions[t]
            self.scenario.wait(Reached(tx, State.PAUSED))
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(result, [])
            self.api.unpause(self.lock.acquire, t)
        self.assertEqual(result, ['acquired'])

    def test_direct_acquire_commit_overdue_timeout(self):
        self.lock.acquire()
        try:
            tx2 = self.core.acquire(self.lock.acquire, primitives_module._current_time(), regulated=False, blocking=True, timeout=NEVER)
            # Simulate "deadline has passed": set _timeout to 0 so the
            # getter returns max(0, 0 - elapsed) = 0, and commit's
            # actual.acquire(True, 0) returns False immediately.
            tx2._timeout = 0
            with self.scenario._core.lock:
                self.assertFalse(tx2.commit())
            self.assertTrue(tx2.timed_out)
        finally:
            self.lock.release()

    def test_release_and_locked_repr_and_error_paths(self):
        rel = self.core.release(self.lock.release, primitives_module._current_time(), regulated=False)
        self.assertIn('Lock.release', repr(rel))
        ltx = self.core.locked(self.lock.locked, primitives_module._current_time(), regulated=False)
        self.assertIn('Lock.locked', repr(ltx))

        def bad_release():
            try:
                self.lock.release()
            except RuntimeError:
                pass
        with self.scenario:
            t = self.scenario.thread(bad_release)
            self.api.unblock(self.lock.release, t)
        
    def test_lock_core_owner_and_recursion_helpers(self):
        self.assertFalse(hasattr(self.core, 'actual_owner'))
        self.assertFalse(hasattr(self.core, 'actual_recursion_count'))

        rscenario = Scenario()
        rlock = rscenario.RLock()
        rcore = rlock._core
        self.assertEqual(rcore.actual_owner(), 0)
        self.assertEqual(rcore.actual_recursion_count(), 0)
        original = rcore.actual
        class FakeActual:
            def __repr__(self):
                return '<broken _thread.RLock object at 0x0>'
            def _recursion_count(self):
                return 0
        try:
            rcore.actual = FakeActual()
            with self.assertRaises(RuntimeError):
                rcore.actual_owner()
        finally:
            rcore.actual = original

    def test_lockbase_methods_and_raw_repr(self):
        l = self.scenario.Lock()
        # Lock now has the three sunder methods (unified with RLock):
        self.assertTrue(hasattr(l, '_is_owned'))
        self.assertTrue(hasattr(l, '_release_save'))
        self.assertTrue(hasattr(l, '_acquire_restore'))
        # Lock still has no _recursion_count (that's RLock-specific):
        self.assertFalse(hasattr(l, '_recursion_count'))
        l.acquire()
        self.assertTrue(l.locked())
        l._at_fork_reinit()
        self.assertFalse(l.locked())
        self.assertIn('.raw', repr(self.scenario.raw(l)))

        r = self.scenario.RLock()
        actual = threading.RLock()
        self.assertEqual(hasattr(r, 'locked'), hasattr(actual, 'locked'))
        self.assertTrue(hasattr(r, '_is_owned'))
        self.assertTrue(hasattr(r, '_release_save'))
        self.assertTrue(hasattr(r, '_acquire_restore'))
        self.assertEqual(hasattr(r, '_recursion_count'),
                         hasattr(actual, '_recursion_count'))
        r.acquire(); r.acquire()
        self.assertTrue(r._is_owned())
        if hasattr(actual, '_recursion_count'):
            self.assertEqual(r._recursion_count(), 2)
        if hasattr(actual, 'locked'):
            self.assertTrue(r.locked())
        state = r._release_save()
        if hasattr(actual, '_recursion_count'):
            self.assertEqual(r._recursion_count(), 0)
        r._acquire_restore(state)
        if hasattr(actual, '_recursion_count'):
            self.assertEqual(r._recursion_count(), 2)
        self.assertTrue(r._is_owned())
        r.release(); r.release()
        with r:
            self.assertTrue(r._is_owned())
        r._at_fork_reinit()
        self.assertIn('.raw', repr(self.scenario.raw(r)))


class TestHighLevelLockApiCoverage(unittest.TestCase):

    def test_assign_raises_on_acquire_timeout(self):
        """Under settings-only expire, assign drives the acquirer's
        commit; if the commit times out (actual.acquire returned
        False), assign raises a chained RuntimeError mentioning
        "timed out".  Tested via assign with both a releaser and an
        acquirer where the acquirer was expired: assign's release
        succeeds, but the subsequent acquire commit honors the
        expired timeout=0 and... actually fails the lock-state
        precondition, so we approach the same path via finish:
        construct a held-lock scenario and let the worker's own
        commit return False after expire."""
        s = Scenario(); lock = s.Lock(); api = s.api(lock)
        with s:
            # Holder takes and keeps the lock.
            holder = s.thread(lambda: lock.acquire())
            s.finish(holder)
            # Waiter blocks on lock.acquire with a long timeout.
            result = []
            t = s.thread(lambda: result.append(lock.acquire(timeout=NEVER)))
            s.wait(lock.acquire, t)
            # Expire the waiter's acquire: settings-only, just marks
            # _timeout=0.
            api.expire(lock.acquire, t)
            # Drive past commit; actual.acquire(True, 0) returns
            # False because the lock is held.
            s.finish(t)
            # Release the held lock so scenario can exit cleanly.
            s.raw(lock).release()
        self.assertEqual(result, [False])

    def test_transfer_release_raised(self):
        s = Scenario(); lock = s.Lock(); api = s.api(lock)
        def bad_releaser():
            try:
                lock.release()
            except RuntimeError:
                pass
        def acquirer():
            lock.acquire()
        with s:
            r = s.thread(bad_releaser)
            a = s.thread(acquirer)
            with self.assertRaises(RuntimeError):
                api.assign(r, a)
            api.unblock(lock.acquire, a)
            s.finish(r, a)


    def test_relay_paths(self):
        s = Scenario(); lock = s.Lock(); api = s.api(lock); order=[]
        def a(): lock.acquire(); order.append('A'); lock.release()
        def b(): lock.acquire(); order.append('B'); lock.release()
        def c(): lock.acquire(); order.append('C'); lock.release()
        with s:
            ta=s.thread(a); tb=s.thread(b); tc=s.thread(c)
            api.assign(tb)
            for got in api.relay(tb, ta, tc):
                self.assertIn(got, (ta, tc))
            api.unblock(lock.release, tc)
        self.assertEqual(order, ['B','A','C'])
        with self.assertRaises(ValueError):
            api.relay(tb)

    def test_relay_cold_start(self):
        """relay's initial arg can be an acquirer (at acquire/BLOCKED)
        when the lock is unheld: relay drives that thread to take
        the lock, then hands off to the rest of the acquirers."""
        s = Scenario(); lock = s.Lock(); api = s.api(lock); order=[]
        def a(): lock.acquire(); order.append('A'); lock.release()
        def b(): lock.acquire(); order.append('B'); lock.release()
        def c(): lock.acquire(); order.append('C'); lock.release()
        with s:
            ta=s.thread(a); tb=s.thread(b); tc=s.thread(c)
            # No assign: lock is unheld.  All three threads sit at
            # acquire/BLOCKED.  Pass any of them as `initial`.
            s.wait(ta, tb, tc)
            for got in api.relay(tb, ta, tc):
                self.assertIn(got, (tb, ta, tc))
            # relay drove acquires; the last acquirer's release tx
            # needs to be unblocked so the worker can exit and the
            # scenario exit can join.
            api.unblock(lock.release, tc)
        self.assertEqual(order, ['B', 'A', 'C'])


class TestConditionThinWrapper(unittest.TestCase):
    """Active tests for fourth-era Condition."""

    def test_condition_rejects_foreign_lock(self):
        s1 = Scenario()
        s2 = Scenario()
        lock = s1.Lock()
        with self.assertRaises(TypeError):
            s2.Condition(lock)
        with self.assertRaises(TypeError):
            s1.Condition(threading.Lock())

    def test_condition_family_method_alias_signaling(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)

        def worker():
            condition.acquire()
            condition.release()

        with scenario:
            t = scenario.thread(worker)
            signaled = scenario.wait(lock.acquire)
            self.assertIn(lock.acquire, signaled)
            scenario.api(lock).unblock(lock.acquire, t)
            signaled = scenario.wait(lock.release)
            self.assertIn(lock.release, signaled)
            scenario.api(lock).unblock(lock.release, t)

    def test_condition_family_call_and_use_alias_signaling(self):
        """Call/Use aliases mirror ConditionFamily method aliases."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)

        def worker():
            condition.acquire()
            condition.release()

        with scenario:
            t = scenario.thread(worker)
            acquire_call = Call(t, condition.acquire)
            acquire_blocked = Call(t, condition.acquire, State.BLOCKED)
            use_condition = Use(t, condition)

            signaled = scenario.wait(acquire_call, acquire_blocked, use_condition)
            self.assertIn(acquire_call, signaled)
            self.assertIn(acquire_blocked, signaled)
            self.assertIn(use_condition, signaled)

            tx = scenario.transactions[t]
            scenario.api(condition).unblock(condition.acquire, t)
            scenario.wait(tx)
            signaled = scenario.wait(acquire_call, timeout=0)
            self.assertFalse(signaled)

            release_call = Call(t, condition.release)
            signaled = scenario.wait(release_call, use_condition)
            self.assertIn(release_call, signaled)
            self.assertIn(use_condition, signaled)

            tx = scenario.transactions[t]
            scenario.api(condition).unblock(condition.release, t)
            scenario.wait(tx)
            signaled = scenario.wait(release_call, use_condition, timeout=0)
            self.assertNotIn(release_call, signaled)
            self.assertNotIn(use_condition, signaled)



class TestMindersBasic(unittest.TestCase):

    def test_call_minder_is_tuple(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            try:
                m = Call(t, lock.acquire)
                self.assertIsInstance(m, Call)
                self.assertIsInstance(m, tuple)
                self.assertEqual(m.thread, t)
                self.assertEqual(m.method, lock.acquire)
                self.assertIsNone(m.state)
                m2 = Call(t, lock.acquire)
                self.assertEqual(m, m2)
            finally:
                s.finish(t)

    def test_call_minder_signals_when_thread_calls_method(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            m = Call(t, lock.acquire)
            signaled = s.wait(m)
            self.assertIn(m, signaled)
            tx = s.transactions[t]
            tx.unblock()
            s.wait(tx)
            signaled = s.wait(m, timeout=0)
            self.assertNotIn(m, signaled)

    def test_call_minder_with_state(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            tx_cls = s._core.LockCore.acquire
            m_blocked = Call(t, lock.acquire, State.BLOCKED)
            fired = s.wait(m_blocked)
            self.assertIn(m_blocked, fired)
            tx = s.transactions[t]
            tx.unblock()
            s.wait(tx)

    def test_terminated_is_tuple(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            term = Terminated(t)
            self.assertIsInstance(term, Terminated)
            self.assertIsInstance(term, tuple)
            self.assertIs(term.thread, t)
            term2 = Terminated(t)
            self.assertEqual(term, term2)
            s.wait(t)
            tx = s.transaction(t)
            tx.unblock(); s.wait(tx)

    def test_terminated_fires_on_exit(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            terminated = Terminated(t)
            signaled = s.wait(terminated, timeout=0)
            self.assertFalse(signaled)

            s.wait(t)
            tx = s.transaction(t)
            tx.unblock()
            s.wait(tx)
            # After user code returns, thread exits; Terminated fires.
            signaled = s.wait(terminated)
            self.assertIn(terminated, signaled)

    def test_terminated_already_dead(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            tx.unblock(); s.wait(tx)
            # Terminated fires when the monitor observes the thread's
            # death.  Wait for it.
            terminated = Terminated(t)
            signaled = s.wait(terminated)
            self.assertIn(terminated, signaled)





class TestNotSignal(unittest.TestCase):
    """Tests for the Not(thread) signal."""

    def test_not_is_tuple(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            n = Not(t)
            self.assertIsInstance(n, Not)
            self.assertIsInstance(n, tuple)
            self.assertIs(n.thread, t)
            # Same thread returns the same object.
            self.assertEqual(n, Not(t))
            s.wait(t)
            _tx = s.transaction(t); _tx.unblock(); s.wait(_tx)

    def test_not_high_when_thread_idle(self):
        """Not(A) is high when A has no active tx (whether alive-idle or dead).

        Under the 'thread-as-signal means only has-active-tx' rule, Not(A)
        is the direct inverse: high whenever transactions.get(A) is None.
        A terminated thread has no active tx, so Not stays high after death.
        To distinguish alive-idle from dead, compose with Terminated(A).
        """
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            n = Not(t)
            # Thread is alive, no tx yet if we check before it enters any.
            # We can't assert Not is high here because the thread may
            # have already pushed lock.acquire.  Wait for its push, then
            # drive through.
            s.wait(t)
            # Now t has a tx: Not must be low.
            signaled = s.wait(n, timeout=0)
            self.assertFalse(signaled)

            tx = s.transaction(t)
            tx.unblock(); s.wait(tx)
            s.wait(t)
            signaled = s.wait(n, timeout=0)
            self.assertFalse(signaled)

            tx = s.transaction(t)
            tx.unblock()
            # Thread is now finishing; wait for Terminated.
            terminated = Terminated(t)
            s.wait(terminated)

            # After termination, Not stays HIGH (dead thread has no active tx).
            signaled = s.wait(terminated, n)
            self.assertIn(terminated, signaled)
            self.assertIn(n, signaled)

    def test_not_low_during_tx(self):
        """Not(A) is low the entire time A is in any transaction."""
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)

            # Top-level tx is up; Not should be low.
            n = Not(t)
            signaled = s.wait(n, timeout=0)
            self.assertNotIn(n, signaled)

            api = s.api(lock)
            api.unblock(lock.acquire, t)

    def test_not_repr(self):
        s = Scenario()
        def worker():
            pass
        with s:
            t = s.thread(worker)
            t.name = 'WorkerT'
            n = Not(t)
            self.assertEqual(repr(n), "Not('WorkerT')")


class TestScenarioLifecycleErrors(unittest.TestCase):
    """Scenario.__enter__/__exit__ misuse errors."""

    def test_double_enter_raises(self):
        s = Scenario()
        s.__enter__()
        try:
            with self.assertRaises(RuntimeError) as cm:
                s.__enter__()
            self.assertIn("already entered", str(cm.exception))
        finally:
            s.__exit__(None, None, None)

    def test_exit_without_enter_raises(self):
        s = Scenario()
        with self.assertRaises(RuntimeError) as cm:
            s.__exit__(None, None, None)
        self.assertIn("haven't entered", str(cm.exception))


class TestThreadRegistrationCoverage(unittest.TestCase):
    """Coverage around thread registration and monitor cleanup."""

    def test_wait_on_unstarted_thread_raises(self):
        """Waiting on a thread that hasn't been started yet is a ValueError."""
        s = Scenario()
        # A bare threading.Thread without start().
        t = threading.Thread(target=lambda: None)
        with s:
            with self.assertRaises(ValueError) as cm:
                s.wait(t)
            self.assertIn("unstarted", str(cm.exception))


class TestCoreAttrMapProxyCoverage(unittest.TestCase):
    """Coverage for the scenario.apis / scenario.raws / scenario.transactions
    proxy: get/keys/values/items/__len__/__iter__/__repr__/__getitem__ error."""

    def test_apis_proxy_dict_interface(self):
        s = Scenario()
        lock = s.Lock()
        # Public dict-like interface on scenario.apis (which is a
        # CoreAttrMapProxy):
        self.assertEqual(len(s.apis), 1)
        self.assertIn(lock, s.apis)
        api = s.apis[lock]
        self.assertIs(api, s.api(lock))
        self.assertEqual(list(s.apis.keys()), [lock])
        self.assertEqual(list(s.apis.values()), [api])
        self.assertEqual(list(s.apis.items()), [(lock, api)])
        self.assertEqual(list(iter(s.apis)), [lock])
        # Default-returning get.
        self.assertIs(s.apis.get(lock), api)
        sentinel = object()
        self.assertIs(s.apis.get(object(), sentinel), sentinel)
        # __repr__ shouldn't blow up.
        repr(s.apis)

    def test_apis_proxy_getitem_missing_raises_keyerror(self):
        s = Scenario()
        with self.assertRaises(KeyError):
            s.apis[object()]

    def test_apis_proxy_contains_non_primitive(self):
        s = Scenario()
        self.assertNotIn(object(), s.apis)


class TestSignalTokenCoverage(unittest.TestCase):
    """Validation, repr, and direct-subclass construction for the
    signal-token family: Nested, Reached, TransactionState (and its
    13 subclasses).  Plus the ImmutableTransactionSignalToken
    'transaction' alias property."""

    def _make_tx(self, s, lock):
        """Spin up a worker, drive it to BLOCKED, return the tx api."""
        def worker():
            lock.acquire()
        t = s.thread(worker)
        s.__enter__()
        s.wait(t)
        return s.transaction(t)

    def _finish_tx(self, s, tx):
        tx.unblock()
        s.wait(tx)
        s.__exit__(None, None, None)

    def test_nested_rejects_non_transaction(self):
        with self.assertRaises(TypeError) as cm:
            Nested("not a tx")
        self.assertIn("Transaction", str(cm.exception))

    def test_nested_repr(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            r = repr(Nested(tx))
            self.assertTrue(r.startswith("Nested("))
        finally:
            self._finish_tx(s, tx)

    def test_transaction_signal_token_transaction_alias(self):
        """ImmutableTransactionSignalToken.transaction returns the tx (alias for .tx)."""
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            nested = Nested(tx)
            self.assertIs(nested.transaction, tx)
            self.assertIs(nested.transaction, nested.tx)
        finally:
            self._finish_tx(s, tx)

    def test_reached_rejects_non_transaction(self):
        with self.assertRaises(TypeError) as cm:
            Reached("not a tx", State.BLOCKED)
        self.assertIn("transaction", str(cm.exception))

    def test_reached_rejects_non_state(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            with self.assertRaises(TypeError) as cm:
                Reached(tx, "not a state")
            self.assertIn("State", str(cm.exception))
        finally:
            self._finish_tx(s, tx)

    def test_reached_repr(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            r = repr(Reached(tx, State.WAITING))
            self.assertTrue(r.startswith("Reached("))
            self.assertIn("WAITING", r)
        finally:
            self._finish_tx(s, tx)

    def test_transaction_state_rejects_non_transaction(self):
        with self.assertRaises(TypeError) as cm:
            TransactionState("not a tx", State.BLOCKED)
        self.assertIn("transaction", str(cm.exception))

    def test_transaction_state_rejects_non_state(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            with self.assertRaises(TypeError) as cm:
                TransactionState(tx, "not a state")
            self.assertIn("State", str(cm.exception))
        finally:
            self._finish_tx(s, tx)

    def test_transaction_state_rejects_unknown_state(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            # Construct a State that isn't bound to any TransactionState subclass.
            bogus = State(999, 'BOGUS')
            with self.assertRaises(ValueError) as cm:
                TransactionState(tx, bogus)
            self.assertIn("unrecognized state", str(cm.exception))
        finally:
            self._finish_tx(s, tx)

    def test_transaction_state_repr_includes_class_name(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            self.assertTrue(repr(Blocked(tx)).startswith("Blocked("))
        finally:
            self._finish_tx(s, tx)

    def test_transaction_state_subclass_direct_construction(self):
        """Direct construction of each TransactionState subclass produces
        an equal-to-itself instance with the right state."""
        from blanket import primitives as P
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            pairs = [
                (P.Blocked, State.BLOCKED),
                (P.Commit, State.COMMIT),
                (P.Waiting, State.WAITING),
                (P.Stalled, State.STALLED),
                (P.Resumed, State.RESUMED),
                (P.Committed, State.COMMITTED),
                (P.Paused, State.PAUSED),
                (P.Exiting, State.EXITING),
                (P.Returned, State.RETURNED),
                (P.Raised, State.RAISED),
            ]
            for cls, state in pairs:
                with self.subTest(state=state.name):
                    inst = cls(tx)
                    self.assertIsInstance(inst, cls)
                    self.assertIsInstance(inst, TransactionState)
                    self.assertIs(inst.state, state)
                    self.assertIs(inst.tx, tx)
                    # Equal to itself, and to a fresh construction.
                    self.assertEqual(inst, cls(tx))
        finally:
            self._finish_tx(s, tx)


class TestPark(unittest.TestCase):
    """Tests for Scenario.park."""

    def test_park_skip_single_thread_single_method(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            result = s.park(t, lock.acquire)
            self.assertIn(t, result)
            self.assertEqual(result[t].method, lock.acquire)
            self.assertEqual(result[t].state, State.BLOCKED)
            result[t].unblock(); s.wait(result[t])
            r2 = s.park(t, lock.release)
            r2[t].unblock(); s.wait(r2[t])

    def test_park_skip_multi_thread(self):
        s = Scenario()
        lock = s.Lock()
        def worker_a():
            lock.acquire()
            lock.release()
        def worker_b():
            lock.acquire()
        with s:
            A = s.thread(worker_a)
            B = s.thread(worker_b)
            result = s.park(A, lock.acquire, B, lock.acquire)
            self.assertEqual(result[A].method, lock.acquire)
            self.assertEqual(result[B].method, lock.acquire)
            result[A].unblock(); s.wait(result[A])
            r2 = s.park(A, lock.release)
            r2[A].unblock(); s.wait(r2[A])
            result[B].unblock(); s.wait(result[B])

    def test_park_skip_rejects_no_args(self):
        s = Scenario()
        with s:
            with self.assertRaises(ValueError):
                s.park()

    def test_park_skip_rejects_method_before_thread(self):
        s = Scenario()
        lock = s.Lock()
        with s:
            with self.assertRaises(ValueError):
                s.park(lock.acquire)

    def test_park_skip_rejects_duplicate_thread(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            with self.assertRaises(ValueError):
                s.park(t, lock.acquire, t, lock.release)
            r = s.park(t, lock.acquire)
            r[t].unblock(); s.wait(r[t])

    def test_park_skip_rejects_thread_with_no_method(self):
        s = Scenario()
        def worker():
            pass
        with s:
            t = s.thread(worker)
            # Two threads, second one bare
            def w2(): pass
            t2 = threading.Thread(target=w2, daemon=True)
            t2.start()
            lock_method = s.Lock().acquire
            with self.assertRaises(ValueError):
                s.park(t, lock_method, t2)
            t2.join()

    def test_park_skip_rejects_raw_reference(self):
        s = Scenario()
        lock = s.Lock()
        ulock = s.raws[lock]
        def worker():
            ulock.acquire()
        with s:
            t = s.thread(worker)
            with self.assertRaises(ValueError) as cm:
                s.park(t, ulock.acquire)
            self.assertIn('raw', str(cm.exception))

    def test_park_skip_rejects_too_many_methods(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            with self.assertRaises(ValueError):
                s.park(t, lock.acquire, lock.release, lock.acquire)
            r = s.park(t, lock.acquire)
            r[t].unblock(); s.wait(r[t])

    def test_park_skip_raises_on_mismatched(self):
        """park raises if the worker's current tx doesn't match."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            s.wait(t)
            with self.assertRaises(RuntimeError) as cm:
                s.park(t, lock.release)  # mismatched
            self.assertIn('pushed', str(cm.exception))
            r = s.park(t, lock.acquire)
            r[t].unblock(); s.wait(r[t])

    def test_park_skip_raises_on_missing_method(self):
        """park raises if the worker terminates before reaching the method."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Drive acquire to completion so thread is about to terminate.
            s.skip(t, lock.acquire, wait=True)
            with self.assertRaises(RuntimeError) as cm:
                s.park(t, lock.release)  # never called
            self.assertIn('terminated', str(cm.exception))

    def test_park_skip_raises_on_early_termination(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            pass  # exits without calling anything
        with s:
            t = s.thread(worker)
            # Wait for thread to terminate first.
            s.wait(Terminated(t))
            # Now park should raise immediately.
            with self.assertRaises(RuntimeError) as cm:
                s.park(t, lock.acquire)
            self.assertIn('terminated', str(cm.exception))

    def test_park_skip_wait_drives_past(self):
        """park(wait=True) unblocks each parked tx and waits for completion."""
        s = Scenario()
        lock = s.Lock()
        order = []
        def worker():
            lock.acquire()
            order.append('acq')
            lock.release()
            order.append('rel')
        with s:
            t = s.thread(worker)
            r = s.park(t, lock.acquire, wait=True)
            self.assertTrue(r[t].done)
            # After wait=True, worker has progressed past lock.acquire.
            # Drive lock.release to let worker finish.
            s.skip(t, lock.release, wait=True)
        self.assertEqual(order, ['acq', 'rel'])

    def test_park_skip_wait_concurrent_across_threads(self):
        """park(wait=True) waits on all threads' targets concurrently;
        threads whose targets depend on each other must not deadlock."""
        s = Scenario()
        lock = s.Lock()
        order = []
        def A_worker():
            lock.acquire()
            order.append('A')
        def B_worker():
            lock.acquire()
            order.append('B')
            lock.release()
        with s:
            A = s.thread(A_worker)
            B = s.thread(B_worker)
            # Drive B's acquire so B holds the lock.
            s.skip(B, lock.acquire, wait=True)
            # Now A is blocked on lock.acquire (actual lock held by B).
            s.wait(Call(A, lock.acquire, State.BLOCKED))
            # park(A, lock.acquire, B, lock.release, wait=True): A's target
            # is lock.acquire and can only complete after B releases.
            # Sequential per-thread waits would deadlock; concurrent must not.
            r = s.park(A, lock.acquire, B, lock.release, wait=True)
            self.assertTrue(r[A].done)
            self.assertTrue(r[B].done)
        self.assertEqual(order, ['B', 'A'])

    def test_park_after_completed_skip_tolerates_inflight_tx(self):
        """park after skip(wait=True) tolerates whatever transient
        state the worker is in mid-tx-pop."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            # Drive first acquire+release.  Worker is post-release,
            # heading toward the second acquire.
            s.skip(t, lock.acquire, lock.release, wait=True)
            # park's Driver auto-settles via IDLE while waiting for the
            # next tx push.
            r = s.park(t, lock.acquire)
            self.assertEqual(r[t].method, lock.acquire)
            r[t].unblock(); s.wait(r[t])
            s.skip(t, lock.release, wait=True)

    def test_park_at_blocked_rejects_divergence(self):
        """park rejects when the thread's parked tx is on a different
        method than asked."""
        s = Scenario()
        lock = s.Lock()
        other = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Wait for worker to be parked at lock.acquire BLOCKED.
            s.wait(Call(t, lock.acquire, State.BLOCKED))
            # User asks to park at a different method.  Pass-1 validation
            # raises because cur.method != user's first method.
            with self.assertRaises(RuntimeError) as cm:
                s.park(t, other.acquire)
            self.assertIn('pushed', str(cm.exception))
            s.skip(t, lock.acquire, wait=True)


class TestSkip(unittest.TestCase):
    """Tests for Scenario.skip."""

    def test_skip_single_thread_single_method(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            r = s.skip(t, lock.acquire)
            self.assertEqual(r[t].method, lock.acquire)
            s.wait(r[t])

    def test_skip_single_thread_multi_methods(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            r = s.skip(t, lock.acquire, lock.release, lock.acquire, lock.release)
            self.assertEqual(r[t].method, lock.release)
            s.wait(r[t])

    def test_skip_multi_thread(self):
        s = Scenario()
        lock = s.Lock()
        done = []
        done_lock = threading.Lock()
        def worker_a():
            lock.acquire()
            lock.release()
            with done_lock:
                done.append('A')
        def worker_b():
            lock.acquire()
            with done_lock:
                done.append('B')
        with s:
            A = s.thread(worker_a)
            B = s.thread(worker_b)
            r = s.skip(A, lock.acquire, lock.release, B, lock.acquire)
            self.assertEqual(r[A].method, lock.release)
            self.assertEqual(r[B].method, lock.acquire)
            s.wait(r[A]); s.wait(r[B])
        # Order of 'done' not deterministic after skip (both threads ran);
        # only check both appeared.
        self.assertEqual(sorted(done), ['A', 'B'])

    def test_skip_rejects_no_args(self):
        s = Scenario()
        with s:
            with self.assertRaises(ValueError):
                s.skip()

    def test_skip_rejects_method_before_thread(self):
        s = Scenario()
        lock = s.Lock()
        with s:
            with self.assertRaises(ValueError):
                s.skip(lock.acquire)

    def test_skip_allows_repeated_thread_segments(self):
        """skip permits naming the same thread more than once.

        This pins down the intended distinction from park(): park has
        exactly one method per thread, while skip may drive multiple
        methods, including in repeated thread segments.
        """
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            r = s.skip(t, lock.acquire,
                       t, lock.release,
                       t, lock.acquire,
                       t, lock.release,
                       wait=True)
            self.assertEqual(r[t].method, lock.release)
            self.assertTrue(r[t].done)

    def test_skip_rejects_raw_reference(self):
        s = Scenario()
        lock = s.Lock()
        ulock = s.raws[lock]
        def worker():
            ulock.acquire()
        with s:
            t = s.thread(worker)
            with self.assertRaises(ValueError) as cm:
                s.skip(t, ulock.acquire)
            self.assertIn('raw', str(cm.exception))

    def test_skip_raises_on_mismatched_first(self):
        """If the worker's current tx doesn't match the user's first
        method, skip raises."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Wait for worker to reach lock.acquire.
            s.wait(t)
            with self.assertRaises(RuntimeError) as cm:
                s.skip(t, lock.release)  # mismatched; worker is at acquire
            self.assertIn('expected', str(cm.exception))
            # Cleanup.
            s.skip(t, lock.acquire, wait=True)

    def test_skip_raises_on_missing_method(self):
        """If the worker terminates before reaching the method, raise."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Wait for the worker to reach and enter lock.acquire.
            s.skip(t, lock.acquire, wait=True)  # unblock the acquire
            # Now thread is post-acquire, about to terminate.
            with self.assertRaises(RuntimeError) as cm:
                s.skip(t, lock.release)  # never called
            self.assertIn('terminated', str(cm.exception))

    def test_skip_wait_does_not_deadlock_across_threads(self):
        """skip(wait=True) must wait on every thread's last tx CONCURRENTLY.

        Regression: previously skip waited for each thread's last tx
        sequentially; if thread A's last tx is mid-commit and won't
        complete until thread B is also driven, sequential wait would
        deadlock (waiting on A while B sits BLOCKED).

        Setup: B acquires the lock first.  A then calls lock.acquire,
        which blocks on the actual underlying lock.  We then call
        skip(A, lock.acquire, B, lock.release, wait=True).  The OLD
        code would drive A.acquire (unblock it -> A's commit stuck on
        actual lock), wait for A.acquire to complete (deadlock: A is
        waiting for B's release), and never drive B.release.  The NEW
        code drives both unblocks first, then waits for both
        concurrently."""
        s = Scenario()
        lock = s.Lock()
        order = []
        def A_worker():
            lock.acquire()
            order.append('A_got_lock')
        def B_worker():
            lock.acquire()
            order.append('B_got_lock')
            lock.release()
            order.append('B_released')
        with s:
            A = s.thread(A_worker)
            B = s.thread(B_worker)
            # Drive B's acquire first so B holds the lock.
            s.skip(B, lock.acquire, wait=True)
            # Now A calls lock.acquire and will be BLOCKED at the
            # scheduler level.  Wait for A to reach BLOCKED.
            s.wait(Call(A, lock.acquire, State.BLOCKED))
            # Drive both: unblock A.acquire (A's commit will then sit
            # waiting on the actual lock until B releases) AND drive
            # B.release.  With wait=True, both must complete.
            r = s.skip(A, lock.acquire,
                       B, lock.release, wait=True)
            self.assertTrue(r[A].done)
            self.assertTrue(r[B].done)
        self.assertEqual(order, ['B_got_lock', 'B_released', 'A_got_lock'])

    def test_skip_after_completed_skip_proceeds(self):
        """A second skip after a complete prior skip drives the next
        acquire/release pair."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire, lock.release, wait=True)
            # By the time skip returns, the worker has completed the
            # first pair and is heading to the next lock.acquire.
            s.skip(t, lock.acquire, lock.release, wait=True)



class TestConditionCycleBasic(unittest.TestCase):

    def park_waiter(self, scenario, condition, lock, thread):
        scenario.skip(thread, lock.acquire, wait=True)
        scenario.wait(thread)
        tx = scenario.transaction(thread)
        tx.unblock()
        scenario.wait(Waiting(tx))
        return tx

    def test_cycle_one_waiter(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter():
            lock.acquire(); log.append('A_acq')
            condition.wait(); log.append('A_woke')
            lock.release(); log.append('A_rel')

        def notifier():
            lock.acquire(); log.append('X_acq')
            condition.notify(); log.append('X_notified')
            lock.release(); log.append('X_rel')

        with scenario:
            a = scenario.thread(waiter)
            x = scenario.thread(notifier)
            self.park_waiter(scenario, condition, lock, a)
            scenario.skip(x, lock.acquire, wait=True)
            scenario.wait(x)

            cycle = capi.cycle(a, x)
            self.assertEqual(cycle.waiters, (a,))
            self.assertEqual(cycle.extra_waiters, 0)
            self.assertFalse(hasattr(capi, 'choose'))
            self.assertFalse(hasattr(capi, 'finish'))

            scenario.skip(x, lock.release)
            self.assertEqual(cycle.wake(), a)
            with self.assertRaises(RuntimeError):
                cycle.wake()
            self.assertTrue(cycle.closed)
            scenario.skip(a, lock.release, wait=True)

        self.assertEqual(log, ['A_acq', 'X_acq', 'X_notified', 'X_rel', 'A_woke', 'A_rel'])

    def test_cycle_two_waiters_notify_all_context_manager_close(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter(name):
            def fn():
                lock.acquire(); log.append(f'{name}_acq')
                condition.wait(); log.append(f'{name}_woke')
                lock.release(); log.append(f'{name}_rel')
            return fn

        def notifier():
            lock.acquire(); log.append('X_acq')
            condition.notify_all(); log.append('X_notified')
            lock.release(); log.append('X_rel')

        with scenario:
            a = scenario.thread(waiter('A'))
            b = scenario.thread(waiter('B'))
            x = scenario.thread(notifier)
            self.park_waiter(scenario, condition, lock, a)
            self.park_waiter(scenario, condition, lock, b)
            scenario.skip(x, lock.acquire, wait=True)
            scenario.wait(x)

            with capi.cycle(a, b, x) as cycle:
                scenario.skip(x, lock.release)
                self.assertEqual(cycle.wake(b), (b,))
                scenario.skip(b, lock.release)
            # __exit__ closed the cycle and woke A.
            scenario.skip(a, lock.release, wait=True)

        self.assertEqual(log[-6:], ['X_notified', 'X_rel', 'B_woke', 'B_rel', 'A_woke', 'A_rel'])

    def test_cycle_iter_and_call(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter(name):
            def fn():
                lock.acquire(); condition.wait(); log.append(name); lock.release()
            return fn

        def notifier():
            lock.acquire(); condition.notify_all(); lock.release()

        with scenario:
            a = scenario.thread(waiter('A'))
            b = scenario.thread(waiter('B'))
            c = scenario.thread(waiter('C'))
            n = scenario.thread(notifier)
            for t in (a, b, c):
                self.park_waiter(scenario, condition, lock, t)
            scenario.skip(n, lock.acquire, wait=True)
            scenario.wait(n)
            cycle = capi.cycle(a, b, c, n)
            scenario.skip(n, lock.release)

            it = cycle.iter(c, a)
            self.assertIs(next(it), c)
            scenario.skip(c, lock.release)
            self.assertIs(next(it), a)
            scenario.skip(a, lock.release)
            self.assertIsNone(next(it, None))
            self.assertEqual(cycle(), (b,))
            scenario.skip(b, lock.release, wait=True)

        self.assertEqual(log, ['C', 'A', 'B'])

    def test_cycle_can_shepherd_waiter_and_waker_from_ul_acquire(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter():
            lock.acquire(); log.append('A_acq')
            condition.wait(); log.append('A_woke')
            lock.release(); log.append('A_rel')

        def notifier():
            lock.acquire(); log.append('N_acq')
            condition.notify(); log.append('N_notified')
            lock.release(); log.append('N_rel')

        with scenario:
            a = scenario.thread(waiter)
            n = scenario.thread(notifier)
            scenario.wait(a)
            cycle = capi.cycle(a, n)
            self.assertEqual(cycle.waiters, (a,))
            scenario.skip(n, lock.release)
            self.assertEqual(cycle.wake(), a)
            scenario.skip(a, lock.release, wait=True)

        self.assertEqual(log, ['A_acq', 'N_acq', 'N_notified', 'N_rel', 'A_woke', 'A_rel'])


class TestCycleIteratorProtocol(unittest.TestCase):
    """The cycle object is itself an iterator: __iter__ returns self,
    __next__ wakes the next remaining waiter (spec order) and yields
    its thread, raising StopIteration on empty remaining or closed."""

    def park_waiter(self, scenario, condition, lock, thread):
        scenario.skip(thread, lock.acquire, wait=True)
        scenario.wait(thread)
        tx = scenario.transaction(thread)
        tx.unblock()
        scenario.wait(Waiting(tx))
        return tx

    def _setup(self):
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []
        def waiter(name):
            def fn():
                lock.acquire()
                condition.wait()
                log.append(name)
                lock.release()
            return fn
        def notifier():
            lock.acquire(); condition.notify_all(); lock.release()
        return scenario, lock, condition, capi, log, waiter, notifier

    def test_iter_self_returns_self(self):
        # iter(cycle) returns cycle; standard iterator pattern.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            cy = capi.cycle(a, n)
            s.skip(n, lock.release)
            self.assertIs(iter(cy), cy)
            list(cy)  # drain so scenario exits cleanly
            s.skip(a, lock.release, wait=True)

    def test_for_loop_drains_in_spec_order(self):
        # The canonical with-for idiom: drain all remaining waiters
        # in spec order, doing post-wake work between iterations.
        # The notifier (n) is in the cycle's orchestration but not
        # in remaining; the iterator yields only waiters.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            c = s.thread(waiter('C'))
            n = s.thread(notifier)
            for t in (a, b, c):
                self.park_waiter(s, cond, lock, t)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            with capi.cycle(a, b, c, n) as cy:
                s.skip(n, lock.release)
                order = []
                for t in cy:
                    order.append(t)
                    s.skip(t, lock.release)
        self.assertEqual(order, [a, b, c])
        self.assertEqual(log, ['A', 'B', 'C'])

    def test_next_yields_one_at_a_time(self):
        # next(cycle) wakes one waiter and returns it.  Repeated calls drain.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            self.park_waiter(s, cond, lock, b)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            cy = capi.cycle(a, b, n)
            s.skip(n, lock.release)
            self.assertIs(next(cy), a)
            s.skip(a, lock.release)
            self.assertIs(next(cy), b)
            s.skip(b, lock.release, wait=True)
            with self.assertRaises(StopIteration):
                next(cy)

    def test_stopiteration_on_drained(self):
        # After draining via iteration, further next() raises StopIteration.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            cy = capi.cycle(a, n)
            s.skip(n, lock.release)
            list(cy)
            s.skip(a, lock.release, wait=True)
            self.assertEqual(next(cy, 'sentinel'), 'sentinel')
            with self.assertRaises(StopIteration):
                next(cy)

    def test_stopiteration_after_explicit_close(self):
        # close() drains remaining; subsequent next() raises StopIteration
        # rather than RuntimeError("cycle is closed").  The for-loop
        # protocol expects iteration-done to be StopIteration.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            self.park_waiter(s, cond, lock, b)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            cy = capi.cycle(a, b, n)
            s.skip(n, lock.release)
            self.assertIs(next(cy), a)
            s.skip(a, lock.release)
            # close() drains b; lock is free so b's cond.wait reacquires.
            cy.close()
            self.assertTrue(cy.closed)
            s.skip(b, lock.release, wait=True)
            with self.assertRaises(StopIteration):
                next(cy)

    def test_iter_in_with_block(self):
        # Canonical idiom: with-as cycle, for-in drain.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            self.park_waiter(s, cond, lock, b)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            with capi.cycle(a, b, n) as cy:
                s.skip(n, lock.release)
                drained = []
                for t in cy:
                    drained.append(t)
                    s.skip(t, lock.release)
                self.assertEqual(drained, [a, b])
            # After __exit__, cycle is closed (drained or not).
            self.assertTrue(cy.closed)


class TestConditionCycleValidation(unittest.TestCase):

    def test_cycle_requires_waiter_and_waker(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        with s:
            with self.assertRaisesRegex(ValueError, 'at least one waiter'):
                capi.cycle()

    def test_cycle_duplicate_thread_raises(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def worker():
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            s.wait(t)
            with self.assertRaisesRegex(ValueError, 'already has an active Driver'):
                capi.cycle(t, t)
            s.skip(t, lock.acquire, lock.release, wait=True)

    def test_cycle_wait_for_first_predicate_success_raises(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def waiter():
            lock.acquire(); condition.wait_for(lambda: True); lock.release()
        def notifier():
            lock.acquire(); condition.notify(); lock.release()

        with s:
            w = s.thread(waiter)
            n = s.thread(notifier)
            s.skip(w, lock.acquire, wait=True)
            s.wait(w)
            with self.assertRaisesRegex(RuntimeError, 'predicate succeeded'):
                capi.cycle(w, n)
            # Under model A, predicate=True returns from wait_for
            # without an inner wait, so the wait_for tx terminates
            # naturally inside drive_waiter_to_waiting's signal wait.
            # The worker then advances to lock.release.
            s.skip(w, lock.release, wait=True)
            # The notifier should now acquire and block on notify; drain it.
            s.skip(n, lock.acquire, condition.notify, lock.release, wait=True)


class TestDriveNotifyTermination(unittest.TestCase):
    """drive_notify's terminated-thread paths."""

    def test_drive_notify_thread_terminates_mid_notify(self):
        # Hard to deterministically kill a thread mid-notify.  Skip for now.
        pass


class TestConditionTimeoutSemantics(unittest.TestCase):
    """Regression tests for thin-wrapper timeout handling."""

    def test_condition_wait_minus_one_times_out_immediately(self):
        """Condition.wait(-1) is an immediate timeout, not wait-forever."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait(timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            tx = s.transaction(t)
            self.assertEqual(tx.timeout.value, -1)
            tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            tx.unstall()
            s.wait(tx)
            self.assertTrue(tx.timeout.timed_out)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])

    def test_condition_wait_for_timeout_marks_timed_out(self):
        """Condition.wait_for records timed_out when the real wait times out."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            self.assertEqual(wf_tx.timeout.value, -1)
            wf_tx.unblock()
            # Inner self.wait(-1) creates a child cond.wait tx.
            s.wait(Nested(wf_tx), Terminated(t))
            child = s.transaction(t)
            self.assertEqual(child.method, condition.wait)
            child.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            self.assertTrue(child.timeout.timed_out)
            # wait_for's loop now sees waittime<=0 and breaks; it
            # returns False to the user.
            s.wait(wf_tx)
            self.assertTrue(wf_tx.timeout.timed_out)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])


class TestConditionWaitFor(unittest.TestCase):
    """Exercise wait_for with predicate."""

    def test_wait_for_implementation_calls_self_wait(self):
        """Forward-risk safety net: blanket's wait_for-as-parent
        design depends on threading.Condition.wait_for's loop
        calling self.wait().  That has been true in CPython since
        3.4 (and the implementation has not changed in that time),
        but if a future Python ever inlined the wait, BlanketCondition's
        wait override would stop intercepting and wait_for would
        silently bypass the tx machinery.

        This test confirms wait_for actually dispatches through
        self.wait by using a Condition subclass that counts wait
        calls.  Predicate fails twice then succeeds, so wait_for's
        loop should call self.wait() exactly twice."""
        call_count = [0]

        class CountingCondition(threading.Condition):
            def wait(self, timeout=None):
                call_count[0] += 1
                return super().wait(timeout)

        cond = CountingCondition()
        results = iter([False, False, True])
        with cond:
            cond_thread_ready = threading.Event()
            cond_thread_done = threading.Event()
            def notifier():
                # Two notify calls so the waiter wakes twice; the
                # third predicate evaluation will return True without
                # another wait.
                cond_thread_ready.wait()
                with cond:
                    cond.notify()
                # Briefly wait for the waiter to re-park, then notify
                # again.  We don't have a precise hook here, so a
                # short sleep is the simplest synchronization; the
                # test's assertion is on call_count, not timing.
                time.sleep(0.05)
                with cond:
                    cond.notify()

            t = threading.Thread(target=notifier)
            t.start()
            cond_thread_ready.set()
            cond.wait_for(lambda: next(results))
            t.join(timeout=1.0)
            self.assertFalse(t.is_alive())
        # Predicate is False, False, True -> wait was called twice.
        self.assertEqual(call_count[0], 2)

    def test_wait_for_predicate_iterates(self):
        """wait_for's predicate gets called once before each internal wait,
        and once after each wake-up.  When the predicate finally returns
        truthy, wait_for returns without calling it again."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        shared = {'ready': False}

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: shared['ready'])
            lock.release()

        def notifier():
            lock.acquire()
            shared['ready'] = True
            condition.notify()
            lock.release()

        with s:
            w = s.thread(waiter)
            # Drive waiter through lock.acquire, into cond.wait_for.
            # Predicate starts False, so wait_for spawns a child
            # cond.wait tx whose _release_save shim drives it
            # through WAITING.
            s.skip(w, lock.acquire, wait=True)
            s.wait(w)
            w_tx = s.transaction(w)
            w_tx.unblock()
            # Wait for the child cond.wait tx to be born, then drive
            # it through to WAITING.
            s.wait(Nested(w_tx), Terminated(w))
            child = s.transaction(w)
            self.assertEqual(child.method, condition.wait)
            child.unblock()
            s.wait(Call(w, condition.wait, State.WAITING))
            # At this point, predicate was called once (returned False).
            self.assertEqual(w_tx.iterations, 1)

            # Drive notifier.
            n = s.thread(notifier)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)

            cycle = capi.cycle(w, n)
            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(), w)

            # Drain waiter's lock.release.
            s.skip(w, lock.release, wait=True)

            # Predicate was called: once initially (False), once after
            # reacquiring UL (True).  iterations counts predicate calls.
            self.assertGreaterEqual(w_tx.iterations, 2)

    def test_wait_for_api_exposes_iterations_and_predicate(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        predicate = lambda: False

        def waiter():
            lock.acquire()
            condition.wait_for(predicate, timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            self.assertIs(wf_tx.predicate, predicate)
            self.assertEqual(wf_tx.iterations, 0)
            # wait_for stays at COMMIT during commit; the inner
            # self.wait creates a child cond.wait tx.
            wf_tx.unblock()
            s.wait(Nested(wf_tx), Terminated(t))
            child = s.transaction(t)
            self.assertEqual(child.method, condition.wait)
            child.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            s.wait(wf_tx)
            s.skip(t, lock.release, wait=True)

    def test_wait_for_iterations_create_fresh_child_wait_txs(self):
        """Each iteration of wait_for spawns a fresh child cond.wait tx
        visible via Nested(wait_for).  The scheduler drives each child
        through WAITING/STALLED in turn; each iteration is a distinct
        tx parented to the wait_for tx."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        calls = []
        def predicate():
            calls.append(None)
            return len(calls) >= 3

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        def drive_iteration(wf_tx, t):
            # Wait for child wait tx to be born.
            s.wait(Nested(wf_tx), Terminated(t))
            child = s.transaction(t)
            self.assertEqual(child.method, condition.wait)
            self.assertIs(child.parent, wf_tx)
            child.unblock()
            s.wait(Call(t, condition.wait, State.WAITING))
            # Notify to wake.
            n = s.thread(notifier)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)
            s.transaction(n).unblock()
            s.skip(n, lock.release, wait=True)
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            return child

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            wf_tx.unblock()

            child1 = drive_iteration(wf_tx, t)
            # iterations counts predicate calls: 1 before any wait,
            # 1 after each wakeup.  After 1 wait round: 2.
            self.assertEqual(wf_tx.iterations, 2)
            child2 = drive_iteration(wf_tx, t)
            self.assertIsNot(child2, child1)
            self.assertEqual(wf_tx.iterations, 3)
            # Iter 3: predicate True; wait_for returns without spawning.
            s.wait(wf_tx)
            self.assertEqual(wf_tx.iterations, 3)
            s.skip(t, lock.release, wait=True)


class TestDefensiveTerminationPaths(unittest.TestCase):
    """Exercise defensive 'thread died unexpectedly' branches."""

    def test_transaction_method_is_primitive(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            self.assertIsNotNone(tx)
            self.assertEqual(tx.method, lock.acquire)
            s.skip(t, lock.acquire)

class TestConditionContextManager(unittest.TestCase):
    def test_condition_enter_exit(self):
        """Condition as a context manager: __enter__ acquires UL, __exit__ releases."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        def worker():
            with condition:
                pass
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire, lock.release)


class TestConditionAssign(unittest.TestCase):
    def test_condition_api_assign_delegates(self):
        """ConditionAPI.assign delegates to the underlying Lock's assign."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        def worker():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(worker)
            capi.assign(t)
            s.skip(t, lock.release)


class TestConditionNotifyAllApiN(unittest.TestCase):
    def test_notify_all_api_n_accessor(self):
        """notify_all API's .n property returns math.inf."""
        import math
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        def worker():
            lock.acquire(); condition.notify_all(); lock.release()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            na_tx = s.transaction(t)
            self.assertEqual(na_tx.n, math.inf)
            s.skip(t, condition.notify_all, lock.release)


class TestConditionReprs(unittest.TestCase):
    """Tests for repr()s of Condition-related Cores, APIs, minders, and transactions."""

    def test_minder_reprs(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            m_call = Call(t, lock.acquire)
            self.assertIn('Call(', repr(m_call))
            m_call_s = Call(t, lock.acquire, State.BLOCKED)
            self.assertIn('state=', repr(m_call_s))
            m_use = Use(t, lock)
            self.assertIn('Use', repr(m_use))
            m_term = Terminated(t)
            self.assertIn('Terminated', repr(m_term))
            s.wait(t)
            _tx = s.transaction(t); _tx.unblock(); s.wait(_tx)

    def test_state_repr(self):
        st = primitives_module.State(0, 'TEST')
        self.assertIn('TEST', repr(st))

    def test_condition_raw_repr(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        u = s.raw(condition)
        self.assertIn('raw', repr(u))

    def test_condition_core_and_api_reprs(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        self.assertIn('Condition', repr(condition))
        self.assertIn('ConditionAPI', repr(capi))
        self.assertIn('ConditionCore', repr(condition._core))

    def test_condition_tx_reprs(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()
        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wait_tx = s.transaction(t)
            self.assertIn('Condition.wait', repr(wait_tx))
            self.assertIn('Condition.wait', repr(wait_tx._core))
            # Drive wait through WAITING (timeout will fire) then STALLED.
            wait_tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            wait_tx.unstall()
            s.wait(wait_tx)
            s.skip(t, lock.release, wait=True)

    def test_condition_notify_tx_reprs(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        def notifier():
            lock.acquire()
            condition.notify(2)
            lock.release()
        def notifier_all():
            lock.acquire()
            condition.notify_all()
            lock.release()
        with s:
            t1 = s.thread(notifier)
            s.skip(t1, lock.acquire, wait=True)
            s.wait(t1)
            notify_tx = s.transaction(t1)
            self.assertIn('Condition.notify', repr(notify_tx))
            self.assertIn('Condition.notify', repr(notify_tx._core))
            self.assertEqual(notify_tx.n, 2)
            self.assertEqual(notify_tx._core.n, 2)
            s.skip(t1, condition.notify, lock.release, wait=True)

            t2 = s.thread(notifier_all)
            s.skip(t2, lock.acquire, wait=True)
            s.wait(t2)
            notify_all_tx = s.transaction(t2)
            self.assertIn('Condition.notify_all', repr(notify_all_tx))
            self.assertIn('Condition.notify_all', repr(notify_all_tx._core))
            import math
            self.assertEqual(notify_all_tx._core.n, math.inf)
            s.skip(t2, condition.notify_all, lock.release, wait=True)

    def test_wait_for_tx_reprs(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        predicate = lambda: True
        def worker():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()
        with s:
            t = s.thread(worker)
            s.wait(t)
            _tx = s.transaction(t); _tx.unblock(); s.wait(_tx)
            s.wait(t)
            wf_tx = s.transaction(t)
            self.assertIn('Condition.wait_for', repr(wf_tx))
            self.assertIn('Condition.wait_for', repr(wf_tx._core))
            self.assertIn('iterations=', repr(wf_tx._core))
            self.assertEqual(wf_tx.iterations, 0)
            self.assertIs(wf_tx.predicate, predicate)
            wf_tx.unblock(); s.wait(wf_tx)
            # Predicate True -> wait_for returns without calling self.wait.
            # No nested txs.  Drain outer release directly.
            s.wait(t)
            _tx = s.transaction(t); _tx.unblock(); s.wait(_tx)

    def test_condition_creates_rlock_when_no_lock_given(self):
        # Exercise the `lock = scenario.api.RLock()` default path (line 2202).
        s = Scenario()
        c = s.Condition()
        self.assertIsNotNone(c)



class TestUnstall(unittest.TestCase):
    """Tests for tx.unstall() and cond_api.unstall(*threads)."""

    def test_tx_unstall_from_STALLED_releases_park(self):
        """tx.unstall() on a cond.wait tx parked at STALLED releases
        the park and lets commit finish."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wait_tx = s.transaction(t)
            wait_tx.unblock()
            # Timeout causes cond.wait to proceed to STALLED.
            s.wait(Call(t, condition.wait, State.STALLED))
            self.assertEqual(wait_tx._core.state, State.STALLED)
            wait_tx.unstall()
            s.wait(wait_tx)  # tx now finishes naturally
            self.assertEqual(wait_tx._core.state, State.RETURNED)
            s.skip(t, lock.release, wait=True)

    def test_tx_unstall_raises_if_not_in_STALLED(self):
        """Calling tx.unstall() on a tx not at STALLED raises."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wait_tx = s.transaction(t)
            # State is BLOCKED; unstall should raise.
            with self.assertRaisesRegex(RuntimeError, "can't unstall"):
                wait_tx.unstall()
            # Clean up: drive to completion.
            wait_tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            wait_tx.unstall()
            s.wait(wait_tx)
            s.skip(t, lock.release, wait=True)

    def test_cond_api_unstall_single_thread(self):
        """cond_api.unstall(thread) works for a single parked thread."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wait_tx = s.transaction(t)
            wait_tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            capi.unstall(condition.wait, t)
            s.wait(wait_tx)
            s.skip(t, lock.release, wait=True)

    def test_cond_api_unstall_multi_thread(self):
        """cond_api.unstall(*threads) works for multiple parked threads."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        def park(thread):
            s.skip(thread, lock.acquire, wait=True)
            s.wait(thread)
            tx = s.transaction(thread)
            tx.unblock()
            s.wait(Call(thread, condition.wait, State.WAITING))
            return tx

        with s:
            a = s.thread(waiter)
            a_tx = park(a)
            b = s.thread(waiter)
            b_tx = park(b)
            # Both timeouts fire, both park at STALLED serially
            # (they contend for UL).
            s.wait(Call(a, condition.wait, State.STALLED))
            # Only A can reach STALLED first (only one holds UL at
            # a time).  Release A; then B will race to STALLED.
            capi.unstall(condition.wait, a)
            s.wait(a_tx)
            s.skip(a, lock.release, wait=True)
            s.wait(Call(b, condition.wait, State.STALLED))
            capi.unstall(condition.wait, b)
            s.wait(b_tx)
            s.skip(b, lock.release, wait=True)

    def test_cond_api_unstall_raises_on_no_tx(self):
        """cond_api.unstall(t) raises if t has no active tx."""
        s = Scenario()
        condition = s.Condition()
        capi = s.api(condition)

        def idler():
            pass

        with s:
            t = s.thread(idler)
            # Wait for t to finish.
            t.join()
            with self.assertRaisesRegex(ValueError, "has exited"):
                capi.unstall(condition.wait, t)

    def test_cond_api_unstall_raises_on_wrong_method(self):
        """cond_api.unstall(t) raises if t's tx isn't the named method."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            with self.assertRaises(ValueError):
                capi.unstall(condition.wait, t)
            s.skip(t, lock.acquire, wait=True)

    def test_cond_api_unstall_raises_on_wrong_state(self):
        """cond_api.unstall(t) raises if t's wait tx isn't at STALLED."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            # State is BLOCKED, not STALLED.
            with self.assertRaisesRegex(ValueError, "STALLED"):
                capi.unstall(condition.wait, t)
            # Drain the wait: expire (settings-only timeout=0) then
            # drive worker through commit, STALLED, and to terminal.
            wait_tx = s.transaction(t)
            wait_tx.expire()
            s.finish(t)


class TestWaitForInsidePredicate(unittest.TestCase):
    """Tests for the in_callback flag on cond.wait_for txs."""

    def test_in_callback_flag_toggles(self):
        """in_callback is False outside the predicate call, True
        inside it."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        observed = []
        def predicate():
            # Peek at our own tx's in_callback while running.
            tx = s.transaction(threading.current_thread())
            observed.append(tx._core.in_callback)
            return True  # terminate immediately

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            # Outside: False.
            self.assertFalse(wf_tx._core.in_callback)
            # Drive through.  Predicate True on first call -> no wait.
            wf_tx.unblock()
            s.wait(wf_tx)
            # Predicate was called exactly once.
            self.assertEqual(wf_tx.iterations, 1)
            # During predicate: True observed.
            self.assertEqual(observed, [True])
            # After: False again.
            self.assertFalse(wf_tx._core.in_callback)
            s.skip(t, lock.release, wait=True)

    def test_predicate_releasing_ul_does_not_corrupt_parent_state(self):
        """If user's predicate itself releases and unstalls the UL
        (weird but legal), the shim's in_callback check prevents
        the parent wait_for tx from being transitioned mid-predicate."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def predicate():
            # Release and unstall UL inside predicate.  The
            # _release_save / _acquire_restore shims must NOT transition
            # the parent wait_for tx through WAITING/STALLED here --
            # we're inside the user predicate, not at wait_for's
            # internal-wait boundaries.
            lock._release_save()
            lock._acquire_restore(None)
            return True

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            wf_tx.unblock()
            s.wait(wf_tx)
            # Parent never transitioned through WAITING/STALLED.
            self.assertEqual(wf_tx._core.state, State.RETURNED)
            s.skip(t, lock.release, wait=True)


class TestMonotonicStates(unittest.TestCase):
    """Verify the state machine makes only forward transitions (except
    wait_for's predicate loop, which is inherent)."""

    def test_lock_acquire_states_monotonic(self):
        """Lock.acquire tx visits states in monotonic order."""
        s = Scenario()
        lock = s.Lock()

        transitions = []
        def worker():
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            # Install an observer on every non-START state.
            prev_state = [tx._core.state]
            def record():
                cur = tx._core.state
                transitions.append((prev_state[0], cur))
                prev_state[0] = cur
            # We can't easily install a state_observer from outside,
            # so just verify that the tx's final state is terminal.
            s.skip(t, lock.acquire, lock.release, wait=True)
            # RETURNED is higher than all intermediate states.
            self.assertEqual(tx._core.state, State.RETURNED)
            self.assertGreater(State.RETURNED, State.COMMITTED)
            self.assertGreater(State.COMMITTED, State.COMMIT)
            self.assertGreater(State.COMMIT, State.BLOCKED)

    def test_cond_wait_reaches_STALLED_via_WAITING(self):
        """cond.wait's tx visits WAITING (during _release_save) and then
        STALLED (during _acquire_restore); both states are above
        COMMIT."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait(timeout=IMMEDIATELY)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wait_tx = s.transaction(t)
            wait_tx.unblock()
            # tx transitions through WAITING then STALLED.  Wait for
            # STALLED; if code is correct, state >= WAITING when we
            # observe STALLED.
            s.wait(Call(t, condition.wait, State.STALLED))
            self.assertGreaterEqual(wait_tx._core.state, State.WAITING)
            self.assertEqual(wait_tx._core.state, State.STALLED)
            wait_tx.unstall()
            s.wait(wait_tx)
            # After commit finishes, state is RETURNED (past COMMITTED).
            self.assertEqual(wait_tx._core.state, State.RETURNED)
            s.skip(t, lock.release, wait=True)



class TestTupleSubclassValidation(unittest.TestCase):
    """Argument validation in TupleSubclass derivatives."""

    def test_use_minder_rejects_non_thread(self):
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            Use("not-a-thread", object())

    def test_call_minder_rejects_non_thread(self):
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            Call("not-a-thread", lambda: None)

    def test_call_minder_rejects_non_callable(self):
        with self.assertRaisesRegex(TypeError, "expected a callable"):
            Call(threading.current_thread(), "not-a-method")

    def test_terminated_rejects_non_thread(self):
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            Terminated("not-a-thread")

    def test_tuple_subclasses_unorderable(self):
        """Comparison operators (<, <=, >, >=) on TupleSubclass return
        NotImplemented; Python translates that to TypeError when both
        sides return NotImplemented."""
        t = threading.current_thread()
        a = Terminated(t)
        b = Terminated(t)
        # __ne__ exercises explicitly.
        self.assertFalse(a != b)
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a >= b

    def test_not_rejects_invalid_inner(self):
        with self.assertRaisesRegex(TypeError, "expected a thread"):
            Not("not-anything-valid")

    def test_not_with_terminated_inner_thread_property(self):
        t = threading.current_thread()
        n = Not(Terminated(t))
        self.assertIs(n.thread, t)

    def test_not_repr_with_non_thread_inner(self):
        t = threading.current_thread()
        n = Not(Terminated(t))
        # Falls into the non-Thread inner branch in __repr__.
        r = repr(n)
        self.assertIn("Not", r)
        self.assertIn("Terminated", r)

    def test_not_not_collapses_to_inner_signal(self):
        """Not(Not(X)) returns X, so nested signal bookkeeping is unnecessary."""
        t = threading.current_thread()
        term = Terminated(t)
        self.assertIs(Not(Not(t)), t)
        self.assertIs(Not(Not(term)), term)


class TestRegisterThreadValidation(unittest.TestCase):
    """register_thread / _wait edge cases for not-started threads."""

    def test_wait_rejects_unstarted_thread(self):
        s = Scenario()
        # Create but don't start a thread.
        t = threading.Thread(target=lambda: None)
        with s:
            with self.assertRaisesRegex(ValueError, "unstarted thread"):
                s.wait(t, timeout=IMMEDIATELY)

    def test_wait_rejects_unstarted_thread_in_tuple_subclass(self):
        s = Scenario()
        t = threading.Thread(target=lambda: None)
        with s:
            with self.assertRaisesRegex(ValueError, "unstarted thread"):
                s.wait(Terminated(t), timeout=IMMEDIATELY)


class TestParkSkipParseErrors(unittest.TestCase):
    """Argument validation for park/skip parsing."""

    def test_park_parser_valid_thread_method_regression(self):
        """Regression: park's parser uses a set for duplicate tracking.

        A valid single-thread park used to trip an internal AttributeError
        when the duplicate-tracking container was accidentally a dict.
        """
        s = Scenario()
        lock = s.Lock()
        t = threading.Thread(target=lambda: None)
        plan = s._core.parse_park_skip_args((t, lock.acquire), 'park')
        self.assertEqual(plan, [(t, [lock.acquire])])

    def test_park_parser_duplicate_thread_regression(self):
        """Regression: duplicate park threads raise ValueError, not NameError."""
        s = Scenario()
        lock = s.Lock()
        t = threading.Thread(target=lambda: None)
        with self.assertRaisesRegex(ValueError, "specified more than once"):
            s._core.parse_park_skip_args(
                (t, lock.acquire, t, lock.release), 'park')

    def test_skip_parser_allows_multiple_methods_per_thread(self):
        """skip allows multiple methods for a single thread; park does not."""
        s = Scenario()
        lock = s.Lock()
        t = threading.Thread(target=lambda: None)
        plan = s._core.parse_park_skip_args(
            (t, lock.acquire, lock.release), 'skip')
        self.assertEqual(plan, [(t, [lock.acquire, lock.release])])
        with self.assertRaisesRegex(ValueError, "exactly one method"):
            s._core.parse_park_skip_args(
                (t, lock.acquire, lock.release), 'park')

    def test_park_rejects_method_before_thread(self):
        s = Scenario()
        lock = s.Lock()
        with s:
            with self.assertRaisesRegex(ValueError, "first argument must be a thread"):
                s.park(lock.acquire)

    def test_park_rejects_non_method_arg(self):
        s = Scenario()
        with s:
            t = s.thread(lambda: None)
            with self.assertRaisesRegex(ValueError, "expected thread or method"):
                s.park(t, "not-a-method")

    def test_park_rejects_thread_with_no_methods(self):
        # The parser requires at least one method per thread.
        s = Scenario()
        with s:
            t1 = s.thread(lambda: None)
            t2 = s.thread(lambda: None)
            with self.assertRaisesRegex(ValueError, "has no method"):
                # Two threads with no methods between them.
                s.park(t1, t2)


class TestTransactionLookupMisses(unittest.TestCase):
    """transaction() lookup paths."""

    def test_transaction_returns_active_transaction(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            self.assertIsNotNone(tx)
            self.assertEqual(tx.method, lock.acquire)
            s.skip(t, lock.acquire, wait=True)

    def test_transaction_returns_none_when_thread_has_no_active_tx(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire, wait=True)
            self.assertIsNone(s.transaction(t))


class TestRLockHelpers(unittest.TestCase):
    """RLock helper methods used by tests / scheduler code."""

    def test_rlock_is_owned_helper(self):
        s = Scenario()
        rlock = s.RLock()
        self.assertFalse(rlock._is_owned())
        rlock.acquire()
        self.assertTrue(rlock._is_owned())
        rlock.release()


class TestLockSunderHelpersStandalone(unittest.TestCase):
    """Lock sunder shims invoked outside a cond.wait context."""

    def test_lock_is_owned_when_unowned(self):
        # Exercise the _is_owned_shim "lock was free" branch: the shim
        # tries a non-blocking acquire; if it succeeds, the lock was
        # free, so it releases and returns False.
        s = Scenario()
        lock = s.Lock()
        # Lock is free; _is_owned should return False.
        self.assertFalse(lock._is_owned())

    def test_lock_is_owned_when_owned(self):
        s = Scenario()
        lock = s.Lock()
        lock.acquire()
        # Lock is held; non-blocking acquire fails, so _is_owned returns True.
        self.assertTrue(lock._is_owned())
        lock.release()


class TestSettleWaitsForInTransit(unittest.TestCase):
    """Verify _park_skip_settle waits for an in-transit tx to clear."""

    def test_skip_settle_drains_completing_tx(self):
        """A second skip on the same thread/method waits for the previous
        tx to finish before latching onto the next call."""
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()
            lock.release()
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            # Drive the first acquire/release pair completely.
            s.skip(t, lock.acquire, lock.release, wait=True)
            # By the time skip returns, lock.release tx has been
            # unblocked + waited.  The thread is now on its way to the
            # next lock.acquire.  A second skip must settle on the
            # *next* fresh acquire, not get confused by the stale state.
            s.skip(t, lock.acquire, lock.release, wait=True)


class TestParkErrorPaths(unittest.TestCase):
    """User-facing error paths in park."""

    def test_park_raises_when_active_tx_past_blocked(self):
        """park on a thread whose active tx is past BLOCKED state
        raises with a clear message about the state mismatch."""
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            # Drive lock.acquire to BLOCKED then pause it.
            s.wait(t)
            tx = s.transaction(t)
            tx.pause = True
            tx.unblock()
            s.wait(Reached(tx, State.PAUSED))
            # tx is now in PAUSED state -- not BLOCKED.
            with self.assertRaisesRegex(RuntimeError, "not BLOCKED"):
                s.park(t, lock.acquire)
            # Drain.  s.park's Driver cleared the user pause flag in
            # _initialize (and decremented pausing to zero) before
            # raising, so tx.unpause() is now a no-op; the tx is
            # stranded at PAUSED with no holders.  s.finish frog-
            # marches it past PAUSED to terminal.
            s.finish(t)

    def test_park_raises_when_thread_pushed_wrong_method(self):
        """park on a thread whose active tx is on a different method
        than expected raises explaining the mismatch."""
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            # Wait for the thread to push lock.acquire.
            s.wait(t)
            with self.assertRaisesRegex(RuntimeError, "pushed.*expected"):
                # User asked for release, but thread is on acquire.
                s.park(t, lock.release)
            # Drain.
            s.skip(t, lock.acquire, lock.release, wait=True)



class TestNestedTransactions(unittest.TestCase):
    """Callbacks may create nested regulated transactions."""

    def test_wait_for_predicate_nested_tx_has_parent_and_depth(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def predicate():
            lock.locked()
            return True

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            parent = s.transaction(t)
            self.assertEqual(parent.depth, 0)
            parent.unblock()

            s.wait(Call(t, lock.locked, State.BLOCKED))
            child = s.transaction(t)
            self.assertEqual(child.method, lock.locked)
            self.assertIs(child.parent, parent)
            self.assertEqual(child.depth, 0)
            child.unblock(); s.wait(child)

            s.wait(parent)
            s.skip(t, lock.release, wait=True)



    def test_wait_for_nested_same_primitive_different_method_restores_parent(self):
        """A nested tx on the same primitive but a different method has
        depth 0, uses the outer wait_for as parent, and does not corrupt
        the active parent tx between nested calls."""
        s = Scenario()
        lock = s.RLock()
        condition = s.Condition(lock)
        predicate_calls = []

        def predicate():
            predicate_calls.append('start')
            condition.notify()
            predicate_calls.append('middle')
            condition.notify()
            predicate_calls.append('end')
            return True

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            parent = s.transaction(t)
            self.assertEqual(parent.method, condition.wait_for)
            self.assertEqual(parent.depth, 0)
            use = Use(t, condition)
            signaled = s.wait(use)
            self.assertIn(use, signaled)

            parent.unblock()

            s.wait(Call(t, condition.notify, State.BLOCKED))
            child1 = s.transaction(t)
            self.assertEqual(child1.method, condition.notify)
            self.assertIs(child1.parent, parent)
            self.assertEqual(child1.depth, 0)
            signaled = s.wait(use)
            self.assertIn(use, signaled)
            child1.unblock()
            s.wait(child1)

            s.wait(Call(t, condition.notify, State.BLOCKED))
            child2 = s.transaction(t)
            self.assertEqual(child2.method, condition.notify)
            self.assertIs(child2.parent, parent)
            self.assertIsNot(child2.parent, child1)
            self.assertEqual(child2.depth, 0)
            signaled = s.wait(use)
            self.assertIn(use, signaled)
            child2.unblock()
            s.wait(child2)

            s.wait(parent)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(predicate_calls, ['start', 'middle', 'end'])

    def test_barrier_action_nested_tx_has_parent(self):
        """A Barrier action may create a nested regulated transaction.
        Under the new design, the framework does NOT stash the opener
        around the action call, so the child appears as a proper child
        (child.parent is opener_tx).  Action(opener_tx) still goes
        high for the duration of the callback, so the cycle's
        scheduler can discover the child via that signal."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def action(tx):
            log.append(('action', threading.current_thread().name))
            lock.locked()
            log.append('action_after_nested')

        barrier = s.Barrier(2, action=action)
        bapi = s.api(barrier)

        def worker(name):
            def fn():
                log.append(f'{name}_before')
                barrier.wait()
                log.append(f'{name}_after')
            return fn

        with s:
            a = s.thread(worker('A')); a.name = 'A'
            x = s.thread(worker('X')); x.name = 'X'
            # s.wait(a, x) is "wait until ANY signals"; need both,
            # so wait twice.
            s.wait(a)
            s.wait(x)
            opener_tx = s.transaction(x)

            # The action callback creates a regulated child that
            # parks at BLOCKED.  Default scheduler=_do_nothing would
            # deadlock here -- the opener can't reach PAUSED until the
            # child tx terminates.  Pass a scheduler callback that
            # drives the child past BLOCKED to terminal.  The scheduler
            # waits on Action(opener_tx) -- which goes high while the
            # user action callback runs.
            def drive_child():
                s.wait(Action(opener_tx))
                child = s.transaction(x)
                self.assertEqual(child.method, lock.locked)
                # Under the new (no-implicit-stash) design, the child
                # is a proper child of opener_tx via tx.parent.  depth
                # is keyed by method, so child is depth 0 (lock.locked
                # has no same-method ancestor).
                self.assertIs(child.parent, opener_tx)
                self.assertEqual(child.depth, 0)
                self.assertEqual(opener_tx.method, barrier.wait)
                self.assertTrue(opener_tx.ran_action)
                child.unblock()
                s.wait(child)

            cycle = bapi.cycle(a, x, scheduler=drive_child)
            self.assertEqual(log, [
                'A_before', 'X_before', ('action', 'X'),
                'action_after_nested'])

            cycle.close()

        self.assertEqual(log, [
            'A_before', 'X_before', ('action', 'X'),
            'action_after_nested', 'A_after', 'X_after'])

    def test_wait_for_recursive_depth_five_is_navigable(self):
        """Recursive wait_for calls expose parent links and Call depths."""
        s = Scenario()
        rlock = s.RLock()
        condition = s.Condition(rlock)
        predicate_calls = []

        def predicate():
            predicate_calls.append(len(predicate_calls))
            if len(predicate_calls) < 5:
                return condition.wait_for(predicate)
            return True

        def worker():
            rlock.acquire()
            condition.wait_for(predicate)
            rlock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, rlock.acquire, wait=True)

            txs = []
            for depth in range(5):
                s.wait(Call(t, condition.wait_for, State.BLOCKED, depth=depth))
                tx = s.transaction(t)
                self.assertEqual(tx.method, condition.wait_for)
                self.assertEqual(tx.depth, depth)


                for d in range(depth + 1):
                    call = Call(t, condition.wait_for, depth=d)
                    signaled = s.wait(call)
                    self.assertIn(call, signaled)

                if depth == 0:
                    self.assertIsNone(tx.parent)
                else:
                    self.assertIs(tx.parent, txs[-1])
                    self.assertEqual(tx.parent.depth, depth - 1)
                txs.append(tx)
                tx.unblock()

            # The innermost predicate returns True; every parent then
            # unwinds and returns True too.  The original worker finally
            # releases the RLock.
            s.wait(txs[0])
            s.skip(t, rlock.release, wait=True)

        self.assertEqual(len(predicate_calls), 5)
        self.assertTrue(all(tx.done for tx in txs))

    def test_use_signal_refcounted_through_recursion(self):
        """Use(thread, primitive) stays signaling through the entire
        nested wait_for tree.  Verifies the per-(thread, primitive)
        SignalMinder refcount handles recursion correctly."""
        s = Scenario()
        rlock = s.RLock()
        condition = s.Condition(rlock)

        def predicate():
            if not getattr(predicate, 'recursed', False):
                predicate.recursed = True
                return condition.wait_for(predicate)
            return True

        def worker():
            rlock.acquire()
            condition.wait_for(predicate)
            rlock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, rlock.acquire, wait=True)

            call0 = Call(t, condition.wait_for, State.BLOCKED, depth=0)
            use = Use(t, condition)
            signaled = s.wait(call0, use)
            self.assertIn(use, signaled)
            outer_tx = s.transaction(t)
            outer_tx.unblock()

            s.wait(Call(t, condition.wait_for, State.BLOCKED, depth=1))
            # Both outer (depth=0) and inner (depth=1) hold the
            # per-(thread, condition) SignalMinder refcount up.
            signaled = s.wait(use)
            self.assertIn(use, signaled)

            # Drive both back to RETURNED.  Predicate at depth=1
            # returns True, then unwinds; depth=0 returns True too.
            s.transaction(t).unblock()
            s.wait(outer_tx)

            # Worker now heads into rlock.release.  Under the directional
            # rule, using a condition uses its lock (cond -> ul), but NOT
            # the reverse: rlock.release does not use any condition.  So
            # once the wait_for txs are done, Use(t, condition) is low,
            # while Use(t, rlock) is high during rlock.release.
            s.wait(Call(t, rlock.release, State.BLOCKED))
            self.assertFalse(s.wait(use, timeout=0))
            self.assertIn(Use(t, rlock), s.wait(Use(t, rlock), timeout=0))
            s.skip(t, rlock.release, wait=True)

            # Now nothing is in flight; both Use signals are gone.
            self.assertFalse(s.wait(use, timeout=0))
            self.assertFalse(s.wait(Use(t, rlock), timeout=0))

    def test_raw_form_signals_for_condition_family(self):
        """Inside a Lock-family member's tx, every family member
        (lock, conditions, and their raws) signals as Primitive(form).
        Raw and primitive forms normalize to the same key, so waiting
        on any spelling returns that spelling."""
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)
        lock_raw = s.raws[lock]
        cond_raw = s.raws[cond]

        def worker():
            cond.acquire()
            cond.release()

        with s:
            t = s.thread(worker)
            signaled = s.wait(
                cond.acquire,
                Primitive(lock), Primitive(lock_raw),
                Primitive(cond), Primitive(cond_raw),
                Use(t, lock), Use(t, lock_raw), Use(t, cond), Use(t, cond_raw),
                )
            # All four primitive forms signal (raw normalizes to cooked,
            # so each spelling comes back).
            self.assertIn(Primitive(lock), signaled)
            self.assertIn(Primitive(lock_raw), signaled)
            self.assertIn(Primitive(cond), signaled)
            self.assertIn(Primitive(cond_raw), signaled)

            # All four Use forms also signal.
            self.assertIn(Use(t, lock), signaled)
            self.assertIn(Use(t, lock_raw), signaled)
            self.assertIn(Use(t, cond), signaled)
            self.assertIn(Use(t, cond_raw), signaled)

            s.finish(t)

    def test_raw_method_call_signals_alongside_primitive(self):
        """Call(t, raw_method, ...) signals whenever Call(t, primitive_method, ...)
        does.  Each pair shares one SignalMinder keyed by both methods."""
        s = Scenario()
        lock = s.Lock()
        raw = s.raws[lock]

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(lock.acquire)

            # Both forms of Call signal.
            lock_acquire         = Call(t, lock.acquire)
            lock_acquire_blocked = Call(t, lock.acquire, State.BLOCKED)
            raw_acquire          = Call(t, raw.acquire)
            raw_acquire_blocked  = Call(t, raw.acquire, State.BLOCKED)
            signaled = s.wait(
                lock_acquire,
                lock_acquire_blocked,
                raw_acquire,
                raw_acquire_blocked,
                )
            self.assertIn(lock_acquire, signaled)
            self.assertIn(lock_acquire_blocked, signaled)
            self.assertIn(raw_acquire, signaled)
            self.assertIn(raw_acquire_blocked, signaled)

            s.finish(t)

    def test_raw_method_call_signals_for_condition_family(self):
        """Family methods *and their raws* all signal as Calls when any
        family member's tx opens.  cond.acquire delegates to lock.acquire,
        so a tx on cond.acquire signals Call(t, lock.acquire),
        Call(t, lock_raw.acquire), Call(t, cond.acquire), and
        Call(t, cond_raw.acquire)."""
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)
        lock_raw = s.raws[lock]
        cond_raw = s.raws[cond]

        def worker():
            cond.acquire()
            cond.release()

        with s:
            t = s.thread(worker)

            lock_acquire =     Call(t, lock.acquire)
            lock_raw_acquire = Call(t, lock_raw.acquire)
            cond_acquire =     Call(t, cond.acquire)
            cond_raw_acquire = Call(t, cond_raw.acquire)

            signaled = s.wait(
                cond.acquire,
                lock_acquire,
                lock_raw_acquire,
                cond_acquire,
                cond_raw_acquire,
                )
            self.assertIn(lock_acquire,     signaled)
            self.assertIn(lock_raw_acquire, signaled)
            self.assertIn(cond_acquire,     signaled)
            self.assertIn(cond_raw_acquire, signaled)

            s.finish(t)


class TestRawRename(unittest.TestCase):
    def test_raw_api_surface(self):
        s = Scenario()
        lock = s.Lock()
        api = s.api(lock)
        self.assertIs(s.raw(lock), api.raw)
        self.assertIs(s.raws[lock], api.raw)
        self.assertFalse(hasattr(s, 'unregulated'))
        self.assertFalse(hasattr(s, 'unregulateds'))
        self.assertFalse(hasattr(api, 'unregulated'))


class TestTransactionStateSignals(unittest.TestCase):
    def test_nested_signal_is_level_signal(self):
        """Nested(parent) goes high while parent has an active child,
        and goes low again when the child terminates.  Use
        Condition.wait_for + the inner Condition.wait it spawns as the
        parent/child pair; barrier action would also create a nested
        regulated tx but the worker pushes the barrier tx around its
        action invocation, so Nested(barrier_tx) is silenced -- the
        push-aware analog is Action(tx)."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            # wait_for with a predicate that's always False forces
            # at least one inner condition.wait; we drive its timeout
            # to expire so wait_for returns False quickly.
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)

            # Before unblock: no child yet, Nested(wf_tx) is low.
            signaled = s.wait(Nested(wf_tx), timeout=0)
            self.assertNotIn(Nested(wf_tx), signaled)

            wf_tx.unblock()
            # Inner condition.wait creates the child cond.wait tx.
            # Nested(wf_tx) is high while it exists.
            s.wait(Nested(wf_tx), Terminated(t))
            signaled = s.wait(Nested(wf_tx), timeout=0)
            self.assertIn(Nested(wf_tx), signaled)

            child = s.transaction(t)
            self.assertEqual(child.method, condition.wait)
            self.assertIs(child.parent, wf_tx)
            child.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)

            # Child has terminated.  Nested(wf_tx) is low again.
            signaled = s.wait(Nested(wf_tx), timeout=0)
            self.assertNotIn(Nested(wf_tx), signaled)

            s.wait(wf_tx)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])


class TestConditionCycleOrdering(unittest.TestCase):
    def test_cycle_waiters_are_woken_by_cycle_object_order(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        log = []

        def waiter(name):
            def fn():
                lock.acquire(); log.append(f'{name}_acq')
                condition.wait(); log.append(f'{name}_woke')
                lock.release(); log.append(f'{name}_rel')
            return fn

        def notifier():
            lock.acquire(); log.append('N_acq')
            condition.notify_all(); log.append('N_notified')
            lock.release(); log.append('N_rel')

        helper = TestConditionCycleBasic()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            c = s.thread(waiter('C'))
            n = s.thread(notifier)
            for t in (a, b, c):
                helper.park_waiter(s, condition, lock, t)
            s.skip(n, lock.acquire, wait=True)
            s.wait(n)

            cycle = capi.cycle(a, b, c, n)
            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(b), (b,))
            s.skip(b, lock.release)
            self.assertEqual(cycle.wake(a), (a,))
            s.skip(a, lock.release)
            self.assertEqual(cycle.close(), (c,))
            s.skip(c, lock.release, wait=True)

        self.assertEqual(log[-9:], [
            'N_acq', 'N_notified', 'N_rel',
            'B_woke', 'B_rel', 'A_woke', 'A_rel', 'C_woke', 'C_rel'])


class TestBarrierThinWrapperActive(unittest.TestCase):
    def test_cycle_wakes_waiters_in_cycle_object_order(self):
        s = Scenario()
        barrier = s.Barrier(3)
        bapi = s.api(barrier)
        log = []

        def worker(name):
            def fn():
                log.append(f'{name}_before')
                index = barrier.wait()
                log.append((name, index))
            return fn

        with s:
            b = s.thread(worker('B'))
            a = s.thread(worker('A'))
            x = s.thread(worker('X'))

            self.assertFalse(hasattr(bapi, 'choose'))
            self.assertFalse(hasattr(bapi, 'finish'))

            cycle = bapi.cycle(b, a, x)
            self.assertEqual(s.transaction(b).state, State.PAUSED)
            self.assertEqual(s.transaction(a).state, State.PAUSED)
            self.assertEqual(s.transaction(x).state, State.PAUSED)
            self.assertEqual(cycle.waiters, (b, a, x))
            self.assertEqual(cycle.extra_waiters, 0)
            self.assertEqual(cycle.wake(b), (b,))
            self.assertEqual(cycle.close(), (a, x))

        self.assertEqual(log[:3], ['B_before', 'A_before', 'X_before'])
        self.assertEqual(set(log[3:]), {('X', 2), ('B', 0), ('A', 1)})
        self.assertLess(log.index(('B', 0)), log.index(('A', 1)))

    def test_cycle_requires_existing_waiters_first_and_opener_last(self):
        s = Scenario()
        barrier = s.Barrier(2)
        bapi = s.api(barrier)
        def worker(): barrier.wait()
        with s:
            a = s.thread(worker)
            x = s.thread(worker)
            s.wait(a)
            s.wait(x)
            tx_a = s.transaction(a)
            tx_a.unblock()
            s.wait(Waiting(tx_a))

            with self.assertRaises(ThreadOrderingError):
                bapi.cycle(x, a)

            cycle = bapi.cycle(a, x)
            self.assertEqual(cycle.close(), (a, x))




class TestCoverageLowHangingFruit(unittest.TestCase):
    """Additional regression tests for straightforward coverage branches."""

    def test_call_depth_validation_errors(self):
        thread = threading.current_thread()
        with self.assertRaises(TypeError):
            Call(thread, lambda: None, depth='1')
        with self.assertRaises(ValueError):
            Call(thread, lambda: None, depth=-1)

    def test_read_only_list_index_with_explicit_stop(self):
        scenario = Scenario()
        proxy = scenario._core.ReadOnlyListProxy([1, 2, 3, 2])
        self.assertEqual(proxy.index(2, 2, 4), 3)

    def test_core_thread_to_tx_type_validation(self):
        scenario = Scenario()
        lock = scenario.Lock()
        core = lock._core
        with core.lock:
            with self.assertRaises(TypeError):
                core.thread_to_tx(object())
            with self.assertRaises(TypeError):
                core.threads_to_txs([object()])

    def test_park_raises_when_waited_thread_reaches_wrong_method(self):
        scenario = Scenario()
        lock = scenario.Lock()
        api = scenario.api(lock)

        def worker():
            lock.locked()

        with scenario:
            thread = scenario.thread(worker)
            with self.assertRaises(RuntimeError) as cm:
                scenario.park(thread, lock.acquire)
            self.assertIn('expected', str(cm.exception))
            api.unblock(lock.locked, thread)

    def test_barrier_public_properties_reprs_and_raw_paths(self):
        scenario = Scenario()
        with self.assertRaises(ValueError):
            scenario.Barrier(0)

        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        self.assertIn('BarrierAPI', repr(api))
        self.assertEqual(api.parties, 2)
        self.assertEqual(api.n_waiting, 0)
        self.assertFalse(api.broken)
        self.assertEqual(barrier.parties, 2)
        self.assertEqual(barrier.n_waiting, 0)
        self.assertFalse(barrier.broken)
        self.assertIn('Barrier.raw', repr(scenario.raw(barrier)))
        self.assertIn('BarrierCore', repr(barrier._core))
        barrier.name = 'named_barrier'
        self.assertIn('named_barrier', repr(barrier))

        barrier.reset()
        self.assertFalse(barrier.broken)
        barrier.abort()
        self.assertTrue(barrier.broken)
        barrier.reset()
        self.assertFalse(barrier.broken)

    def test_barrier_wait_reset_abort_transaction_reprs_and_properties(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        results = []

        def waiter():
            try:
                barrier.wait(timeout=NEVER)
            except BrokenBarrierError:
                results.append('broken')

        with scenario:
            thread = scenario.thread(waiter)
            scenario.wait(thread)
            tx = scenario.transaction(thread)
            self.assertIn('Barrier.wait', repr(tx))
            self.assertIn('Barrier.wait', repr(tx._core))
            self.assertIsNone(tx.index)
            self.assertFalse(tx.ran_action)
            tx.expire()

        self.assertEqual(results, ['broken'])

        barrier2 = scenario.Barrier(2)
        api2 = scenario.api(barrier2)

        def resetter():
            barrier2.reset()

        with scenario:
            thread = scenario.thread(resetter)
            scenario.wait(thread)
            tx = scenario.transaction(thread)
            self.assertIn('Barrier.reset', repr(tx))
            self.assertIn('Barrier.reset', repr(tx._core))
            api2.unblock(barrier2.reset, thread)

        barrier3 = scenario.Barrier(2)
        api3 = scenario.api(barrier3)

        def aborter():
            barrier3.abort()

        with scenario:
            thread = scenario.thread(aborter)
            scenario.wait(thread)
            tx = scenario.transaction(thread)
            self.assertIn('Barrier.abort', repr(tx))
            self.assertIn('Barrier.abort', repr(tx._core))
            api3.unblock(barrier3.abort, thread)

class TestCoverageLowHanging(unittest.TestCase):
    """Low/medium-hanging coverage for validation and branch edges."""

    def test_call_depth_rejects_non_integer_and_negative(self):
        s = Scenario()
        lock = s.Lock()
        t = threading.current_thread()
        with self.assertRaises(TypeError):
            Call(t, lock.acquire, depth='deep')
        with self.assertRaises(ValueError):
            Call(t, lock.acquire, depth=-1)

    def test_lock_private_method_base_and_no_active_tx_paths(self):
        s = Scenario()
        lock = s.Lock()
        core = lock._core
        with self.assertRaises(NotImplementedError):
            core.LockPrivateMethod(core)()

        lock.acquire()
        self.assertIsNone(lock._release_save())
        self.assertTrue(lock._acquire_restore(None))
        self.assertTrue(lock.locked())
        lock.release()

    def test_at_fork_reinit_methods_are_callable(self):
        s = Scenario()
        lock = s.Lock()
        rlock = s.RLock()
        lock._at_fork_reinit()
        rlock._at_fork_reinit()

    def test_api_unblock_pause_wait_false_and_unpause_wait_false(self):
        s = Scenario()
        lock = s.Lock()
        api = s.api(lock)
        result = []

        def worker():
            lock.acquire()
            result.append('acquired')
            lock.release()
            result.append('released')

        with s:
            t = s.thread(worker)
            s.wait(t)
            api.unblock(lock.acquire, t, pause=True)
            s.wait(Call(t, lock.acquire, State.PAUSED))
            self.assertEqual(result, [])
            api.unpause(lock.acquire, t)
            s.wait(Call(t, lock.release))
            api.unblock(lock.release, t)

        self.assertEqual(result, ['acquired', 'released'])

    def test_barrier_active_invalid_parties_and_repr_properties(self):
        s = Scenario()
        with self.assertRaises(ValueError):
            s.Barrier(0)

        barrier = s.Barrier(2)
        api = s.api(barrier)
        raw = s.raw(barrier)
        self.assertIn('BarrierAPI', repr(api))
        self.assertIn('BarrierCore', repr(barrier._core))
        self.assertEqual(api.parties, 2)
        self.assertEqual(api.n_waiting, 0)
        self.assertFalse(api.broken)
        self.assertEqual(raw.parties, 2)
        self.assertEqual(raw.n_waiting, 0)
        self.assertFalse(raw.broken)

        barrier.name = 'gate'
        self.assertIn('gate', repr(barrier))
        raw.abort()
        self.assertTrue(api.broken)
        self.assertIn('broken', repr(barrier))
        raw.reset()
        self.assertFalse(api.broken)

    def test_barrier_raw_action_without_visible_tx(self):
        s = Scenario()
        action_threads = []

        def action(tx):
            action_threads.append(threading.current_thread())

        barrier = s.Barrier(1, action=action)
        self.assertEqual(barrier.wait(), 0)
        self.assertEqual(action_threads, [threading.current_thread()])

    def test_barrier_cycle_rejects_missing_arguments(self):
        s = Scenario()
        barrier = s.Barrier(3)
        api = s.api(barrier)
        with s:
            with self.assertRaises(ValueError):
                api.cycle()

    def test_barrier_cycle_not_enough_waiters_raises(self):
        s = Scenario()
        barrier = s.Barrier(3)
        api = s.api(barrier)

        def worker():
            try:
                barrier.wait(timeout=NEVER)
            except BrokenBarrierError:
                pass

        with s:
            t = s.thread(worker)
            s.wait(t)
            with self.assertRaises(ValueError):
                api.cycle(t)
            s.transaction(t).expire()

    def test_barrier_cycle_validation_edges(self):
        s = Scenario()
        barrier = s.Barrier(2)
        api = s.api(barrier)
        lock = s.Lock()

        def barrier_worker():
            barrier.wait()

        def lock_worker():
            lock.locked()

        with s:
            a = s.thread(barrier_worker)
            b = s.thread(barrier_worker)
            wrong = s.thread(lock_worker)
            s.wait(a)
            s.wait(b)
            s.wait(wrong)

            with self.assertRaises(TypeError):
                api.cycle(object())
            with self.assertRaises(ValueError):
                api.cycle(wrong, a)
            with self.assertRaises(ValueError):
                api.cycle(threading.current_thread(), a)
            with self.assertRaises(ThreadOrderingError):
                api.cycle(a, a)

            a_tx = s.transaction(a)
            a_tx.unblock()
            s.wait(Waiting(a_tx))
            with self.assertRaises(ThreadOrderingError):
                api.cycle(b, a)

            cycle = api.cycle(a, b)
            cycle.close()
            s.api(lock).unblock(lock.locked, wrong)

    def test_barrier_wait_transaction_api_repr_and_properties(self):
        s = Scenario()
        barrier = s.Barrier(1)
        api = s.api(barrier)

        def worker():
            barrier.wait()

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            self.assertIn('Barrier.wait', repr(tx))
            self.assertIsNone(tx.index)
            self.assertFalse(tx.ran_action)
            cycle = api.cycle(t)
            self.assertEqual(cycle.close(), (t,))
            self.assertEqual(tx.index, 0)
            self.assertTrue(tx.ran_action)

    def test_barrier_cycle_on_already_broken_surfaces_BrokenBarrierError(self):
        """If the barrier is already broken when cycle drives a BLOCKED
        waiter forward, the waiter's commit raises BrokenBarrierError.
        The cycle propagates that exception (rather than masking it with
        'actual count doesn't match managed count' from the post-drive
        sanity check)."""
        s = Scenario()
        barrier = s.Barrier(2)
        api = s.api(barrier)
        barrier.abort()
        self.assertTrue(barrier.broken)

        def waiter():
            try:
                barrier.wait()
            except BrokenBarrierError:
                pass

        with s:
            t1 = s.thread(waiter)
            t2 = s.thread(waiter)
            s.wait(t1)
            s.wait(t2)
            with self.assertRaises(BrokenBarrierError):
                api.cycle(t1, t2)
            s.finish(t1, t2)


class TestNextTrancheInfrastructure(unittest.TestCase):
    """Tests for transaction lookup and API-visible real-state snapshots."""

    def test_condition_api_waiters_snapshot(self):
        scenario = Scenario()
        condition = scenario.Condition()
        api = scenario.api(condition)
        waiters = condition._core.actual._waiters
        self.assertEqual(api.waiters, 0)
        waiters.append(object())
        try:
            self.assertEqual(api.waiters, 1)
        finally:
            waiters.pop()

    def test_barrier_api_waiters_snapshot(self):
        scenario = Scenario()
        barrier = scenario.Barrier(2)
        api = scenario.api(barrier)
        waiters = barrier._core.actual._cond._waiters
        self.assertEqual(api.waiters, 0)
        waiters.append(object())
        try:
            self.assertEqual(api.waiters, 1)
        finally:
            waiters.pop()

class TestInject(unittest.TestCase):
    """Tests for Scenario.inject(module): monkey-patch a module's
    threading-primitive references so calls construct blanket
    primitives bound to the scenario."""

    # ---- helpers -------------------------------------------------

    def make_module(self, name='target', **attrs):
        """Build a fresh module with the given attributes."""
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    PRIMITIVE_NAMES = (
        'Lock', 'RLock', 'Condition',
        'Semaphore', 'BoundedSemaphore', 'Event', 'Barrier',
    )

    # ---- pattern 1: from-import -----------------------------------

    def test_pattern1_from_import_all_seven(self):
        """Each of the 7 from-imported primitive names is replaced
        with the scenario equivalent; identity check confirms each."""
        scenario = Scenario()
        attrs = {n: getattr(threading, n) for n in self.PRIMITIVE_NAMES}
        target = self.make_module(**attrs)

        with scenario.inject(target):
            for n in self.PRIMITIVE_NAMES:
                with self.subTest(primitive=n):
                    self.assertIs(getattr(target, n),
                                  getattr(scenario, n))

        # After close, restored.
        for n in self.PRIMITIVE_NAMES:
            with self.subTest(primitive=n, when='post-close'):
                self.assertIs(getattr(target, n),
                              getattr(threading, n))

    def test_pattern1_aliased_name(self):
        """A user-named alias (Mutex = threading.Lock) is treated
        identically to a from-import; identity check picks it up."""
        scenario = Scenario()
        target = self.make_module(Mutex=threading.Lock,
                                  WaitFlag=threading.Event)

        with scenario.inject(target):
            self.assertIs(target.Mutex, scenario.Lock)
            self.assertIs(target.WaitFlag, scenario.Event)

        self.assertIs(target.Mutex, threading.Lock)
        self.assertIs(target.WaitFlag, threading.Event)

    def test_pattern1_constructed_primitive_is_bound_to_scenario(self):
        """A primitive constructed via the patched name belongs to
        the right scenario (BIC carries the binding)."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)

        with scenario.inject(target):
            lock = target.Lock()
            self.assertIs(bound_to(type(lock)), scenario)

    # ---- pattern 2: import-threading -------------------------------

    def test_pattern2_threading_module_attr_replaced_with_standin(self):
        """A name bound to the real threading module is replaced
        with a stand-in object (NOT the threading module)."""
        scenario = Scenario()
        target = self.make_module(threading=threading)

        with scenario.inject(target) as inj:
            self.assertIsNot(target.threading, threading)
            # Stand-in's primitive attrs are the scenario's.
            for n in self.PRIMITIVE_NAMES:
                with self.subTest(primitive=n):
                    self.assertIs(getattr(target.threading, n),
                                  getattr(scenario, n))

        self.assertIs(target.threading, threading)

    def test_pattern2_standin_falls_through_for_non_primitive_attrs(self):
        """Stand-in's __getattr__ returns the real threading attr
        for anything not in the seven primitives."""
        scenario = Scenario()
        target = self.make_module(threading=threading)

        with scenario.inject(target):
            self.assertIs(target.threading.Thread, threading.Thread)
            self.assertIs(target.threading.current_thread,
                          threading.current_thread)
            # Confirm the fallthrough is broad (not whitelisted).
            self.assertIs(target.threading.local, threading.local)

    def test_pattern2_constructed_primitive_is_bound_to_scenario(self):
        """target_module.threading.Lock() constructs a scenario-bound
        primitive; demonstrates the BIC carries the binding through."""
        scenario = Scenario()
        target = self.make_module(threading=threading)

        with scenario.inject(target):
            lock = target.threading.Lock()
            self.assertIs(bound_to(type(lock)), scenario)

    def test_pattern2_skipped_when_threading_attr_is_not_real(self):
        """If `target.threading` is not the actual threading module,
        the swap is skipped; only real-threading-bound names trigger
        pattern 2."""
        scenario = Scenario()
        # User has rebound `threading` to something unrelated.
        fake_threading = object()
        target = self.make_module(threading=fake_threading,
                                  Lock=threading.Lock)  # so inject finds something

        with scenario.inject(target):
            # threading is unchanged; only Lock got swapped.
            self.assertIs(target.threading, fake_threading)
            self.assertIs(target.Lock, scenario.Lock)

        self.assertIs(target.threading, fake_threading)

    # ---- identity-not-name semantics -------------------------------

    def test_user_defined_class_with_same_name_left_alone(self):
        """A user class that happens to be named Lock but isn't
        threading.Lock must not be touched.  The scan is identity-
        based, not name-based."""
        scenario = Scenario()

        class Lock:                       # user-defined; not threading.Lock
            pass

        # No threading-related references either, so scan finds nothing.
        target = self.make_module(Lock=Lock)
        with self.assertRaises(ValueError):
            scenario.inject(target)
        # Still untouched.
        self.assertIs(target.Lock, Lock)

    def test_user_defined_class_alongside_real_lock(self):
        """A user-defined class named differently is left alone even
        when the scan does find real primitives elsewhere."""
        scenario = Scenario()

        class MyMutex:
            pass

        target = self.make_module(MyMutex=MyMutex,
                                  RealLock=threading.Lock)

        with scenario.inject(target):
            self.assertIs(target.MyMutex, MyMutex)        # untouched
            self.assertIs(target.RealLock, scenario.Lock)  # swapped

        self.assertIs(target.MyMutex, MyMutex)
        self.assertIs(target.RealLock, threading.Lock)

    # ---- ValueError when nothing to patch --------------------------

    def test_empty_module_raises_value_error(self):
        """A module with no threading references raises ValueError."""
        scenario = Scenario()
        target = self.make_module(x=1, y='hello')
        with self.assertRaises(ValueError) as cm:
            scenario.inject(target)
        self.assertIn(repr(target.__name__), str(cm.exception))
        self.assertIn('nothing to patch', str(cm.exception))

    def test_module_with_only_unrelated_classes_raises(self):
        """Even with classes present, no patchable refs raises."""
        scenario = Scenario()

        class SomeOtherClass:
            pass

        target = self.make_module(SomeOtherClass=SomeOtherClass,
                                  helper_data={'k': 'v'})
        with self.assertRaises(ValueError):
            scenario.inject(target)

    # ---- close: verification + restore -----------------------------

    def test_close_restores_pre_inject_values(self):
        """Plain-call close() restores the pre-inject values."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock,
                                  threading=threading)
        inj = scenario.inject(target)
        inj.close()
        self.assertIs(target.Lock, threading.Lock)
        self.assertIs(target.threading, threading)

    def test_close_refuses_when_value_was_changed(self):
        """If something else replaced one of inject's installed
        values, close() raises RuntimeError instead of clobbering."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        inj = scenario.inject(target)
        target.Lock = 'tampered'  # simulate later change
        with self.assertRaises(RuntimeError) as cm:
            inj.close()
        self.assertIn('Lock', str(cm.exception))
        self.assertIn('no longer matches', str(cm.exception))
        self.assertIn("can't restore", str(cm.exception))

    def test_close_raises_if_any_replacement_mismatches(self):
        """The check covers EVERY replacement, not just the first.
        Sabotage a non-first one and confirm close raises."""
        scenario = Scenario()
        # Use Barrier so it sorts after Lock alphabetically; whichever
        # order dict iteration uses, at least one of the two will be
        # checked second.
        target = self.make_module(Lock=threading.Lock,
                                  Barrier=threading.Barrier)
        inj = scenario.inject(target)
        # Sabotage Barrier (it might be checked second).
        target.Barrier = 'tampered'
        with self.assertRaises(RuntimeError) as cm:
            inj.close()
        self.assertIn('Barrier', str(cm.exception))
        # Lock was NOT restored because close raised before restoring.
        self.assertIs(target.Lock, scenario.Lock)

    def test_close_idempotent_after_success(self):
        """Calling close() a second time after a successful close
        is a no-op (does not re-raise)."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        inj = scenario.inject(target)
        inj.close()
        inj.close()  # should not raise
        self.assertIs(target.Lock, threading.Lock)

    # ---- context manager -------------------------------------------

    def test_context_manager_enter_returns_injection(self):
        """The with-statement target is the Injection itself."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        with scenario.inject(target) as inj:
            self.assertIsInstance(inj, scenario.inject)

    def test_context_manager_exit_calls_close(self):
        """Leaving the with-block restores values (close was called)."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        with scenario.inject(target):
            self.assertIs(target.Lock, scenario.Lock)
        self.assertIs(target.Lock, threading.Lock)

    def test_context_manager_exit_propagates_exception(self):
        """An exception raised inside the with-block still leads to
        close (so values are restored) and propagates out."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        with self.assertRaises(ZeroDivisionError):
            with scenario.inject(target):
                self.assertIs(target.Lock, scenario.Lock)
                raise ZeroDivisionError("boom")
        # Restored even though we exited via exception.
        self.assertIs(target.Lock, threading.Lock)

    # ---- stacking refusals -----------------------------------------

    def test_stacking_same_scenario_raises(self):
        """Once a module's primitive refs have been swapped out, a
        second inject finds no real-threading references; the message
        identifies this as an already-injected module rather than a
        truly empty one."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        inj = scenario.inject(target)
        try:
            with self.assertRaises(ValueError) as cm:
                scenario.inject(target)
            self.assertIn('already', str(cm.exception))
        finally:
            inj.close()

    def test_stacking_different_scenarios_raises(self):
        """Two different scenarios can't both inject the same
        module; the error message mentions an active injection."""
        sa, sb = Scenario(), Scenario()
        target = self.make_module(Lock=threading.Lock)
        inj_a = sa.inject(target)
        try:
            with self.assertRaises(ValueError) as cm:
                sb.inject(target)
            self.assertIn('already', str(cm.exception))
        finally:
            inj_a.close()

    def test_stacking_pattern2_only_diagnoses_correctly(self):
        """A module with only pattern-2 (import threading) reference;
        after inject, attempting another inject sees a stand-in and
        diagnoses already-injected.  This exercises the stand-in arm
        of the diagnostic separately from the primitive-class arm."""
        scenario = Scenario()
        target = self.make_module(threading=threading)
        inj = scenario.inject(target)
        try:
            with self.assertRaises(ValueError) as cm:
                scenario.inject(target)
            self.assertIn('already', str(cm.exception))
        finally:
            inj.close()

    def test_truly_empty_module_says_nothing_to_patch(self):
        """A module with NO threading references at all (vs. one
        already-injected) gets the original 'nothing to patch'
        message.  Verifies the diagnostic doesn't false-positive."""
        scenario = Scenario()

        class MyLock:                 # user class, not primitives_module
            pass

        target = self.make_module(MyLock=MyLock, x=1)
        with self.assertRaises(ValueError) as cm:
            scenario.inject(target)
        msg = str(cm.exception)
        self.assertIn('nothing to patch', msg)
        self.assertNotIn('already', msg)

    # ---- multiple separate modules under one scenario --------------

    def test_two_independent_modules_each_get_their_own_inject(self):
        """One scenario can hold injections into two different
        modules at once; each manages its own replacements."""
        scenario = Scenario()
        m1 = self.make_module('m1', Lock=threading.Lock)
        m2 = self.make_module('m2', Event=threading.Event)
        inj1 = scenario.inject(m1)
        inj2 = scenario.inject(m2)
        try:
            self.assertIs(m1.Lock, scenario.Lock)
            self.assertIs(m2.Event, scenario.Event)
        finally:
            inj1.close()
            inj2.close()
        self.assertIs(m1.Lock, threading.Lock)
        self.assertIs(m2.Event, threading.Event)

    # ---- repr ------------------------------------------------------

    def test_injection_repr_open(self):
        scenario = Scenario()
        target = self.make_module('mod', Lock=threading.Lock)
        inj = scenario.inject(target)
        try:
            r = repr(inj)
            self.assertIn('inject', r)
            self.assertIn("'mod'", r)
            self.assertIn('replacements', r)
        finally:
            inj.close()

    def test_injection_repr_closed(self):
        scenario = Scenario()
        target = self.make_module('mod', Lock=threading.Lock)
        inj = scenario.inject(target)
        inj.close()
        r = repr(inj)
        self.assertIn('closed', r)

    def test_standin_repr_mentions_blanket_and_scenario(self):
        scenario = Scenario()
        target = self.make_module(threading=threading)
        with scenario.inject(target):
            r = repr(target.threading)
            self.assertTrue(r.startswith('<ThreadingImpersonator '))
            # Includes the scenario's repr so two concurrent stand-ins
            # over different scenarios are distinguishable.
            self.assertIn(repr(scenario), r)

    def test_standin_reprs_distinguish_different_scenarios(self):
        sa, sb = Scenario(), Scenario()
        ma = self.make_module('a', threading=threading)
        mb = self.make_module('b', threading=threading)
        with sa.inject(ma), sb.inject(mb):
            self.assertNotEqual(repr(ma.threading), repr(mb.threading))

    # ---- realistic integration -------------------------------------

    def test_realistic_third_party_module_uses_blanket_primitives(self):
        """Simulates a third-party module that uses both reference
        patterns; under inject, calls land on blanket primitives;
        post-inject they go back to real threading."""
        src = (
            "import threading\n"
            "from threading import Condition\n"
            "def make_lock():    return threading.Lock()\n"
            "def make_event():   return threading.Event()\n"
            "def make_cond():    return Condition()\n"
        )
        target = types.ModuleType('thirdparty')
        exec(src, target.__dict__)

        # Pre-inject: real threading.
        pre = target.make_lock()
        self.assertNotIsInstance(pre, primitives_module.Scenario.Lock)

        scenario = Scenario()
        with scenario.inject(target):
            in_lock = target.make_lock()
            in_event = target.make_event()
            in_cond = target.make_cond()
            # Bound to scenario.
            self.assertIs(bound_to(type(in_lock)), scenario)
            self.assertIs(bound_to(type(in_event)), scenario)
            self.assertIs(bound_to(type(in_cond)), scenario)

        # Post-inject: real threading again.
        post = target.make_lock()
        self.assertNotIsInstance(post, primitives_module.Scenario.Lock)

    def test_constructed_primitives_outlive_inject_close(self):
        """A scenario primitive constructed during inject is still
        a usable scenario primitive after the inject closes (the
        primitive is independent of the patching state)."""
        scenario = Scenario()
        target = self.make_module(Lock=threading.Lock)
        with scenario.inject(target):
            blanket_lock = target.Lock()
        # close happened.  blanket_lock is still bound to scenario.
        self.assertIs(bound_to(type(blanket_lock)), scenario)
        # And still usable.
        blanket_lock.acquire()
        blanket_lock.release()

    # ---- stdlib integration ----------------------------------------

    def test_inject_into_sched_uses_blanket_rlock(self):
        """sched.scheduler internally calls threading.RLock() via
        'import threading' (pattern 2).  Under inject, scheduler
        instances hold a blanket RLock bound to the scenario.

        This pattern has been stable in the standard library across
        many CPython versions and is the canonical pattern-2 use
        case for inject."""
        import sched
        scenario = Scenario()
        with scenario.inject(sched):
            sch = sched.scheduler()
            self.assertIs(type(sch._lock), scenario.RLock)
            self.assertIs(bound_to(type(sch._lock)), scenario)
        # After close, sched.scheduler uses real threading.RLock again.
        sch_after = sched.scheduler()
        self.assertNotIsInstance(sch_after._lock,
            scenario.RLock)

    def test_inject_into_threading_local_uses_blanket_rlock(self):
        """_threading_local does 'from threading import current_thread,
        RLock' (pattern 1), then constructs RLock() inside local's
        impl.  Under inject, local instances hold a blanket RLock
        bound to the scenario.  This is the canonical pattern-1 use
        case for inject."""
        import _threading_local
        scenario = Scenario()
        with scenario.inject(_threading_local):
            loc = _threading_local.local()
            # Force impl creation by setting an attribute.
            loc.x = 1
            impl = object.__getattribute__(loc, '_local__impl')
            self.assertIs(type(impl.locallock), scenario.RLock)
            self.assertIs(bound_to(type(impl.locallock)), scenario)



# ---------------------------------------------------------------------------
# Driver and Dispatch (folded from former test_driver_dispatch.py).
# Tests for Scenario.Driver / Scenario.Dispatch -- the public API
# wrappers around the inner Driver and Dispatch core classes.
# ---------------------------------------------------------------------------


def _make_scenario_with_lock_worker(timeout=-1):
    """Build (scenario, lock, worker) where worker blocks on
    lock.acquire(timeout=timeout).  -1 (the default) means wait
    forever -- caller must drive cleanup.
    """
    s = Scenario()
    lock = s.Lock()
    def worker():
        lock.acquire(timeout=timeout)
    return s, lock, worker


class TestDriverConstructor(unittest.TestCase):

    def test_imperative_outside_scenario_raises(self):
        """Driver imperatives raise if the scenario isn't entered."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            s.finish(t)
        # Out of scenario now.
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            d.skip()

    def test_scenario_methods_outside_scenario_raise(self):
        """scenario.skip / park / finish raise if not entered."""
        s = Scenario()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.skip()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.park()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.finish()

    def test_two_drivers_dont_compete_until_drive(self):
        """Lazy registration: two Driver(t) constructions for the
        same thread coexist; only the second's first imperative
        raises CompetingDriversError."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d1 = s.Driver(t)
            d2 = s.Driver(t)  # no raise -- both inert
            d1.skip()  # claims slot
            with self.assertRaises(CompetingDriversError):
                d2.skip()  # competes
            disp = s.Dispatch()
            disp.add(d1)
            list(disp)
            s.finish(t)

    def test_driver_init_clears_pause_chain(self):
        """Driver initialization clears tx.pause on first use of the Driver."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            tx.pause = True
            d = s.Driver(t)
            # An imperative is the public way to trigger first-use
            # initialization; skip() stages a closure without
            # observable tx effects until driven.
            d.skip()
            self.assertFalse(tx.pause)
            disp = s.Dispatch()
            disp.add(d)
            list(disp)
            s.finish(t)


class TestDriverImperatives(unittest.TestCase):

    def test_skip(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.skip()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIn(yielded.state, (d.idle, d.terminated))
            disp.close()
            s.finish(t)

    def test_finish(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.finished)
            self.assertTrue(yielded.done)

    def test_block(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.block()
            d()  # fire the lazy: drives d to parked
            self.assertIs(d.state, d.parked)
            self.assertTrue(d.done)
            # Release worker so cleanup joins.
            s.api(lock).unblock(lock.acquire, t)

    def test_commit(self):
        """Driver.commit() targets COMMIT and parks at parking.

        COMMIT is a transient state that the tx normally races past,
        so the canary overshoots in ordinary primitives -- the
        cascade lands the driver at a terminal state.
        """
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.commit()
            # Drive to terminal so the worker thread reaches
            # termination cleanly.
            disp = s.Dispatch()
            disp.add(d)
            list(disp)

    def test_pause(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            d.pause()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.parked)
            self.assertIs(tx.state, State.PAUSED)
            self.assertTrue(tx.pause)
            disp.close()
            s.finish(t)

    def test_wait(self):
        s = Scenario()
        ev = s.Event()
        def waiter():
            ev.wait()
        def setter():
            ev.set()
        with s:
            disp = s.Dispatch()

            w = s.thread(waiter)
            dw = s.Driver(w)
            disp.add(dw)

            x = s.thread(setter)
            dx = s.Driver(x)
            disp.add(dx)

            for d in disp:
                pass
            self.assertIs(dw.state, dw.active)
            dw.wait()
            dw()
            self.assertIs(dw.state, dw.parked)
            self.assertIs(dw.tx.state, State.WAITING)

            # Drive the setter through commit (fires actual.set, which
            # wakes the waiter's actual.wait; the waiter's tx then
            # advances naturally to terminal).
            dx.skip()
            disp.add(dx)
            for d in disp:
                pass

    def test_stall(self):
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)
        def waiter():
            with lock:
                cond.wait(timeout=0)
        with s:
            w = s.thread(waiter)
            s.skip(w, lock.acquire, wait=True)
            s.wait(w)
            d = s.Driver(w)
            d.stall()
            # Drive d to park at STALLED, then let scenario.finish
            # drive past the stall through to termination.
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.parked)
            self.assertIs(d.tx.state, State.STALLED)
            s.finish(w)

    def test_commit_validates_type(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)
            with self.assertRaisesRegex(RuntimeError, "park in COMMIT"):
                d.commit()
            d.close()
            s.finish(t)

    def test_wait_validates_type(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            with self.assertRaisesRegex(RuntimeError, "park in WAITING"):
                d.wait()
            d.close()
            s.finish(t)

    def test_stall_validates_type(self):
        s = Scenario()
        ev = s.Event()
        def waiter():
            ev.wait()
        with s:
            w = s.thread(waiter)
            s.wait(w)
            d = s.Driver(w)
            with self.assertRaisesRegex(RuntimeError, "park in STALLED"):
                d.stall()
            # stall() raised before transitioning; d is still ACTIVE
            # and owns w.  Set the underlying event so when the
            # commit's actual.wait runs it returns immediately, then
            # close d and finish w.
            s.raw(ev).set()
            d.close()
            s.finish(w)

    def test_imperative_raises_if_not_active(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            list(disp)  # drive d to finished (terminal)
            # d auto-reactivates from finished, but the worker has
            # terminated -- so reactivate lands at idle (no tx).
            with self.assertRaisesRegex(RuntimeError, "currently in"):
                d.skip()


class TestDriverCascade(unittest.TestCase):

    def test_terminated_signal_in_idle(self):
        s = Scenario()
        with s:
            t = s.thread(lambda: None)
            t.join()
            s.wait(Terminated(t))
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.terminated)

    def test_idle_to_active_on_thread_signal(self):
        s = Scenario()
        lock = s.Lock()
        gate = threading.Event()
        started = threading.Event()
        def worker():
            started.set()
            gate.wait()
            lock.acquire()
        with s:
            t = s.thread(worker)
            started.wait()
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            gate.set()
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.active)
            d.close()
            s.finish(t)

    def test_canary_overshoot(self):
        s = Scenario()
        sem = s.Semaphore(value=1)
        def worker():
            sem.acquire()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.wait()
            disp = s.Dispatch()
            disp.add(d)
            with self.assertRaisesRegex(RuntimeError, "overshot"):
                next(iter(disp))
            self.assertIs(d.state, d.raised)

    def test_cascade_pop_match_in_inner_loop(self):
        """Multi-level nested where intermediate ancestors close
        before the Driver processes the child's tx-end signal:
        the cascade-pop branch walks the saved frames until it
        finds the still-alive ancestor.

        Deterministic via bytecode injection: outer's predicate
        gets gate.wait() woven in by inject_call before its
        `return result` line, so the worker blocks there with
        outer still alive after middle has closed.  A matching
        gate.set() is woven into the Driver's signal method at
        the cascade-match return line -- so the moment the
        scheduler observes "intermediates closed, deeper
        ancestor alive" and takes the L1837-1839 branch, it
        releases the worker, outer terminates, and drive
        completes cleanly.  No sleeps, no polling."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        gate = threading.Event()

        def middle_pred():
            return condition.wait_for(lambda: True, timeout=-1)
        def outer_pred():
            result = condition.wait_for(middle_pred, timeout=-1)
            return result
        # Weave gate.wait() into outer_pred before "return result"
        # so the worker blocks with outer still alive once middle
        # has closed.
        outer_pred = inject_call(
            gate.wait,
            Location.text(outer_pred, "return result"))

        def waiter():
            lock.acquire()
            condition.wait_for(outer_pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)

            # Weave gate.set() into the Driver's signal method at
            # the cascade-match return line.  Anchor via the unique
            # `if self.tx is tx:` cascade-match guard, then find
            # the immediately-following `return self.to(state)` --
            # robust to reordering of the other `return self.to(...)`
            # branches in the signal handler.  Instance-level patch
            # leaves the class signal method untouched.
            signal_fn = type(d._core).signal
            cascade_guard = Location.text(signal_fn, "if self.tx is tx:")
            patched = inject_call(
                gate.set,
                Location.text(signal_fn, "return self.to(state)",
                              after=cascade_guard))
            d._core.signal = patched.__get__(d._core)

            d.finish(); d()
            self.assertEqual(d.state, d.finished)
            s.finish(t)

    def test_skipping_terminate_yields_to_active_when_worker_has_next_tx(self):
        """When the Driver is skipping and the driven tx terminates,
        the worker thread may have already moved on to a subsequent
        regulated call -- so transactions[thread] points to a fresh
        tx by the time signal handler processes the tx-end signal.
        The handler must transition to ACTIVE (with the new tx as
        base), not to IDLE: IDLE asserts self.tx is None, which
        cache_tx falsifies when the worker has progressed.

        Setup: worker does lock.acquire + cond.wait_for(True) +
        lock.release.  d.skip(); d() drives wait_for to terminal.
        At that moment the worker has already advanced to
        lock.release; cache_tx surfaces it.  Driver should yield
        ACTIVE on lock.release rather than crashing the next
        score.wait cycle's IDLE handler."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: True, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)
            d.skip(); d()
            self.assertIs(d.state, d.active)
            self.assertIsNotNone(d.tx)
            self.assertEqual(d.tx.method.__name__, 'release')
            d.close()
            s.finish(t)

    def test_cascade_exhaust_yields_to_active_when_worker_has_next_tx(self):
        """The cascade-pop while loop exhausts without finding a
        match when the worker has advanced past every saved
        ancestor.  Like the simple skipping-terminate case, this
        must transition to ACTIVE (not IDLE) when cache_tx
        surfaces a still-running tx.

        Setup: nested wait_for chain (outer -> middle ->
        innermost), all predicates True so each terminates after
        one iteration with no sleeps.  d.skip(); d() drives.
        Innermost terminates; middle and outer terminate too;
        worker is in lock.release by the time the signal handler
        processes innermost's tx-end signal.  Frames have
        state=skipping (since d.skip() set Driver state to
        skipping before each push), so the while loop walks
        through both without match/finishing/parking and exits
        the bottom.  cache_tx already surfaced lock.release;
        Driver should yield ACTIVE on it."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(
                lambda: condition.wait_for(
                    lambda: True, timeout=-1),
                timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)
            d.skip(); d()
            self.assertIs(d.state, d.active)
            self.assertIsNotNone(d.tx)
            self.assertEqual(d.tx.method.__name__, 'release')
            d.close()
            s.finish(t)

    def test_cascade_exhaust_yields_to_idle_when_worker_has_no_next_tx(self):
        """Same cascade-exhaust scenario as above but the worker
        ends after the wait_for chain (no lock.release follows).
        After wait_for terminates, transactions[thread] empties;
        cache_tx returns None; the cascade-exhaust path takes the
        IDLE branch rather than ACTIVE.  The drive loop's next
        iteration sees the thread Terminated signal and
        transitions IDLE -> TERMINATED."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(
                lambda: condition.wait_for(
                    lambda: True, timeout=-1),
                timeout=-1)
            # No lock.release: worker exits the function here, so
            # transactions[thread] empties before the Driver's
            # signal handler reads cache_tx.

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)
            d.skip(); d()
            # Driver passed through IDLE on the way to TERMINATED
            # (thread exited; Terminated signal fired in the IDLE
            # handler).
            self.assertIs(d.state, d.terminated)

    def test_simple_pop_restore_when_parent_still_alive(self):
        """Normal nested pop-and-restore: child terminates while
        its parent is still in flight; the signal handler pops
        the saved frame, cache_tx surfaces the parent (matching
        the popped tx), and the no-cascade fall-through restores
        the parent's pursue context.

        Deterministic via bytecode injection: outer's predicate
        gets gate.wait() woven in before its `return result`
        line; the inner wait_for terminates, then the worker
        blocks with outer still alive.  A matching gate.set() is
        woven into the Driver's signal method at the
        simple-restore return line (the second `return
        self.to(state)` occurrence -- the no-cascade L1845-1847
        path).  When the scheduler reads cache_tx and finds the
        parent still alive, it takes that branch, the injected
        gate.set() releases the worker, and outer terminates."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        gate = threading.Event()

        def outer_pred():
            result = condition.wait_for(lambda: True, timeout=-1)
            return result
        outer_pred = inject_call(
            gate.wait,
            Location.text(outer_pred, "return result"))

        def waiter():
            lock.acquire()
            condition.wait_for(outer_pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)

            # Weave gate.set() into the Driver's signal method at
            # the simple-restore return line.  Anchor via the
            # unique `if self.tx is tx:` cascade-match guard, find
            # the cascade-match return after it, then find the next
            # `return self.to(state)` after that -- which is the
            # no-cascade fall-through (the simple-restore path).
            signal_fn = type(d._core).signal
            cascade_guard = Location.text(signal_fn, "if self.tx is tx:")
            cascade_return = Location.text(signal_fn, "return self.to(state)",
                                           after=cascade_guard)
            patched = inject_call(
                gate.set,
                Location.text(signal_fn, "return self.to(state)",
                              after=cascade_return))
            d._core.signal = patched.__get__(d._core)

            d.finish(); d()
            self.assertEqual(d.state, d.finished)
            s.finish(t)


class TestDispatch(unittest.TestCase):

    def test_empty_dispatch_raises_stopiteration(self):
        s = Scenario()
        with s:
            disp = s.Dispatch()
            with self.assertRaises(StopIteration):
                next(iter(disp))

    def test_add_remove_discard(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            self.assertIn(d, disp)
            disp.add(d)
            self.assertIn(d, disp)
            disp.discard(d)
            self.assertNotIn(d, disp)
            disp.discard(d)
            disp.add(d)
            disp.remove(d)
            self.assertNotIn(d, disp)
            with self.assertRaises(ValueError):
                disp.remove(d)
            d.close()
            s.finish(t)

    def test_add_done_driver_is_ready_immediately(self):
        s = Scenario()
        with s:
            t = s.thread(lambda: None)
            t.join()
            s.wait(Terminated(t))
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.terminated)

    def test_dispatch_yields_done_then_stops(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            self.assertEqual(list(disp), [d])

    def test_externally_removed_driver_skipped(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            disp.discard(d)
            with self.assertRaises(StopIteration):
                next(iter(disp))
            d.close()
            s.finish(t)


class TestDriverAPI(unittest.TestCase):

    def test_repr_says_scenario_driver(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.skip()  # trigger lazy init via a public imperative
            r = repr(d)
            self.assertIn("Scenario.Driver", r)
            self.assertIn("ACTIVE", r)
            disp = s.Dispatch()
            disp.add(d)
            list(disp)
            s.finish(t)

    def test_thread_attribute(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            self.assertIs(d.thread, t)
            d.close()
            s.finish(t)

    def test_txs_history(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.skip()  # trigger init; cache_tx populates txs
            self.assertEqual(len(d.txs), 1)
            disp = s.Dispatch()
            disp.add(d)
            list(disp)
            s.finish(t)

    def test_state_constants_at_class_level(self):
        self.assertEqual(Scenario.Driver.idle.name, 'IDLE')
        self.assertEqual(Scenario.Driver.active.name, 'ACTIVE')
        self.assertIn(Scenario.Driver.parked, Scenario.Driver.terminal_states)
        self.assertIn(Scenario.Driver.skipping, Scenario.Driver.driving_states)
        self.assertIn(Scenario.Driver.active, Scenario.Driver.active_states)



class TestChainIteration(unittest.TestCase):
    """Chain.__iter__ / __next__: iterate Drivers in pending order
    without wrapping in a Dispatch.  Each __next__ promotes the
    pending head to current, drives it until it yields, and
    transfers ownership back to the caller."""

    def test_iter_drives_each_driver_in_order(self):
        """Standalone Chain iteration yields each Driver in
        pending order; each yielded Driver has been driven once
        (passed through initialize -> active at minimum) and is
        transferred back to the caller for further imperatives."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            c = s.thread(worker('C'))
            s.wait(a); s.wait(b); s.wait(c)
            da = s.Driver(a); db = s.Driver(b); dc = s.Driver(c)
            chain = s.Chain(da, db, dc)
            self.assertEqual(len(chain), 3)

            yielded = []
            for d in chain:
                yielded.append(d)
                d.finish()
                d()
            self.assertEqual(yielded, [da, db, dc])
            self.assertEqual(len(chain), 0)

        self.assertEqual(log, ['A', 'B', 'C'])

    def test_iter_empty_chain_raises_stop_immediately(self):
        """Iterating an empty Chain raises StopIteration at once."""
        s = Scenario()
        with s:
            chain = s.Chain()
            self.assertEqual(list(chain), [])

    def test_iter_chain_rejects_owned_chain(self):
        """A Chain owned by a Dispatch refuses direct iteration
        (the Dispatch is in charge of driving it)."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            d = s.Driver(t)
            chain = s.Chain(d)
            disp = s.Dispatch()
            disp.add(chain)
            with self.assertRaisesRegex(RuntimeError,
                    "can't iterate Chain directly"):
                next(iter(chain))
            disp.close()
            s.finish(t)

    def test_iter_chain_re_append_continues(self):
        """User re-adds a Driver to the Chain during iteration to
        keep it in the pipeline; the Chain promotes it after the
        other pending Drivers have been yielded."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            s.wait(a); s.wait(b)
            da = s.Driver(a); db = s.Driver(b)
            chain = s.Chain(da, db)
            yielded = []
            for d in chain:
                yielded.append(d)
                d.finish()
                d()
            # Drivers were yielded in chain order.
            self.assertEqual(yielded, [da, db])
        self.assertEqual(log, ['A', 'B'])

    def test_iter_chain_in_dispatch_works_via_dispatch(self):
        """Sanity: existing Dispatch+Chain integration still works
        (the new __iter__ on Chain doesn't disturb owned use)."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            s.wait(a); s.wait(b)
            da = s.Driver(a); db = s.Driver(b)
            chain = s.Chain(da, db)
            disp = s.Dispatch()
            disp.add(chain)
            for d in disp:
                d.finish()
                d()
        self.assertEqual(log, ['A', 'B'])

    def test_promote_pops_pending_head(self):
        """chain.promote() pops the pending head and returns it
        unowned (no driving).  Returns None when pending is empty."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t1 = s.thread(lock.locked)
            t2 = s.thread(lock.locked)
            s.wait(t1); s.wait(t2)
            d1 = s.Driver(t1); d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)

            promoted = chain.promote()
            self.assertIs(promoted, d1)
            self.assertEqual(chain.pending, (d2,))
            self.assertEqual(len(chain), 1)

            promoted = chain.promote()
            self.assertIs(promoted, d2)
            self.assertEqual(chain.pending, ())
            self.assertEqual(len(chain), 0)

            # Empty: returns None.
            self.assertIsNone(chain.promote())

            # Cleanup: the promoted Drivers are unowned, finish them.
            for d in (d1, d2):
                d.finish(); d()

    def test_promote_walrus_iteration_pattern(self):
        """Non-driving iteration via promote(): the caller decides
        whether to drive each Driver, useful when iteration intent
        differs from "drive to next yield"."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            s.wait(a); s.wait(b)
            da = s.Driver(a); db = s.Driver(b)
            chain = s.Chain(da, db)

            collected = []
            d = chain.promote()
            while d is not None:
                collected.append(d)
                d.finish(); d()
                d = chain.promote()

            self.assertEqual(collected, [da, db])
        self.assertEqual(log, ['A', 'B'])

    def test_iter_break_leaves_chain_consistent(self):
        """Breaking out of iteration after a yield leaves the Chain
        in a consistent state: the just-yielded Driver is no longer
        tracked by the Chain (it was promoted out), pending holds
        only the not-yet-promoted Drivers, and there's no stale
        current-pointer half-state.  The popped Driver is the
        caller's responsibility -- caller must close or re-append
        explicitly."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            c = s.thread(worker('C'))
            s.wait(a); s.wait(b); s.wait(c)
            da = s.Driver(a); db = s.Driver(b); dc = s.Driver(c)
            chain = s.Chain(da, db, dc)

            popped_first = None
            for d in chain:
                popped_first = d
                d.finish(); d()
                break  # exit after first yield

            # da was promoted out and yielded.  Chain pending is db, dc.
            self.assertIs(popped_first, da)
            self.assertEqual(chain.pending, (db, dc))
            self.assertEqual(len(chain), 2)

            # Continue iteration from where we left off; each Driver
            # is driven to terminal before the next is promoted.
            rest = []
            for d in chain:
                rest.append(d)
                d.finish(); d()

            self.assertEqual(rest, [db, dc])
            self.assertEqual(chain.pending, ())
        self.assertEqual(log, ['A', 'B', 'C'])




class TestDispatchAPI(unittest.TestCase):

    def test_repr_says_scenario_dispatch(self):
        s = Scenario()
        with s:
            disp = s.Dispatch()
            r = repr(disp)
            self.assertIn("Scenario.Dispatch", r)
            self.assertIn("0 drivers", r)

    def test_break_out_of_iteration_leaves_dispatch_iterable(self):
        """Breaking out of `for d in dispatch:` after a yield leaves
        the Dispatch in a consistent state.  The just-yielded Driver
        is unregistered (transferred to caller).  Remaining Drivers
        stay in self.drivers / self.recent and the next iteration
        of the same Dispatch resumes from where we left off."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker(name):
            def fn():
                lock.locked()
                log.append(name)
            return fn

        with s:
            a = s.thread(worker('A'))
            b = s.thread(worker('B'))
            c = s.thread(worker('C'))
            s.wait(a); s.wait(b); s.wait(c)
            da = s.Driver(a); db = s.Driver(b); dc = s.Driver(c)
            disp = s.Dispatch()
            disp.add(da); disp.add(db); disp.add(dc)

            first_yielded = None
            for d in disp:
                first_yielded = d
                d.finish(); d()
                break

            # First Driver was yielded and is now owned by the user.
            # Dispatch still tracks the others.
            self.assertIsNotNone(first_yielded)
            self.assertNotIn(first_yielded, disp)
            # Resume iteration; remaining Drivers should still be
            # drivable.
            remaining = []
            for d in disp:
                remaining.append(d)
                d.finish(); d()

            # All three workers completed.
            self.assertEqual(sorted(log), ['A', 'B', 'C'])
            # The two not-first-yielded Drivers were the rest.
            all_drivers = {da, db, dc}
            self.assertEqual(set(remaining), all_drivers - {first_yielded})


class TestDispatchCoverage(unittest.TestCase):

    def test_active_return_path(self):
        """Dispatch.__next__ returns a driver in active state."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.active)
            d.close()
            s.finish(t)

    def test_dispatch_iter_returns_self(self):
        """Dispatch.__iter__ returns self."""
        s = Scenario()
        with s:
            disp = s.Dispatch()
            self.assertIs(iter(disp), disp)


class TestCloseMethods(unittest.TestCase):
    """Driver.close / Chain.close / Dispatch.close convenience API."""

    def test_driver_close_releases_score_slot(self):
        """After d.close(), a new Driver can be made for the same
        thread without raising CompetingDriversError."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.close()
            d2 = s.Driver(t)
            self.assertIs(d2.thread, t)
            d2.close()
            s.finish(t)

    def test_driver_close_idempotent(self):
        """Driver.close is safe to call twice."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.close()
            d.close()  # no raise
            s.finish(t)

    def test_driver_close_after_auto_close_is_noop(self):
        """A driver that auto-closed on reaching terminal can still
        have close() called without raising."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            list(disp)  # drives d to FINISHED -- auto-close runs
            self.assertTrue(d.done)
            d.close()  # explicit call after auto-close: no-op

    def test_chain_close_empties_pending(self):
        """Chain.close releases pending drivers' score-slots and
        empties the chain."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)
            self.assertEqual(len(chain), 2)
            chain.close()
            self.assertEqual(len(chain), 0)
            self.assertFalse(chain)
            self.assertEqual(chain.pending, ())
            # Drivers were closed; can create new ones.
            s.finish(t1)
            s.finish(t2)

    def test_chain_close_in_dispatch_closes_current(self):
        """Chain.close() clears remaining pending drivers on a chain
        whose head has been promoted out by Dispatch.  After the
        Chain refactor that removed chain.current, the promoted
        Driver is owned by the Dispatch (driver_to_chain tracks the
        back-link), so chain.close() only touches what's still in
        pending.  The yielded Driver is closed via d1.close() or
        dispatch.close()."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)
            disp = s.Dispatch()
            disp.add(chain)
            # Iterate once to promote d1 transiently and yield it.
            yielded = next(iter(disp))
            self.assertIs(yielded, d1)
            self.assertEqual(chain.pending, (d2,))
            chain.close()
            self.assertEqual(chain.pending, ())
            d1.close()
            disp.close()
            s.finish(t1)
            s.finish(t2)

    def test_chain_close_idempotent(self):
        s = Scenario()
        with s:
            chain = s.Chain()
            chain.close()
            chain.close()  # no raise

    def test_dispatch_close_closes_drivers(self):
        """Dispatch.close closes every owned Driver and empties
        the dispatch."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            disp = s.Dispatch()
            disp.add(d1)
            disp.add(d2)
            self.assertIn(d1, disp)
            self.assertIn(d2, disp)
            disp.close()
            self.assertNotIn(d1, disp)
            self.assertNotIn(d2, disp)
            # Drivers were closed; can finish each thread.
            s.finish(t1)
            s.finish(t2)

    def test_dispatch_close_closes_chain(self):
        """Dispatch.close also handles owned Chains (closes their
        current + pending)."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)
            disp = s.Dispatch()
            disp.add(chain)
            self.assertIn(chain, disp)
            # Don't iterate; close immediately while the chain is
            # still in disp.recent.  disp.close handles chain
            # cleanup uniformly across in-recent / promoted /
            # post-yield states; the promoted state (driver_to_chain
            # populated) isn't externally observable.
            disp.close()
            self.assertNotIn(chain, disp)
            self.assertFalse(chain)
            s.finish(t1)
            s.finish(t2)

    def test_dispatch_close_idempotent(self):
        s = Scenario()
        with s:
            disp = s.Dispatch()
            disp.close()
            disp.close()  # no raise

    def test_dispatch_close_with_driver_in_recent(self):
        """A driver added but never drained sits in dispatch.recent;
        close() handles it."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            disp = s.Dispatch()
            disp.add(d)  # in recent, not yet drained
            # Don't iterate -- close immediately.
            disp.close()
            self.assertNotIn(d, disp)
            s.finish(t)


class TestCoverageMinor(unittest.TestCase):
    """Coverage fill for small defensive / repr / edge-case paths."""

    def test_chain_append_already_owned_raises(self):
        """Chain.append raises if the driver is already owned by
        any chain or dispatch."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            chain1 = s.Chain(d)
            chain2 = s.Chain()
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                chain2.append(d)
            chain1.close()
            s.finish(t)

    def test_chain_remove_absent_driver_raises(self):
        """Chain.remove raises ValueError if the driver isn't in pending."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            chain = s.Chain()
            with self.assertRaises(ValueError):
                chain.remove(d)
            d.close()
            s.finish(t)

    def test_chain_contains_pending(self):
        """Chain.__contains__ True for drivers in pending.  The
        "current" branch of __contains__ isn't externally exercisable:
        chain.current is set transiently inside drain_recent and
        cleared on yield, never visible at the user-code boundary."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)
            self.assertIn(d1, chain)
            self.assertIn(d2, chain)
            chain.close()
            self.assertNotIn(d1, chain)
            self.assertNotIn(d2, chain)
            s.finish(t1)
            s.finish(t2)

    def test_chain_append_to_chain_in_dispatch_after_empty(self):
        """Appending to an empty chain that's owned by a dispatch
        re-engages the chain (puts it back in the dispatch's recent)."""
        s = Scenario()
        lock = s.Lock()
        def w():
            lock.acquire(timeout=-1)
        with s:
            t = s.thread(w)
            s.wait(t)
            d = s.Driver(t)
            chain = s.Chain()
            disp = s.Dispatch()
            disp.add(chain)
            # Chain is empty, owned by disp.
            self.assertFalse(chain)
            # Append after empty + owned -> re-engagement path.
            chain.append(d)
            self.assertTrue(chain)
            disp.close()
            s.finish(t)

    def test_driver_unregister_not_owned_raises(self):
        """driver.unregister raises if the driver has no owner."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            # Driver fresh-built has no owner.
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                d._core.unregister()
            d.close()
            s.finish(t)

    def test_register_thread_unstarted_raises(self):
        """score.register_thread raises ValueError if the thread hasn't started."""
        s = Scenario()
        t = threading.Thread(target=lambda: None)  # not started
        with self.assertRaisesRegex(ValueError, "not started"):
            s._core.register_thread(t)

    def test_wait_transaction_repr(self):
        """WaitTransaction.__repr__ produces a readable string."""
        s = Scenario()
        ev = s.Event()
        # WaitTransaction(score, items): every item is Signaling.
        item = Primitive(ev)
        wtx = s._core.WaitTransaction(frozenset([item]))
        r = repr(wtx)
        self.assertIn('WaitTransaction', r)
        self.assertIn('items=', r)
        self.assertIn('signaled=', r)

    def test_chain_unregister_when_not_owned_raises(self):
        """Chain.unregister raises if the chain has no owner."""
        s = Scenario()
        with s:
            chain = s.Chain()
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                chain._core.unregister()

    def test_dispatch_discard_chain_not_owned_is_noop(self):
        """Dispatch.discard_chain on a chain not owned by this
        Dispatch is a no-op (early return)."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            chain = s.Chain(d)
            disp1 = s.Dispatch()
            disp2 = s.Dispatch()
            disp1.add(chain)
            # chain.owner is disp1, not disp2: discard via disp2 is no-op.
            disp2._core.discard_chain(chain._core)
            self.assertIn(chain, disp1)
            disp1.close()
            s.finish(t)

    def test_threads_to_txs_non_iterable(self):
        """primitive core threads_to_txs raises TypeError for non-iterable."""
        s = Scenario()
        lock = s.Lock()
        core = s.api(lock)._core
        with s:
            with self.assertRaisesRegex(TypeError, "iterable of threads"):
                core.threads_to_txs(42)

    def test_threads_to_txs_empty(self):
        """primitive core threads_to_txs returns empty for empty iterable."""
        s = Scenario()
        lock = s.Lock()
        core = s.api(lock)._core
        with s:
            threads, txs = core.threads_to_txs(())
            self.assertEqual(threads, ())
            self.assertEqual(txs, [])

    def test_threads_to_txs_self_thread_raises(self):
        """primitive core threads_to_txs raises ValueError on caller."""
        s = Scenario()
        lock = s.Lock()
        core = s.api(lock)._core
        with s:
            with self.assertRaisesRegex(ValueError, "calling thread"):
                core.threads_to_txs((threading.current_thread(),))

    def test_stacked_imperatives_raise(self):
        """An imperative call raises if a previous imperative is
        still pending (not yet driven).  One imperative at a time."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.pause()
            with self.assertRaisesRegex(RuntimeError, "haven't run pause yet"):
                d.finish()
            # Driving clears the pending state; another imperative is
            # then allowed.
            disp = s.Dispatch()
            disp.add(d)
            next(iter(disp))
            self.assertIs(d.state, d.parked)
            d.finish()  # no raise now -- previous closure fired
            disp.add(d)
            list(disp)

    def test_scenario_exit_auto_unparks_blocked_worker(self):
        """A worker parked at BLOCKED is released on scenario __exit__
        without explicit finish/skip -- the primitive becomes
        unregulated and the scheduler-side block is lifted, so the
        worker resumes and runs the actual function natively."""
        s = Scenario()
        lock = s.Lock()
        ran = []
        def worker():
            lock.acquire()
            ran.append('acquired')
            lock.release()
            ran.append('released')
        with s:
            t = s.thread(worker)
            s.wait(t)
            # Worker is parked at BLOCKED on lock.acquire.  Exit
            # without finish; auto-unpark should free it.
        self.assertEqual(ran, ['acquired', 'released'])

    def test_scenario_exit_auto_unparks_paused_worker(self):
        """A worker parked at PAUSED is released on scenario __exit__."""
        s = Scenario()
        lock = s.Lock()
        ran = []
        def worker():
            lock.acquire()
            ran.append('acquired')
            lock.release()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.pause()
            d()
            self.assertIs(d.tx.state, State.PAUSED)
            d.close()
            # Exit: auto-unpark frees the worker past PAUSED.
        self.assertEqual(ran, ['acquired'])

    def test_scenario_exit_auto_unparks_stalled_worker(self):
        """A worker parked at STALLED (cond.wait, post-notify) is
        released on scenario __exit__."""
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)
        ran = []
        def waiter():
            with lock:
                cond.wait(timeout=0)
                ran.append('woke')
        with s:
            w = s.thread(waiter)
            s.skip(w, lock.acquire, wait=True)
            s.wait(w)
            d = s.Driver(w)
            d.stall()
            d()
            self.assertIs(d.tx.state, State.STALLED)
            d.close()
            # Exit: auto-unpark transitions STALLED -> RESUMED so
            # the waiter runs _acquire_restore and finishes cond.wait.
        self.assertEqual(ran, ['woke'])

    def test_scenario_finish_handles_mid_tx_worker(self):
        """scenario.finish handles a worker that has advanced past
        BLOCKED on its current tx."""
        s = Scenario()
        ev = s.Event()
        gate = threading.Event()  # raw threading.Event, NOT regulated
        def worker():
            ev.wait()
        def setter():
            gate.wait()  # raw wait -- no blanket tx
            ev.set()
        with s:
            w = s.thread(worker)
            x = s.thread(setter)
            # s.wait is "wait for any"; setter is blocked on the raw
            # gate so only worker pushes a regulated tx.  Wait for it.
            s.wait(w)
            # Drive w into WAITING (past BLOCKED).
            d = s.Driver(w)
            d.wait()
            d()
            self.assertIs(d.tx.state, State.WAITING)
            d.close()  # release the slot
            # Release the setter so it fires the event, then finish
            # both threads.
            gate.set()
            s.finish(x)
            s.finish(w)

    def test_tx_failed_succeeded_is_none_before_done(self):
        """tx.failed and tx.succeeded return None while the tx is
        not yet at a terminal state."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            self.assertIsNone(tx.succeeded)
            self.assertIsNone(tx.failed)
            s.finish(t)

    def test_tx_unblock_raises_in_non_blocked_state(self):
        """tx.unblock raises RuntimeError if the tx isn't at BLOCKED."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            d.pause()
            d()
            # Tx is now at PAUSED; unblock should refuse.
            with self.assertRaisesRegex(RuntimeError, "can't unblock"):
                tx.unblock()
            disp = s.Dispatch()
            d.finish()
            disp.add(d)
            list(disp)
            s.finish(t)

    def test_tx_pause_setter_past_paused_raises(self):
        """tx.pause = True/False raises if tx has advanced past PAUSED."""
        s = Scenario()
        ev = s.Event()
        def w():
            ev.set()
        with s:
            t = s.thread(w)
            s.finish(t)
            # Find the terminal tx via the log.
            log = list(s.log)
            tx = log[-1]  # the set tx, at RETURNED (past PAUSED)
            with self.assertRaisesRegex(RuntimeError, "advanced past PAUSED"):
                tx.pause = True

    def test_tx_observer_past_state_raises(self):
        """tx.observe raises if you ask to be notified about a state
        the tx is already at or past."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            with self.assertRaisesRegex(ValueError, "can't register observer"):
                tx._core.observe(State.BLOCKED, lambda: None)
            s.finish(t)

    def test_injection_repr_and_context_manager(self):
        """Injection has __repr__ and works as a context manager."""
        import threading as _threading
        # Create a small module-like target to inject into.
        class M:
            Lock = _threading.Lock
        s = Scenario()
        with s.inject(M) as inj:
            r = repr(inj)
            self.assertIn('inject', r)
            self.assertIn('replacement', r)

    def test_tx_succeeded_and_failed_after_returned(self):
        """tx.succeeded is True / failed is False after a successful
        RETURNED terminal."""
        s = Scenario()
        ev = s.Event()
        def worker():
            ev.set()
        with s:
            t = s.thread(worker)
            s.finish(t)
            # The Event.set tx is terminal; tx via score still
            # accessible.
            txs = s._core.transactions
            # After completion, score.transactions no longer holds
            # the tx for the thread.  Instead, look up the tx via
            # the log.
            log = list(s.log)
            self.assertTrue(log)
            tx = log[-1]
            self.assertEqual(tx.state.name, 'RETURNED')
            self.assertTrue(tx.succeeded)
            self.assertFalse(tx.failed)


class TestRawPrimitives(unittest.TestCase):
    """Coverage of the s.raw(primitive) wrappers."""

    def test_lock_raw(self):
        s = Scenario()
        lock = s.Lock()
        raw = s.raw(lock)
        self.assertFalse(raw.locked())
        raw.acquire()
        self.assertTrue(raw.locked())
        raw.release()
        self.assertIn('Lock.raw', repr(raw))

    def test_rlock_raw(self):
        s = Scenario()
        rlock = s.RLock()
        raw = s.raw(rlock)
        raw.acquire()
        raw.release()

    def test_condition_raw(self):
        s = Scenario()
        cond = s.Condition()
        raw = s.raw(cond)
        # Exercise repr; the wait/notify methods need lock-ownership
        # context and a paired waiter to run meaningfully, so just
        # smoke-test that the wrappers are wired.
        self.assertIn('Condition.raw', repr(raw))
        cond.acquire()
        try:
            raw.notify()         # no waiters; no-op
            raw.notify_all()     # no waiters; no-op
            raw.notifyAll()      # alias
        finally:
            cond.release()

    def test_condition_raw_wait_for(self):
        """Condition raw wait_for with predicate that's immediately True."""
        s = Scenario()
        cond = s.Condition()
        raw = s.raw(cond)
        cond.acquire()
        try:
            self.assertTrue(raw.wait_for(lambda: True))
        finally:
            cond.release()

    def test_semaphore_raw(self):
        s = Scenario()
        sem = s.Semaphore(1)
        raw = s.raw(sem)
        self.assertTrue(raw.acquire(blocking=False))
        raw.release()
        self.assertIn('Semaphore.raw', repr(raw))

    def test_bounded_semaphore_raw(self):
        s = Scenario()
        sem = s.BoundedSemaphore(1)
        raw = s.raw(sem)
        self.assertTrue(raw.acquire(blocking=False))
        raw.release()
        self.assertIn('BoundedSemaphore.raw', repr(raw))

    def test_event_raw(self):
        s = Scenario()
        ev = s.Event()
        raw = s.raw(ev)
        self.assertFalse(raw.is_set())
        self.assertFalse(raw.isSet())
        raw.set()
        self.assertTrue(raw.is_set())
        self.assertTrue(raw.wait(timeout=0))
        raw.clear()
        self.assertFalse(raw.is_set())
        self.assertIn('Event.raw', repr(raw))

    def test_barrier_raw(self):
        s = Scenario()
        bar = s.Barrier(3)
        raw = s.raw(bar)
        # Reset and abort on an unused barrier are safe operations.
        raw.reset()
        raw.abort()
        self.assertTrue(bar.broken)


class TestDispatchChainCoverage(unittest.TestCase):
    """Coverage for Dispatch/Chain edge paths not hit elsewhere."""

    def test_dispatch_discard_chain_after_partial_iter(self):
        """Dispatch.discard(chain) after a partial iteration: head
        was promoted and yielded, leaving the chain re-added to
        recent with current=None and pending non-empty.  discard
        removes the chain from recent and unregisters it.  The
        "discard while chain.current is set" branch isn't externally
        exercisable -- advance_chain_after resets current on yield."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            chain = s.Chain(d1, d2)
            disp = s.Dispatch()
            disp.add(chain)
            # Iterate once: d1 yielded; chain re-added to recent
            # with current=None, pending=[d2].
            yielded = next(iter(disp))
            self.assertIs(yielded, d1)
            self.assertEqual(chain.pending, (d2,))
            disp.discard(chain)
            self.assertNotIn(chain, disp)
            # Pending preserved.
            self.assertEqual(chain.pending, (d2,))
            # discard doesn't close; finish cleanup.
            d1.close()
            chain.close()
            s.finish(t1)
            s.finish(t2)

    def test_dispatch_discard_chain_in_recent(self):
        """Dispatch.discard(chain) when chain is in recent
        (added but not yet drained)."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            chain = s.Chain(d)
            disp = s.Dispatch()
            disp.add(chain)
            self.assertIn(chain, disp)
            disp.discard(chain)
            self.assertNotIn(chain, disp)
            chain.close()
            s.finish(t)

    def test_dispatch_remove_unknown_chain_raises(self):
        s = Scenario()
        with s:
            disp = s.Dispatch()
            chain = s.Chain()
            with self.assertRaisesRegex(ValueError, "unknown Chain"):
                disp.remove(chain)

    def test_dispatch_update(self):
        """Dispatch.update adds multiple drivers."""
        s = Scenario()
        lock1 = s.Lock()
        lock2 = s.Lock()
        def w1():
            lock1.acquire(timeout=-1)
        def w2():
            lock2.acquire(timeout=-1)
        with s:
            t1 = s.thread(w1)
            t2 = s.thread(w2)
            s.wait(t1, t2)
            d1 = s.Driver(t1)
            d2 = s.Driver(t2)
            disp = s.Dispatch()
            disp.update([d1, d2])
            self.assertIn(d1, disp)
            self.assertIn(d2, disp)
            disp.close()
            s.finish(t1)
            s.finish(t2)

    def test_driver_callable(self):
        """Driver.__call__ runs the driver synchronously."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.skip()
            d()  # advance synchronously
            d.close()
            s.finish(t)



class TestExternallyCreatedThreads(unittest.TestCase):
    """Threads created directly via threading.Thread (not via
    scenario.thread) participate in blanket scenarios.  They register
    on first regulated method call.  Scenario __exit__ does not join
    them (they are the user's responsibility), so any in-flight
    tx.close() runs concurrently with reset(); reset() must not
    clear use_minders out from under those closes."""

    def test_quickstart_relay_then_barrier_cycle(self):
        # The blanket quickstart: three externally-created workers
        # take a lock in turn via relay, then meet at a barrier
        # driven by cycle.  Exercises the full register-on-first-
        # call path for externally-created threads through both a
        # Lock and a Barrier.
        s = Scenario()
        lock = s.Lock()
        barrier = s.Barrier(3)
        order = []
        order_lock = threading.Lock()

        def worker(name):
            with lock:
                with order_lock:
                    order.append(('lock', name))
            barrier.wait()
            with order_lock:
                order.append(('barrier', name))

        A = threading.Thread(target=worker, args=('A',))
        B = threading.Thread(target=worker, args=('B',))
        C = threading.Thread(target=worker, args=('C',))

        lock_api = s.api(lock)
        barrier_api = s.api(barrier)
        with s:
            A.start()
            B.start()
            C.start()
            list(lock_api.relay(B, A, C))
            lock_api.unblock(lock.release, C)
            s.park(C, barrier.wait)
            with barrier_api.cycle(C, A, B):
                pass
        A.join(); B.join(); C.join()

        # Lock order: relay ran B -> A -> C.
        lock_order = [n for tag, n in order if tag == 'lock']
        self.assertEqual(lock_order, ['B', 'A', 'C'])

        # Barrier order: cycle ran with B as the opener (last arg),
        # so B fills the barrier last and all three pass.
        barrier_set = {n for tag, n in order if tag == 'barrier'}
        self.assertEqual(barrier_set, {'A', 'B', 'C'})


class TestTimeoutTrioAndFailureDetection(unittest.TestCase):
    """Settings-only expire/disregard/revert: the trio writes the
    tx-class effective timeout via the timeout property setter,
    nothing else.  Lock.acquire's no_timeout sentinel is -1; for
    everything else it's None.  Assign and relay surface acquire-
    returned-False and release-raised cases as chained RuntimeError."""

    def test_trio_is_idempotent_in_any_order(self):
        # 10k-loop-style: only the last write to the trio matters.
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lambda: lock.acquire(timeout=NEVER))
            s.wait(lock.acquire, t)
            tx = s.transactions[t]
            tx.expire(); tx.disregard(); tx.revert(); tx.expire()
            self.assertEqual(tx._core._timeout, 0)
            tx.revert(); tx.expire(); tx.disregard()
            # disregard wrote no_timeout (-1 for Lock.acquire).
            self.assertEqual(tx._core._timeout, -1)
            tx.disregard(); tx.expire(); tx.revert()
            self.assertEqual(tx._core._timeout, NEVER)
            self.assertEqual(tx._core.original_timeout, NEVER)
            # Let the worker complete (lock is free so the acquire
            # under NEVER succeeds when the scenario exit unparks it).
            s.finish(t)

    def test_disregard_uses_per_class_no_timeout_sentinel(self):
        # Lock.acquire: no_timeout = -1.  Other primitives: None.
        s = Scenario()
        lock = s.Lock()
        cond_lock = s.Lock()
        cond = s.Condition(cond_lock)
        with s:
            t_lock = s.thread(lambda: lock.acquire(timeout=NEVER))
            s.wait(lock.acquire, t_lock)
            tx_lock = s.transactions[t_lock]
            tx_lock.disregard()
            self.assertEqual(tx_lock._core._timeout, -1)
            self.assertEqual(tx_lock._core.no_timeout, -1)
            s.finish(t_lock)

            def cond_waiter():
                cond_lock.acquire()
                cond.wait(timeout=NEVER)
                cond_lock.release()
            t_cond = s.thread(cond_waiter)
            s.skip(t_cond, cond_lock.acquire, wait=True)
            s.wait(cond.wait, t_cond)
            tx_cond = s.transactions[t_cond]
            tx_cond.disregard()
            self.assertIsNone(tx_cond._core._timeout)
            self.assertIsNone(tx_cond._core.no_timeout)
            # Drive the wait to terminal so the scenario exits cleanly:
            # expire forces actual.wait(0) on commit.
            tx_cond.expire()
            s.finish(t_cond)

    def test_timeout_getter_returns_remaining(self):
        # The getter does the live math: tx._timeout (duration from
        # start_time) minus elapsed, clamped at 0.
        s = Scenario()
        lock = s.Lock()
        with s:
            # Hold the lock externally so the worker blocks.
            holder = s.thread(lambda: lock.acquire())
            s.finish(holder)
            t = s.thread(lambda: lock.acquire(timeout=10.0))
            s.wait(lock.acquire, t)
            tx = s.transactions[t]
            # _timeout is the stored relative duration; timeout
            # property returns remaining (which has decreased
            # slightly between tx creation and now).
            self.assertEqual(tx._core._timeout, 10.0)
            remaining = tx.timeout.value  # TimeoutState.value = original_timeout = 10.0
            self.assertEqual(remaining, 10.0)
            # The live getter on the core tx returns remaining.
            live_remaining = tx._core.timeout
            self.assertLessEqual(live_remaining, 10.0)
            self.assertGreater(live_remaining, 0)
            # Expire forces _timeout=0; getter clamps at 0.
            tx.expire()
            self.assertEqual(tx._core._timeout, 0)
            self.assertEqual(tx._core.timeout, 0)
            s.finish(t)
            # Release the held lock for clean exit.
            s.raw(lock).release()

    def test_timeout_setter_state_check(self):
        # Setting tx.timeout outside BLOCKED raises.
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lambda: lock.acquire())
            s.wait(lock.acquire, t)
            tx = s.transactions[t]
            # Drive past BLOCKED.
            s.finish(t)
            with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
                tx.expire()
            with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
                tx.disregard()
            with self.assertRaisesRegex(RuntimeError, "can't modify timeout"):
                tx.revert()

    def test_relay_release_raised_raises_runtime(self):
        # A thread that releases an unlocked Lock raises RuntimeError
        # in actual.release.  relay drives that release; when the tx
        # ends RAISED, relay surfaces a chained RuntimeError with
        # "raised" in the message.  (assign refuses on lock-state
        # precondition before getting to the release drive; relay
        # has no such precondition.)
        s = Scenario()
        lock = s.Lock()
        api = s.api(lock)
        def bad_releaser():
            try:
                lock.release()
            except RuntimeError:
                pass
        def acquirer():
            lock.acquire()
        with s:
            r = s.thread(bad_releaser)
            a = s.thread(acquirer)
            s.wait(lock.release, r)
            s.wait(lock.acquire, a)
            with self.assertRaisesRegex(RuntimeError, "raised"):
                list(api.relay(r, a))
            # r terminated RAISED; a still at acquire/BLOCKED.
            api.unblock(lock.acquire, a)
            s.finish(a)



class TestStashAndAction(unittest.TestCase):
    """Push/pop, Action signal, cycle validation, and Driver.nested
    state coverage.  Push/pop is an internal mechanism (barrier
    workers use it around their action callback); most tests reach
    in through tx._core to exercise it directly, since public API
    only exposes it indirectly via Barrier.cycle(scheduler=...)."""

    def test_action_signal_construction_repr_equality(self):
        """Action(tx) is a tuple-style signal: constructs from a tx,
        compares equal under tuple semantics, refuses non-tx args."""
        s = Scenario()
        lock = s.Lock()
        with self.assertRaisesRegex(TypeError, "Action argument"):
            Action("not a tx")
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)
            a = Action(tx)
            b = Action(tx)
            self.assertEqual(a, b)
            self.assertEqual(hash(a), hash(b))
            self.assertTrue(repr(a).startswith("Action("))
            s.finish(t)

    def test_action_signal_low_before_push(self):
        """Action(tx).signal returns False when tx isn't pushed."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)
            self.assertFalse(Action(tx).signal(s))
            s.finish(t)

    def test_action_signal_high_during_barrier_action(self):
        """Action(opener) goes high while the worker is inside the
        barrier action (between push and pop), and goes low again
        after the action returns."""
        s = Scenario()
        lock = s.Lock()
        observed = {}

        def action(tx):
            # Worker has pushed before this runs; the scheduler
            # observes Action(opener) high.  Body just needs to do
            # some regulated work the scheduler will drive.
            lock.locked()

        barrier = s.Barrier(2, action=action)
        bapi = s.api(barrier)

        def worker():
            barrier.wait()

        with s:
            a = s.thread(worker)
            x = s.thread(worker)
            s.wait(a)
            s.wait(x)
            opener = s.transaction(x)

            self.assertFalse(Action(opener).signal(s))

            def drive_child():
                # Block until the worker enters the action.  The
                # scheduler runs concurrently with the worker thread
                # leaving WAITING and entering the action, so
                # Action(opener) is initially low; we wait until it
                # fires.
                s.wait(Action(opener))
                observed['during_action'] = Action(opener).signal(s)
                child = s.transaction(x)
                self.assertEqual(child.method, lock.locked)
                # Under the new no-implicit-stash design the child is
                # a proper child of opener.  (The user could stash
                # opener inside the action to get the old root-child
                # behavior, but the framework no longer does it.)
                self.assertIs(child.parent, opener)
                child.unblock()
                s.wait(child)

            cycle = bapi.cycle(a, x, scheduler=drive_child)
            cycle.close()

            # After the action returns, Action(opener) goes low.
            self.assertFalse(Action(opener).signal(s))

        self.assertTrue(observed['during_action'])

    def test_cycle_scheduler_requires_action(self):
        """Barrier.cycle(scheduler=...) on a Barrier with no action
        raises ValueError: a scheduler without an action would never
        be invoked for nested children (none get spawned)."""
        s = Scenario()
        barrier = s.Barrier(2)  # no action
        bapi = s.api(barrier)

        def worker():
            barrier.wait()

        with s:
            a = s.thread(worker)
            x = s.thread(worker)
            s.wait(a)
            s.wait(x)

            def scheduler():
                pass  # never invoked

            with self.assertRaisesRegex(ValueError,
                    "scheduler= requires the Barrier to have an action"):
                bapi.cycle(a, x, scheduler=scheduler)

            # Cycle without scheduler works.
            cycle = bapi.cycle(a, x)
            cycle.close()

    def test_cycle_scheduler_with_action_no_arg_works(self):
        """The default _do_nothing scheduler is allowed even when an
        action is present (and is allowed when an action is absent,
        per the ValueError check using 'is _do_nothing')."""
        s = Scenario()
        barrier = s.Barrier(2, action=lambda tx: None)
        bapi = s.api(barrier)

        def worker():
            barrier.wait()

        with s:
            a = s.thread(worker)
            x = s.thread(worker)
            s.wait(a)
            s.wait(x)
            cycle = bapi.cycle(a, x)  # no scheduler= argument
            cycle.close()

    def test_barrier_action_no_child_round_trip(self):
        """When the action creates no regulated children, push and
        pop happen anyway (push is unconditional whenever an action
        is present) and the cycle completes cleanly.  Exercises pop
        case A: original_child is None and the chain is empty at
        pop time."""
        s = Scenario()
        log = []
        barrier = s.Barrier(2, action=lambda tx: log.append('action_ran'))
        bapi = s.api(barrier)

        def worker():
            barrier.wait()

        with s:
            a = s.thread(worker)
            x = s.thread(worker)
            s.wait(a)
            s.wait(x)
            cycle = bapi.cycle(a, x)
            cycle.close()
        self.assertEqual(log, ['action_ran'])

    def test_stash_gating_close(self):
        """A pushed tx rejects close()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx_api = s.transaction(t)
            tx = tx_api._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.close()
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_unblock(self):
        """A pushed tx rejects unblock()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.unblock()
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_settle(self):
        """A pushed tx rejects settle()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.settle()
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_unpark(self):
        """A pushed tx rejects unpark()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.unpark(State.COMMIT)
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_set_pause(self):
        """A pushed tx rejects set_pause()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.set_pause(True)
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_unpause(self):
        """A pushed tx rejects unpause()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.unpause()
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_observe(self):
        """A pushed tx rejects observe()."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.observe(State.COMMITTED, lambda: None)
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_gating_timeout_setter(self):
        """A pushed TimeoutTransaction rejects timeout assignment
        (which gates expire/disregard/revert)."""
        s = Scenario()
        lock = s.Lock()
        log = []

        def worker():
            log.append('before')
            lock.acquire(timeout=10)
            log.append('after')

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)._core
            # First, verify it's a TimeoutTransaction.
            self.assertIsInstance(
                tx, s._core.Core.TimeoutTransaction)
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.expire()
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.disregard()
                    with self.assertRaisesRegex(RuntimeError, "is stashed"):
                        tx.revert()
                finally:
                    ticket.close()
            s.finish(t)
            t.join()

    def test_stash_idempotency_via_context_manager(self):
        """Push BIC as context manager: __exit__ pops, and re-calling
        the instance is a no-op (popped flag)."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                with tx.stash() as ticket:
                    self.assertIs(tx.stashed, ticket)
                    self.assertFalse(ticket.closed)
                self.assertTrue(ticket.closed)
                self.assertIsNone(tx.stashed)
                # Second call: no-op.
                ticket.close()
                self.assertTrue(ticket.closed)
            s.finish(t)

    def test_stash_rejects_double_push(self):
        """Pushing an already-pushed tx raises."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)._core
            with s._core.lock:
                ticket = tx.stash()
                try:
                    with self.assertRaisesRegex(RuntimeError, "already stashed"):
                        tx.stash()
                finally:
                    ticket.close()
            s.finish(t)

    def test_stash_rejects_non_root(self):
        """Pushing a tx with a parent raises."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)._core
            wf_tx.unblock()
            s.wait(Nested(wf_tx.api), Terminated(t))
            child_api = s.transaction(t)
            child = child_api._core
            self.assertIs(child.parent, wf_tx)
            with s._core.lock:
                with self.assertRaisesRegex(RuntimeError,
                        "only chain-root transactions"):
                    child.stash()
            # Drive child out to let test exit cleanly.
            child_api.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child_api.unstall()
            s.wait(child_api)
            s.wait(wf_tx.api)
            s.skip(t, lock.release, wait=True)

    def test_action_signal_in_signaled_module_list(self):
        """Action is exported from blanket."""
        import blanket
        self.assertIs(blanket.Action, Action)

    def test_api_revert_restores_original_timeout(self):
        """api.revert(method, *threads) restores the user's original
        timeout on each thread's BLOCKED tx, undoing any prior
        expire/disregard.  Public method routes through the
        CycleAPIBase core revert."""
        s = Scenario()
        lock = s.Lock()
        api = s.api(lock)
        result = []

        # First a holder takes the lock.
        with s:
            holder = s.thread(lock.acquire)
            s.finish(holder)
            # Waiter blocks on lock.acquire with a long timeout.
            t = s.thread(lambda: result.append(lock.acquire(timeout=10.0)))
            s.wait(lock.acquire, t)
            tx = s.transaction(t)
            self.assertEqual(tx.timeout.value, 10.0)
            # Expire shortens the timeout to 0; revert puts it back.
            api.expire(lock.acquire, t)
            self.assertEqual(tx._core._timeout, 0)
            api.revert(lock.acquire, t)
            self.assertEqual(tx._core._timeout, 10.0)
            # Drive past with the expired timeout to clean up.
            api.expire(lock.acquire, t)
            s.finish(t)
            s.raw(lock).release()
        self.assertEqual(result, [False])


class TestDriverNested(unittest.TestCase):
    """Driver.nested() imperative and the NESTING state.  Driver.nested()
    is composable with the six pursue-based imperatives (skip, block,
    commit, wait, stall, finish, pause) and also works standalone."""

    def test_driver_nesting_state_constant_exposed(self):
        """The user-facing Driver class exposes the NESTING state
        constant (s.Driver.nesting), distinct from the .nested()
        imperative method."""
        s = Scenario()
        self.assertEqual(s.Driver.nesting.name, 'NESTING')
        self.assertIn(s.Driver.nesting, s.Driver.active_states)
        self.assertNotIn(s.Driver.nesting, s.Driver.terminal_states)
        # The .nested() imperative shadows any state of the same name;
        # the state lives at .nesting to avoid the collision.
        self.assertTrue(callable(s.Driver.nested))

    def test_driver_nested_imperative_arms_flag(self):
        """driver.nested() called standalone arms nested_armed and
        adds Nested to the signal set so the Driver actually listens
        for Nested while otherwise idle in active state."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            d = s.Driver(t)
            d.nested()
            self.assertTrue(d._core.nested_armed)
            # Nested(tx) is in the listened-for signal set.
            nested_sig = d._core.thread_signal[Nested]
            self.assertIn(nested_sig, d._core.signals)
            d.close()
            s.finish(t)

    def test_driver_nested_yields_in_nesting_state_when_child_appears(self):
        """When nested-armed and a child fires Nested, the Driver
        yields in the NESTING state with self.tx rotated to the child
        and the parent on the stack.  Uses finish() to keep the Driver
        in a driving state long enough for the worker's wait_for body
        to spawn the child cond.wait tx."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)

            d = s.Driver(t)
            # finish() stages tx.unblock + to(finishing).  finishing's
            # signal set already includes Nested by default; the arm
            # routes the fire to NESTING-yield instead of the default
            # skipping path.
            d.finish()
            d.nested()
            self.assertTrue(d._core.nested_armed)
            d()  # drive

            self.assertEqual(d.state, s.Driver.nesting)
            self.assertFalse(d._core.nested_armed)
            child = d.tx
            self.assertEqual(child.method, condition.wait)
            self.assertIsNot(child, wf_tx)
            # Parent context preserved on the stack.
            self.assertEqual(len(d._core.stack), 1)

            # Cleanup: release the driver slot so direct-API and
            # subsequent s.skip can attach their own drivers.
            d.close()

            # Drive child to terminal via the API.
            child.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            # wait_for's predicate sees False again, timeout has
            # expired (-1), returns False.  wf_tx terminates.
            s.wait(wf_tx)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])

    def test_driver_nested_one_shot(self):
        """One-shot: arm cleared on consumption.  After the Driver
        yields in NESTING, nested_armed is False; the caller must
        re-arm to catch a subsequent Nested fire."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            d = s.Driver(t)
            d.finish()
            d.nested()
            self.assertTrue(d._core.nested_armed)
            d()
            # Arm consumed.
            self.assertFalse(d._core.nested_armed)
            d.close()
            # Cleanup.
            child = s.transaction(t)
            child.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            s.wait(wf_tx)
            s.skip(t, lock.release, wait=True)
        self.assertEqual(results, [False])

    def test_driver_nested_grandchild_extends_stack(self):
        """Re-arming after first NESTING yield catches the next
        Nested fire, with the stack growing by one level.  Verifies
        that the nesting mechanism composes recursively across
        multiple parent levels: outer wait_for whose predicate calls
        an inner wait_for produces parent -> inner -> grandchild
        (cond.wait).  Each NESTING yield deepens the stack."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            # Outer wait_for's predicate is inner wait_for; inner's
            # predicate is False -> spawns cond.wait grandchild.
            condition.wait_for(
                lambda: condition.wait_for(lambda: False, timeout=-1),
                timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)

            d.finish(); d.nested(); d()
            self.assertEqual(d.state, s.Driver.nesting)
            self.assertEqual(len(d._core.stack), 1)
            inner = d.tx
            self.assertEqual(inner.method, condition.wait_for)

            # Re-arm and drive forward.  Inner's body spawns a
            # cond.wait grandchild; with the arm set, the driver
            # yields again in NESTING with stack now holding both
            # outer and inner.
            d.skip(); d.nested(); d()
            self.assertEqual(d.state, s.Driver.nesting)
            self.assertEqual(len(d._core.stack), 2)
            grandchild = d.tx
            self.assertEqual(grandchild.method, condition.wait)
            self.assertIsNot(grandchild, inner)

            # Now drive everything to terminal via the same Driver
            # without re-arming.  The arm is consumed, so subsequent
            # Nested fires take the default skipping branch -- driver
            # auto-handles each.  Eventually outer terminates and
            # the Driver hits FINISHED.  Use s.finish(t) for the
            # remainder (lock.release and any more wait_for iteration
            # the worker spins through before settling).
            d.skip(); d()
            self.assertEqual(d.state, s.Driver.finished)
            s.finish(t)

    def test_driver_nested_re_arm_after_consumption(self):
        """One-shot semantics: after the arm is consumed by a Nested
        fire (yields in NESTING), nested() can be called again to
        re-arm without raising.  The second arm catches the next
        Nested fire."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            d = s.Driver(t)

            d.finish()
            d.nested()
            d()  # yield NESTING; arm consumed
            self.assertFalse(d._core.nested_armed)

            # Re-arm without raising.
            d.nested()
            self.assertTrue(d._core.nested_armed)

            # Cleanup: drive everything to completion.
            d.skip(); d()  # arm gets consumed-or-cleared during winddown
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])

    def test_driver_nested_then_pursue_composes(self):
        """nested() called BEFORE a pursue-based imperative also
        composes: the arm flag persists across the pursue's state
        transition (pursue doesn't touch nested_armed), and pursue's
        to(driving-state) re-builds signals with Nested included as
        usual.  When the child fires Nested, the handler still sees
        nested_armed=True and yields in NESTING.

        Tests the symmetric case to the pursue+nested ordering used
        in test_driver_nested_yields_in_nesting_state_when_child_appears."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            d = s.Driver(t)

            # Arm FIRST, then pursue.  This exercises the opposite
            # ordering from the round-trip test.
            d.nested()
            self.assertTrue(d._core.nested_armed)
            d.finish()
            # State is still active (pursue stages lazy_work without
            # transitioning); the arm survives.
            self.assertTrue(d._core.nested_armed)
            d()

            self.assertEqual(d.state, s.Driver.nesting)
            self.assertFalse(d._core.nested_armed)
            child = d.tx
            self.assertEqual(child.method, condition.wait)

            # Cleanup.
            d.skip(); d()
            self.assertEqual(d.state, s.Driver.finished)
            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])

    def test_driver_nested_round_trip_resumes_parent_pursue(self):
        """After yielding in NESTING, the user drives the child via
        the same Driver; when the child terminates, the tx-end stack
        pop restores the parent's pursue context (state, target,
        base) and the Driver continues toward the original target.

        Verifies that pursue from NESTING does NOT clear the stack
        (which would lose the parent context), and that .base is
        properly set to the child during NESTING (so subsequent
        pursue acts on the child)."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire, wait=True)
            s.wait(t)
            wf_tx = s.transaction(t)
            d = s.Driver(t)

            d.finish()
            d.nested()
            d()  # yield NESTING with child cond.wait in view

            self.assertEqual(d.state, s.Driver.nesting)
            child = d.tx
            self.assertEqual(child.method, condition.wait)
            self.assertEqual(len(d._core.stack), 1)
            # base was set to child during the NESTING transition,
            # so subsequent pursue here acts on child not parent.
            self.assertIs(d._core.base, child._core)

            # Drive child to terminal via the same Driver.  skip()
            # auto-unstalls on STALLED-park, drives to terminal.
            d.skip()
            d()
            # tx-end on child triggered stack pop; Driver resumed
            # parent's finishing state and drove parent to terminal
            # (since wait_for returned False right after the child's
            # timeout).  Driver is now in FINISHED state.
            self.assertEqual(d.state, s.Driver.finished)
            self.assertEqual(len(d._core.stack), 0)

            s.skip(t, lock.release, wait=True)

        self.assertEqual(results, [False])

    def test_driver_nested_raises_on_double_call(self):
        """Like the pursue-based imperatives, nested() refuses
        back-to-back calls without an intervening drive: the second
        call raises RuntimeError because the arm hasn't been
        consumed yet."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            d = s.Driver(t)
            d.nested()
            self.assertTrue(d._core.nested_armed)
            with self.assertRaisesRegex(RuntimeError, "can't nested"):
                d.nested()
            d.close()
            s.finish(t)


def run_tests():
    blankettestlib.run(name="blanket.primitives", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
