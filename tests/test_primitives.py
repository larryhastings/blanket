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


import contextlib
import importlib.util
import inspect
from itertools import count
from pathlib import Path
import sys
import threading
import queue
import re
import blanket
from threading import BrokenBarrierError
import time
import types
import unittest

from blanket import Scenario
from blanket import Call, Use, Terminated, Not, TimeoutState, DriverStatus, ThreadOrderingError, Blocked, Waiting, Paused, Nested, TransactionState, State, Reached, Action, Predicate
from blanket import Stalled, Commit, Committed, Exiting, CompetingDriversError
from blanket import Location, inject_call
from blanket import primitives as primitives_module
from big.boundinnerclass import bound_to


_temp_primitives_module_counter = count(1)


@contextlib.contextmanager
def _temporarily_removed_attribute(obj, name):
    """Temporarily remove obj.name, restoring it on exit.

    Used for stdlib-version-shape tests: load blanket.primitives while the
    stdlib looks like an older Python, without needing that interpreter handy.
    """
    sentinel = object()
    old = getattr(obj, name, sentinel)
    if old is sentinel:
        yield
        return

    delattr(obj, name)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _temporarily_assigned_attribute(obj, name, value):
    """Temporarily assign obj.name to value, restoring it on exit."""
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)

def _compatibility_repr_like(real_object, blanket_object):
    """Return real_object's repr with blanket_object's uppercase address."""
    real_repr = repr(real_object)
    expected = re.sub(r" at 0[xX][0-9a-fA-F]+(?=>)",
                      f" at {hex(id(blanket_object)).upper()}",
                      real_repr,
                      count=1)
    if expected == real_repr:
        raise AssertionError(f"repr has no object address to replace: {real_repr!r}")
    return expected


def _load_fresh_primitives_module():
    """Load blanket/primitives.py under a unique temporary module name."""
    module_name = f"_blanket_primitives_test_{next(_temp_primitives_module_counter)}"
    path = Path(primitives_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@contextlib.contextmanager
def _fresh_primitives_module():
    module = _load_fresh_primitives_module()
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


class TestTestHelpers(unittest.TestCase):
    def test_temporarily_removed_attribute_when_attribute_is_absent(self):
        class Holder:
            pass
        holder = Holder()
        with _temporarily_removed_attribute(holder, 'missing'):
            self.assertFalse(hasattr(holder, 'missing'))
        self.assertFalse(hasattr(holder, 'missing'))

    def test_temporarily_assigned_attribute_removes_new_attribute(self):
        class Holder:
            pass
        holder = Holder()
        with _temporarily_assigned_attribute(holder, 'temporary', 37):
            self.assertEqual(holder.temporary, 37)
        self.assertFalse(hasattr(holder, 'temporary'))

    def test_compatibility_repr_requires_address(self):
        with self.assertRaisesRegex(AssertionError, "repr has no object address"):
            _compatibility_repr_like(37, object())

    def test_load_fresh_primitives_module_cleans_up_after_exec_failure(self):
        class Loader:
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise RuntimeError("synthetic import failure")

        spec = importlib.machinery.ModuleSpec("synthetic_primitives_failure", Loader())
        original = importlib.util.spec_from_file_location
        try:
            importlib.util.spec_from_file_location = lambda name, path: spec
            with self.assertRaisesRegex(RuntimeError, "synthetic import failure"):
                _load_fresh_primitives_module()
        finally:
            importlib.util.spec_from_file_location = original

        leaked = [name for name in sys.modules
                  if name.startswith("_blanket_primitives_test_")
                  and getattr(sys.modules[name], "__spec__", None) is spec]
        self.assertEqual(leaked, [])


# Standard timeout values for tests.  Use these instead of magic numbers:
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
            scenario.skip(t, lock.acquire)

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
            scenario.skip(t, lock.acquire)

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

    def test_lock_api_unblock_accepts_base_tx_tuple(self):
        """Primitive API thread selectors accept (thread, base_tx)."""
        scenario = Scenario()
        rlock = scenario.RLock()
        condition = scenario.Condition(rlock)
        lock = scenario.Lock()
        results = []

        def predicate():
            results.append(lock.locked())
            return True

        def worker():
            rlock.acquire()
            condition.wait_for(predicate, timeout=NEVER)
            rlock.release()

        with scenario:
            t = scenario.thread(worker)
            scenario.skip(t, rlock.acquire)
            scenario.wait(Call(t, condition.wait_for, State.BLOCKED))
            base_tx = scenario.transaction(t)
            base_tx.unblock()
            scenario.wait(Call((t, base_tx), lock.locked, State.BLOCKED))
            child_tx = scenario.transaction(t)
            self.assertIs(child_tx.parent, base_tx)

            returned = scenario.api(lock).unblock(lock.locked, (t, base_tx))
            self.assertEqual(returned, ((t, base_tx),))
            self.assertTrue(child_tx.done)

            scenario.wait(base_tx)
            scenario.skip(t, rlock.release)

        self.assertEqual(results, [False])

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
            scenario.skip(t, cond.wait)

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
            scenario.skip(t, rlock.acquire)
            scenario.wait(t)
            r = repr(rlock)
            self.assertIn('locked', r)
            self.assertNotIn('owner=0', r)
            self.assertIn('count=1', r)
            scenario.skip(t, rlock.release)

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

    def test_semaphore_api_repr(self):
        """SemaphoreAPI repr shows SemaphoreAPI class name."""
        scenario = Scenario()
        sem = scenario.Semaphore()
        api = scenario.api(sem)
        r = repr(api)
        self.assertIn('SemaphoreAPI', r)
        self.assertNotIn('BoundedSemaphoreAPI', r)

    def test_bounded_semaphore_api_repr(self):
        """BoundedSemaphoreAPI repr shows BoundedSemaphoreAPI class name."""
        scenario = Scenario()
        sem = scenario.BoundedSemaphore()
        api = scenario.api(sem)
        r = repr(api)
        self.assertIn('BoundedSemaphoreAPI', r)

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

    def test_queue_primitives_masquerade_as_queue_primitives(self):
        scenario = Scenario()
        pairs = [
            (scenario.Queue(), queue.Queue()),
            (scenario.LifoQueue(), queue.LifoQueue()),
            (scenario.PriorityQueue(), queue.PriorityQueue()),
        ]
        if hasattr(queue, 'SimpleQueue'):
            pairs.insert(0, (scenario.SimpleQueue(), queue.SimpleQueue()))

        for primitive, real in pairs:
            with self.subTest(primitive=real.__class__.__name__):
                self.assert_masquerades_as(primitive, real)

    def test_queue_compatibility_repr_matches_stdlib_except_upper_id(self):
        scenario = Scenario()
        pairs = [
            (scenario.Queue(), queue.Queue()),
            (scenario.LifoQueue(), queue.LifoQueue()),
            (scenario.PriorityQueue(), queue.PriorityQueue()),
        ]
        if hasattr(queue, 'SimpleQueue'):
            pairs.insert(0, (scenario.SimpleQueue(), queue.SimpleQueue()))

        for primitive, real in pairs:
            with self.subTest(primitive=real.__class__.__name__):
                self.assertEqual(repr(primitive),
                    _compatibility_repr_like(real, primitive))

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

    def test_api_class_aliases_exposed_for_isinstance(self):
        # The API wrapper classes are surfaced on the Scenario API at
        # both class and instance level, as stable isinstance targets.
        scenario = Scenario()
        alias_names = ('LockAPI', 'RLockAPI', 'ConditionAPI', 'EventAPI',
                       'SemaphoreAPI', 'BoundedSemaphoreAPI', 'BarrierAPI',
                       'Transaction')
        for name in alias_names:
            with self.subTest(alias=name):
                self.assertTrue(hasattr(Scenario, name))
                # Same stable class object at class and instance level.
                self.assertIs(getattr(Scenario, name), getattr(scenario, name))
                self.assertIsInstance(getattr(Scenario, name), type)

        # Each primitive-API alias identifies that primitive's api object.
        primitive_checks = (
            (scenario.Lock(), Scenario.LockAPI),
            (scenario.RLock(), Scenario.RLockAPI),
            (scenario.Condition(), Scenario.ConditionAPI),
            (scenario.Event(), Scenario.EventAPI),
            (scenario.Semaphore(), Scenario.SemaphoreAPI),
            (scenario.BoundedSemaphore(), Scenario.BoundedSemaphoreAPI),
            (scenario.Barrier(2), Scenario.BarrierAPI),
        )
        for primitive, alias in primitive_checks:
            with self.subTest(alias=alias.__name__):
                self.assertIsInstance(scenario.api(primitive), alias)

    def test_transaction_alias_identifies_tx_objects(self):
        # Scenario.Transaction (the merged TransactionAPI) catches every
        # transaction the user can hold, regardless of underlying tx
        # class.
        scenario = Scenario()
        lock = scenario.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with scenario:
            t = scenario.thread(worker)
            scenario.wait(Call(t, lock.acquire, State.BLOCKED))
            tx = scenario.transactions[t]
            self.assertIsInstance(tx, Scenario.Transaction)
            self.assertIsInstance(tx, scenario.Transaction)
            scenario.api(lock).unblock(lock.acquire, t)
            scenario.wait(tx)
            scenario.skip(t, lock.release)

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

    def test_event_fancy_reprs_raw_methods_and_transaction_reprs(self):
        s = Scenario()
        event = s.Event()
        event.name = 'named-event'
        self.assertIn('EventCore', repr(event._core))
        self.assertIn('EventAPI', repr(s.api(event)))
        raw = s.raw(event)
        self.assertIn('Event.raw', repr(raw))
        self.assertFalse(raw.is_set())
        self.assertFalse(raw.isSet())
        raw.set()
        self.assertTrue(raw.wait(timeout=0))
        raw.clear()
        self.assertFalse(raw.wait(timeout=0))

        methods = [event.is_set, event.isSet, event.set, event.clear, event.wait]

        def worker():
            event.is_set()
            event.isSet()
            event.set()
            event.clear()
            event.wait(timeout=0)

        with s:
            t = s.thread(worker)
            for method in methods:
                s.wait(method)
                tx = s.transaction(t)
                self.assertIn('Event.', repr(tx))
                self.assertIn('Event.', repr(tx._core))
                s.skip(t, method)

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




class TestConditionDelegation(unittest.TestCase):
    """Tests for Condition delegation to underlying lock."""

    def test_condition_locked_availability_matches_current_stdlib(self):
        """Condition.locked() mirrors stdlib availability."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition1 = scenario.Condition(lock)
        condition2 = scenario.Condition(lock)
        available = hasattr(threading.Condition(), 'locked')
        self.assertEqual(hasattr(condition1, 'locked'), available)
        self.assertEqual(hasattr(condition2, 'locked'), available)
        self.assertEqual(hasattr(scenario.raw(condition1), 'locked'), available)

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

    def test_high_level_drivers_accept_condition_lock_aliases(self):
        """skip/park/block/pause accept Condition.acquire as the target.

        Condition.acquire delegates to the underlying lock core, so the
        transaction's normalized method is the lock's acquire.  The
        high-level Driver helpers should still accept the condition method
        spelling the user actually wrote.
        """

        def exercise(verb):
            scenario = Scenario()
            condition = scenario.Condition()
            results = []

            def worker():
                condition.acquire()
                results.append('acquired')
                condition.release()

            with scenario:
                t = scenario.thread(worker)
                if verb == 'skip':
                    scenario.wait(condition.acquire)
                    txs = scenario.skip(t, condition.acquire)
                    self.assertIn(t, txs)
                elif verb == 'park':
                    txs = scenario.park(t, condition.acquire)
                    tx = txs[t]
                    self.assertEqual(tx.state, State.BLOCKED)
                    tx.unblock()
                    scenario.wait(tx)
                elif verb == 'block':
                    txs = scenario.block(t, condition.acquire)
                    tx = txs[t]
                    self.assertEqual(tx.state, State.BLOCKED)
                    tx.unblock()
                    scenario.wait(tx)
                elif verb == 'pause':
                    txs = scenario.pause(t, condition.acquire)
                    tx = txs[t]
                    self.assertEqual(tx.state, State.PAUSED)
                    tx.unpause()
                    scenario.wait(tx)
                else:
                    raise AssertionError(verb)  # coverage: test-defensive branch

                scenario.wait(condition.release)
                release_txs = scenario.skip(t, condition.release)
                self.assertIn(t, release_txs)

            self.assertEqual(results, ['acquired'])

        for verb in ('skip', 'park', 'block', 'pause'):
            with self.subTest(verb=verb):
                exercise(verb)


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

    def test_semaphore_release_signature_matches_stdlib(self):
        scenario = Scenario()
        stdlib_params = tuple(inspect.signature(threading.Semaphore().release).parameters)
        for sem in (scenario.Semaphore(), scenario.BoundedSemaphore()):
            with self.subTest(primitive=type(sem).__name__):
                raw = scenario.raw(sem)
                self.assertEqual(tuple(inspect.signature(sem.release).parameters), stdlib_params)
                self.assertEqual(tuple(inspect.signature(raw.release).parameters), stdlib_params)

    def test_semaphore_context_manager(self):
        scenario = Scenario()
        sem = scenario.Semaphore(1)

        with sem:
            self.assertFalse(sem.acquire(blocking=False))

        self.assertTrue(sem.acquire(blocking=False))
        sem.release()

    def test_semaphore_fancy_reprs_raw_methods_and_transaction_reprs(self):
        for factory_name in ('Semaphore', 'BoundedSemaphore'):
            with self.subTest(factory=factory_name):
                s = Scenario()
                sem = getattr(s, factory_name)(1)
                sem.name = f'named-{factory_name}'
                self.assertIn('Semaphore', repr(sem._core))
                self.assertIn('SemaphoreAPI', repr(s.api(sem)))
                self.assertIn(factory_name, repr(sem))
                self.assertIn(factory_name + '.raw', repr(s.raw(sem)))
                self.assertEqual(s.api(sem).value, 1)
                self.assertTrue(s.api(sem).available)
                self.assertEqual(s.api(sem).waiters, 0)

                raw = s.raw(sem)
                self.assertTrue(raw.acquire(blocking=False))
                raw.release()

                def worker():
                    sem.acquire()
                    sem.release()

                with s:
                    t = s.thread(worker)
                    for method in (sem.acquire, sem.release):
                        s.wait(method)
                        tx = s.transaction(t)
                        self.assertIn('Semaphore.', repr(tx))
                        self.assertIn('Semaphore.', repr(tx._core))
                        s.skip(t, method)

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
            # allocate drives one semaphore transaction at a time,
            # but the workers' post-commit Python (the results.append
            # calls) races outside blanket's synchronization.  Wait
            # for each worker to actually terminate so the ordering
            # assertion below is deterministic.

        self.assertIn(('A', True), results)
        self.assertIn(('B', True), results)
        self.assertLess(results.index(('A', True)), results.index(('B', True)))

    def test_allocate_reports_nonblocking_failed_acquire(self):
        scenario = Scenario()
        sem = scenario.Semaphore(0)
        sem_api = scenario.api(sem)
        results = []

        def acquirer():
            results.append(sem.acquire(blocking=False))

        with scenario:
            a = scenario.thread(acquirer)
            with self.assertRaisesRegex(RuntimeError, 'timed out|failed'):
                list(sem_api.allocate(a))

        self.assertEqual(results, [False])

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

        self.assertEqual(results, ['overrelease'])

    def test_bounded_semaphore_allocate_acquire_makes_room_for_release(self):
        scenario = Scenario()
        sem = scenario.BoundedSemaphore(1)
        sem_api = scenario.api(sem)
        results = []

        def releaser():
            sem.release()
            results.append('released')

        def acquirer():
            results.append(sem.acquire())

        with scenario:
            r = scenario.thread(releaser)
            a = scenario.thread(acquirer)
            self.assertEqual(list(sem_api.allocate(r, a)), [a])

        self.assertIn(True, results)
        self.assertIn('released', results)
        self.assertEqual(sem_api.value, 1)



class TestVersionSpecificStdlibFidelity(unittest.TestCase):
    """Tests for import-time stdlib-shape gates.

    These load a fresh copy of blanket.primitives while temporarily making
    current-stdlib modules look like older Python versions.  They don't prove
    those old interpreters work end-to-end, but they do keep blanket's
    stdlib-exact API gates from quietly regressing on the only interpreter in
    this test run.
    """

    def test_missing_simplequeue_is_not_impersonated(self):
        with _temporarily_removed_attribute(queue, 'SimpleQueue'):
            with _fresh_primitives_module() as mod:
                scenario = mod.Scenario()

                self.assertFalse(mod._queue_provides_simplequeue)
                self.assertFalse(hasattr(mod.Scenario, 'SimpleQueue'))
                self.assertFalse(hasattr(mod.Scenario, 'RawSimpleQueue'))
                self.assertFalse(hasattr(mod.Scenario, 'SimpleQueueAPI'))
                self.assertFalse(hasattr(scenario, 'SimpleQueue'))
                self.assertFalse(hasattr(scenario.raws, 'SimpleQueue'))
                self.assertFalse(hasattr(scenario.queue, 'SimpleQueue'))

    def test_missing_queue_shutdown_is_not_impersonated(self):
        with _temporarily_removed_attribute(queue.Queue, 'shutdown'):
            with _temporarily_removed_attribute(queue, 'ShutDown'):
                with _fresh_primitives_module() as mod:
                    scenario = mod.Scenario()

                    self.assertFalse(mod._queue_provides_shutdown)
                    self.assertFalse(hasattr(scenario.queue, 'ShutDown'))

                    for name in ('Queue', 'LifoQueue', 'PriorityQueue'):
                        with self.subTest(variant=name):
                            q = getattr(scenario, name)()
                            self.assertFalse(hasattr(q, 'shutdown'))
                            self.assertFalse(hasattr(scenario.raw(q), 'shutdown'))
                            self.assertFalse(hasattr(scenario.api(q), 'shutdown'))

    def test_missing_semaphore_release_n_uses_old_signature(self):
        original_sem_release = threading.Semaphore.release
        original_bounded_release = threading.BoundedSemaphore.release

        def semaphore_release(self):
            return original_sem_release(self)

        def bounded_semaphore_release(self):
            return original_bounded_release(self)

        with _temporarily_assigned_attribute(threading.Semaphore, 'release', semaphore_release):
            with _temporarily_assigned_attribute(threading.BoundedSemaphore, 'release', bounded_semaphore_release):
                with _fresh_primitives_module() as mod:
                    scenario = mod.Scenario()

                    self.assertFalse(mod._semaphore_release_accepts_n)
                    for factory_name in ('Semaphore', 'BoundedSemaphore'):
                        with self.subTest(primitive=factory_name):
                            sem = getattr(scenario, factory_name)(2)
                            raw = scenario.raw(sem)

                            self.assertEqual(tuple(inspect.signature(sem.release).parameters), ())
                            self.assertEqual(tuple(inspect.signature(raw.release).parameters), ())

                            with self.assertRaises(TypeError):
                                sem.release(2)
                            with self.assertRaises(TypeError):
                                raw.release(2)

                            sem.acquire()
                            sem.release()



    def test_rlock_locked_is_impersonated_when_stdlib_has_it(self):
        original_rlock = threading.RLock

        class RLockWithLocked:
            def __init__(self, *args, **kwargs):
                self._actual = original_rlock(*args, **kwargs)

            def acquire(self, *args, **kwargs):
                return self._actual.acquire(*args, **kwargs)

            def release(self):
                return self._actual.release()

            def _is_owned(self):
                return self._actual._is_owned()

            def _release_save(self):
                return self._actual._release_save()

            def _acquire_restore(self, state):
                return self._actual._acquire_restore(state)

            if hasattr(original_rlock(), '_recursion_count'):  # coverage: version-specific branch
                def _recursion_count(self):
                    return self._actual._recursion_count()

            def locked(self):
                # This fake is only used from the owning thread in this
                # test; that is enough to verify blanket exposes and
                # routes the stdlib method when the stdlib has one.
                return self._actual._is_owned()

            def __repr__(self):
                return repr(self._actual)

        fake_rlock = RLockWithLocked()
        self.assertFalse(fake_rlock._is_owned())
        fake_rlock.acquire()
        self.assertTrue(fake_rlock._is_owned())
        if hasattr(fake_rlock, '_recursion_count'):
            self.assertEqual(fake_rlock._recursion_count(), 1)
        state = fake_rlock._release_save()
        self.assertFalse(fake_rlock._is_owned())
        fake_rlock._acquire_restore(state)
        self.assertTrue(fake_rlock._is_owned())
        self.assertIn('RLock', repr(fake_rlock))
        fake_rlock.release()

        with _temporarily_assigned_attribute(threading, 'RLock', RLockWithLocked):
            with _fresh_primitives_module() as mod:
                scenario = mod.Scenario()
                rlock = scenario.RLock()
                raw = scenario.raw(rlock)

                self.assertTrue(mod._rlock_provides_locked)
                self.assertTrue(hasattr(rlock, 'locked'))
                self.assertTrue(hasattr(raw, 'locked'))

                self.assertFalse(rlock.locked())
                rlock.acquire()
                self.assertTrue(rlock.locked())
                self.assertTrue(raw.locked())
                rlock.release()
                self.assertFalse(raw.locked())

    def test_condition_locked_is_impersonated_when_stdlib_has_it(self):
        original_condition = threading.Condition

        class ConditionWithLocked(original_condition):
            def locked(self):
                lock = self._lock
                if hasattr(lock, 'locked'):
                    return lock.locked()
                return lock._is_owned()

        class FakeLockWithoutLocked:
            def __init__(self, owned):
                self.owned = owned
            def _is_owned(self):
                return self.owned
        class FakeLockWithLocked:
            def __init__(self, locked):
                self._locked = locked
            def locked(self):
                return self._locked

        fake_condition = object.__new__(ConditionWithLocked)
        fake_condition._lock = FakeLockWithLocked(True)
        self.assertTrue(fake_condition.locked())
        fake_condition._lock = FakeLockWithLocked(False)
        self.assertFalse(fake_condition.locked())
        fake_condition._lock = FakeLockWithoutLocked(True)
        self.assertTrue(fake_condition.locked())
        fake_condition._lock = FakeLockWithoutLocked(False)
        self.assertFalse(fake_condition.locked())

        with _temporarily_assigned_attribute(threading, 'Condition', ConditionWithLocked):
            with _fresh_primitives_module() as mod:
                scenario = mod.Scenario()
                lock = scenario.Lock()
                condition = scenario.Condition(lock)
                raw = scenario.raw(condition)

                self.assertTrue(mod._condition_provides_locked)
                self.assertTrue(hasattr(condition, 'locked'))
                self.assertTrue(hasattr(raw, 'locked'))

                self.assertFalse(condition.locked())
                lock.acquire()
                self.assertTrue(condition.locked())
                self.assertTrue(raw.locked())
                lock.release()
                self.assertFalse(raw.locked())




class TestQueueFamily(unittest.TestCase):
    """Tests for the Queue / LifoQueue / PriorityQueue family."""

    VARIANTS = ('Queue', 'LifoQueue', 'PriorityQueue')

    def each(self):
        s = Scenario()
        for name in self.VARIANTS:
            yield name, s, getattr(s, name)

    def test_each_iterates_variants(self):
        items = list(self.each())
        self.assertEqual([name for name, _s, _factory in items], list(self.VARIANTS))
        self.assertTrue(all(callable(factory) for _name, _s, factory in items))

    def test_construction_and_masquerade(self):
        s = Scenario()
        for name, real in (('Queue', queue.Queue),
                           ('LifoQueue', queue.LifoQueue),
                           ('PriorityQueue', queue.PriorityQueue)):
            with self.subTest(variant=name):
                q = getattr(s, name)(maxsize=5)
                self.assertIsInstance(q, real)
                self.assertEqual(q.maxsize, 5)
                self.assertTrue(q.empty())

    def test_mutex_and_conditions_swapped_onto_blanket_lock(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                c = getattr(s, name)()._core
                raw_lock = c.underlying_lock._core.raw
                self.assertIs(c.actual.mutex, raw_lock)
                self.assertIs(c.actual.not_empty._lock, raw_lock)
                self.assertIs(c.actual.not_full._lock, raw_lock)
                self.assertIs(c.actual.all_tasks_done._lock, raw_lock)

    def test_nonblocking_put_get_qsize_empty_full(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)(maxsize=2)
                self.assertTrue(q.empty())
                q.put_nowait(1)
                q.put_nowait(2)
                self.assertEqual(q.qsize(), 2)
                self.assertTrue(q.full())
                with self.assertRaises(queue.Full):
                    q.put_nowait(3)
                q.get_nowait()
                self.assertFalse(q.full())
                q.get_nowait()  # drain the last one
                self.assertTrue(q.empty())
                with self.assertRaises(queue.Empty):
                    q.get_nowait()

    def test_ordering_inherited(self):
        s = Scenario()
        lq = s.LifoQueue()
        lq.put_nowait(1); lq.put_nowait(2)
        self.assertEqual([lq.get_nowait(), lq.get_nowait()], [2, 1])
        pq = s.PriorityQueue()
        pq.put_nowait(3); pq.put_nowait(1); pq.put_nowait(2)
        self.assertEqual([pq.get_nowait(), pq.get_nowait(), pq.get_nowait()], [1, 2, 3])

    def test_task_done_overcall_raises(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)()
                q.put_nowait('x')
                q.task_done()  # valid
                with self.assertRaises(ValueError):
                    q.task_done()  # over-call

    def test_shutdown_matches_stdlib_availability(self):
        s = Scenario()
        stdlib_has_shutdown = hasattr(queue.Queue, 'shutdown')
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)()
                raw = s.raw(q)
                api = s.api(q)
                self.assertEqual(hasattr(q, 'shutdown'), stdlib_has_shutdown)
                self.assertEqual(hasattr(raw, 'shutdown'), stdlib_has_shutdown)
                self.assertEqual(hasattr(api, 'shutdown'), stdlib_has_shutdown)

    def test_raw_handle(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)()
                raw = s.raws[q]
                raw.put_nowait('z')
                self.assertEqual(raw.qsize(), 1)
                self.assertEqual(raw.get_nowait(), 'z')

    def test_api_aliases(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)()
                self.assertIsInstance(s.api(q), getattr(Scenario, name + 'API'))

    def test_queue_family_fancy_reprs(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)()
                q.name = f"named-{name}"
                self.assertIn(name, repr(q))
                self.assertIn(name, repr(q._core))
                self.assertIn(name + 'API', repr(s.api(q)))
                self.assertIn(name + '.raw', repr(s.raw(q)))

    def test_raw_queue_family_methods(self):
        s = Scenario()
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                q = getattr(s, name)(maxsize=2)
                raw = s.raw(q)
                raw.put(1)
                self.assertEqual(raw.qsize(), 1)
                self.assertFalse(raw.empty())
                self.assertFalse(raw.full())
                self.assertEqual(raw.get(), 1)
                raw.task_done()
                raw.join()
                raw.put_nowait(2)
                self.assertEqual(raw.get_nowait(), 2)
                raw.task_done()
                if hasattr(raw, 'shutdown'):
                    raw.shutdown()

    def test_queue_transaction_reprs(self):
        s = Scenario()
        q = s.Queue(maxsize=3)
        methods = [
            q.put_nowait,
            q.qsize,
            q.empty,
            q.full,
            q.get_nowait,
            q.task_done,
            q.put,
            q.get,
            q.task_done,
            q.join,
        ]
        if hasattr(q, 'shutdown'):
            methods.append(q.shutdown)

        def worker():
            q.put_nowait(1)
            q.qsize()
            q.empty()
            q.full()
            q.get_nowait()
            q.task_done()
            q.put(2)
            q.get()
            q.task_done()
            q.join()
            if hasattr(q, 'shutdown'):
                q.shutdown()

        with s:
            t = s.thread(worker)
            for method in methods:
                s.wait(method)
                tx = s.transaction(t)
                self.assertIn('Queue.', repr(tx))
                self.assertIn('Queue.', repr(tx._core))
                s.skip(t, method)

    def test_inject_covers_queue_family(self):
        s = Scenario()
        target = types.ModuleType('qtarget')
        target.Queue = queue.Queue
        target.LifoQueue = queue.LifoQueue
        target.PriorityQueue = queue.PriorityQueue
        with s.inject(target):
            self.assertIs(target.Queue, s.Queue)
            self.assertIs(target.LifoQueue, s.LifoQueue)
            self.assertIs(target.PriorityQueue, s.PriorityQueue)
        self.assertIs(target.Queue, queue.Queue)

    # ---- blocking get / put / join ----
    #
    # The three Conditions are plain native Conditions over one shimmed
    # (but unregulated) raw blanket Lock -- the same shape as Event.
    # get/put are WaitingTransactions: the tx itself parks at WAITING via
    # the raw lock's release-save shim while native-blocked in actual.X (no
    # child wait tx).  join is a plain Transaction (it can't time out) and
    # parks at COMMIT as an opaque native block.  Driving is by transaction:
    # park the blocker, then drive the waker to completion and wait on the
    # blocker's *transaction* (waiting on the thread would block forever,
    # since a finished thread stops signalling).

    def test_blocking_get_woken_by_put(self):
        s = Scenario(); q = s.Queue(); out = []
        def getter(): out.append(q.get())
        def putter(): q.put('x')
        with s:
            tg = s.thread(getter)
            s.wait(q.get)
            txg = s.transaction(tg)
            txg.unblock()
            s.wait(Call(tg, q.get, State.WAITING))
            self.assertEqual(txg.state, State.WAITING)
            tp = s.thread(putter)
            s.wait(q.put)
            txp = s.transaction(tp)
            txp.unblock()
            s.wait(txp)
            s.wait(txg)
        self.assertEqual(out, ['x'])

    def test_blocking_put_woken_by_get(self):
        # maxsize=1, pre-filled: the put parks at WAITING until a get
        # frees a slot.
        s = Scenario(); q = s.Queue(maxsize=1); q.put_nowait('a'); out = []
        def putter(): q.put('b')
        def getter(): out.append(q.get())
        with s:
            tp = s.thread(putter)
            s.wait(q.put)
            txp = s.transaction(tp)
            txp.unblock()
            s.wait(Call(tp, q.put, State.WAITING))
            self.assertEqual(txp.state, State.WAITING)
            tg = s.thread(getter)
            s.wait(q.get)
            txg = s.transaction(tg)
            txg.unblock()
            s.wait(txg)
            s.wait(txp)
        self.assertEqual(out, ['a'])
        self.assertEqual(q.qsize(), 1)        # 'b' is now queued

    def test_blocking_join_woken_by_task_done(self):
        s = Scenario(); q = s.Queue(); q.put_nowait('a'); done = []
        def joiner():
            q.join()
            done.append('joined')
        def worker():
            q.get()
            q.task_done()
        with s:
            tj = s.thread(joiner)
            s.wait(q.join)
            txj = s.transaction(tj)
            txj.unblock()
            s.wait(Call(tj, q.join, State.COMMIT))
            self.assertEqual(txj.state, State.COMMIT)
            tw = s.thread(worker)
            s.wait(q.get)
            txg = s.transaction(tw)
            txg.unblock()
            s.wait(txg)
            s.wait(q.task_done)
            txt = s.transaction(tw)
            txt.unblock()
            s.wait(txt)
            s.wait(txj)
        self.assertEqual(done, ['joined'])

    def test_deliver_empty_queue_reorders_get_after_put(self):
        s = Scenario(); q = s.Queue(); out = []
        def getter(): out.append(q.get())
        def putter(): q.put('x')
        with s:
            tg = s.thread(getter)
            tp = s.thread(putter)
            txs = s.api(q).deliver(tg, tp)
        self.assertEqual(out, ['x'])
        self.assertEqual([tx.method for tx in txs], [q.get, q.put])
        self.assertTrue(all(tx.state is State.RETURNED for tx in txs))

    def test_deliver_full_queue_reorders_put_after_get(self):
        s = Scenario(); q = s.Queue(maxsize=1); q.put_nowait('a'); out = []
        def putter(): q.put('b')
        def getter(): out.append(q.get())
        with s:
            tp = s.thread(putter)
            tg = s.thread(getter)
            txs = s.api(q).deliver(tp, tg)
        self.assertEqual(out, ['a'])
        self.assertEqual(q.get_nowait(), 'b')
        self.assertEqual([tx.method for tx in txs], [q.put, q.get])

    def test_deliver_preserves_order_when_operations_can_run(self):
        s = Scenario(); q = s.LifoQueue(); q.put_nowait('old'); out = []
        def putter(): q.put('new')
        def getter(): out.append(q.get())
        with s:
            tp = s.thread(putter)
            tg = s.thread(getter)
            s.api(q).deliver(tp, tg)
        self.assertEqual(out, ['new'])
        self.assertEqual(q.get_nowait(), 'old')

    def test_deliver_supports_nowait_methods(self):
        s = Scenario(); q = s.Queue(); out = []
        def getter(): out.append(q.get_nowait())
        def putter(): q.put_nowait('x')
        with s:
            tg = s.thread(getter)
            tp = s.thread(putter)
            txs = s.api(q).deliver(tg, tp)
        self.assertEqual(out, ['x'])
        self.assertEqual([tx.method for tx in txs], [q.get_nowait, q.put_nowait])

    def test_deliver_supports_duplicate_thread(self):
        s = Scenario(); q = s.Queue(); out = []
        def worker():
            q.put('x')
            out.append(q.get())
        with s:
            t = s.thread(worker)
            txs = s.api(q).deliver(t, t)
        self.assertEqual(out, ['x'])
        self.assertEqual([tx.method for tx in txs], [q.put, q.get])

    def test_deliver_rejects_non_traffic_queue_methods(self):
        s = Scenario(); q = s.Queue(); q.put_nowait('x')
        def worker(): q.task_done()
        with s:
            t = s.thread(worker)
            with self.assertRaises(ValueError):
                s.api(q).deliver(t)

    def test_deliver_propagates_nowait_failure_when_no_call_can_run(self):
        s = Scenario(); q = s.Queue(); errors = []
        def getter():
            try:
                q.get_nowait()
            except queue.Empty:
                errors.append('empty')
        with s:
            t = s.thread(getter)
            with self.assertRaises(queue.Empty):
                s.api(q).deliver(t)
        self.assertEqual(errors, ['empty'])

    def test_blocking_get_woken_by_put_each_variant(self):
        # the blocking machinery is shared across the family; confirm a
        # parked getter is woken for each variant (ints sort for the
        # PriorityQueue case).
        for name in self.VARIANTS:
            with self.subTest(variant=name):
                s = Scenario(); q = getattr(s, name)(); out = []
                def getter(): out.append(q.get())
                def putter(): q.put(7)
                with s:
                    tg = s.thread(getter)
                    s.wait(q.get)
                    txg = s.transaction(tg)
                    txg.unblock()
                    s.wait(Call(tg, q.get, State.WAITING))
                    tp = s.thread(putter)
                    s.wait(q.put)
                    txp = s.transaction(tp)
                    txp.unblock()
                    s.wait(txp)
                    s.wait(txg)
                self.assertEqual(out, [7])


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

    def test_barrier_fancy_reprs_raw_methods_and_transaction_reprs(self):
        s = Scenario()
        barrier = s.Barrier(1)
        barrier.name = 'named-barrier'
        self.assertIn('BarrierCore', repr(barrier._core))
        self.assertIn('BarrierAPI', repr(s.api(barrier)))
        raw = s.raw(barrier)
        self.assertIn('Barrier.raw', repr(raw))
        self.assertEqual(raw.wait(timeout=0), 0)
        raw.reset()
        raw.abort()
        self.assertTrue(s.api(barrier).broken)
        raw.reset()

        methods = [barrier.wait, barrier.reset, barrier.abort]

        def worker():
            barrier.wait(timeout=0)
            barrier.reset()
            barrier.abort()

        with s:
            t = s.thread(worker)
            for method in methods:
                s.wait(method)
                tx = s.transaction(t)
                self.assertIn('Barrier.', repr(tx))
                self.assertIn('Barrier.', repr(tx._core))
                self.assertEqual(tx.method.__name__, method.__name__)
                s.skip(t, method)

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
                result.append(('success', n))  # coverage: defensive success branch
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

    def test_barrier_cycle_without_action_context_manager(self):
        """Barrier.cycle works when the Barrier has no action and no
        scheduler is supplied."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)
        api = scenario.api(barrier)
        results = []

        def worker(name):
            def fn():
                results.append(f'{name}_before')
                index = barrier.wait()
                results.append((name, index))
            return fn

        with scenario:
            a = scenario.thread(worker('A'))
            b = scenario.thread(worker('B'))
            c = scenario.thread(worker('C'))

            with api.cycle(a, b, c) as cycle:
                self.assertEqual(cycle.ready, (a, b, c))
                self.assertEqual(cycle.extra_waiters, 0)
                self.assertEqual(
                    [scenario.transaction(t).state for t in (a, b, c)],
                    [State.PAUSED, State.PAUSED, State.PAUSED])
                self.assertEqual(results, [
                    'A_before', 'B_before', 'C_before'])

            self.assertTrue(cycle.closed)
            self.assertEqual(cycle.ready, ())

        self.assertEqual(results[:3], [
            'A_before', 'B_before', 'C_before'])
        self.assertEqual(set(results[3:]), {
            ('A', 0), ('B', 1), ('C', 2)})

    def test_barrier_cycle_without_action_accepts_waiting_waiters(self):
        """Barrier.cycle also works when earlier parties have
        already reached the underlying Barrier's WAITING state."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)
        api = scenario.api(barrier)
        results = []

        def worker(name):
            def fn():
                results.append(f'{name}_before')
                index = barrier.wait()
                results.append((name, index))
            return fn

        with scenario:
            a = scenario.thread(worker('A'))
            b = scenario.thread(worker('B'))
            c = scenario.thread(worker('C'))

            for thread in (a, b):
                scenario.wait(thread)
                tx = scenario.transaction(thread)
                tx.unblock()
                scenario.wait(Waiting(tx))

            scenario.wait(c)
            self.assertEqual(api.waiters, 2)
            self.assertEqual(api.raw.n_waiting, 2)
            self.assertEqual(results, [
                'A_before', 'B_before', 'C_before'])

            cycle = api.cycle(a, b, c)
            self.assertEqual(cycle.ready, (a, b, c))
            self.assertEqual(api.waiters, 0)
            self.assertEqual(
                [scenario.transaction(t).state for t in (a, b, c)],
                [State.PAUSED, State.PAUSED, State.PAUSED])
            self.assertEqual(results, [
                'A_before', 'B_before', 'C_before'])
            self.assertEqual(cycle.close(), (a, b, c))

        self.assertEqual(results, [
            'A_before', 'B_before', 'C_before',
            ('A', 0), ('B', 1), ('C', 2),
        ])

    def test_barrier_reset(self):
        """Barrier.reset() resets barrier and breaks waiting transactions."""
        scenario = Scenario()
        barrier = scenario.Barrier(3)  # Need 3 parties
        api = scenario.api(barrier)
        results = []

        def waiter():
            try:
                barrier.wait(timeout=NEVER)
                results.append('wait_success')  # coverage: defensive success branch
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
                results.append('wait_success')  # coverage: defensive success branch
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
                results.append(('success', result))  # coverage: defensive success branch
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

            scenario.skip(A, rlock.acquire)
            scenario.wait(A)
            a_wait = scenario.transaction(A)
            a_wait.unblock()
            scenario.wait(Waiting(a_wait))
            self.assertEqual(results, ['waiter_acquired'])

            B = scenario.thread(worker_B)
            B.name = 'B'
            scenario.skip(B, rlock.acquire)
            scenario.wait(B)

            cycle = cond_api.cycle(A, B)
            self.assertFalse(hasattr(cond_api, 'choose'))
            self.assertFalse(hasattr(cond_api, 'finish'))
            self.assertEqual(cycle.waiters, (A,))

            # The notifier still holds the UL after notify.  Release it,
            # then wake the waiter from the cycle object.
            scenario.skip(B, rlock.release)
            self.assertEqual(cycle.wake(), A)
            scenario.skip(A, rlock.release)

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
        """API unblock can be sequenced explicitly one transaction at a time.

        This is a regression test verifying that a scheduler can unblock one
        transaction, observe it finish, then unblock the next one, rather than
        being forced to batch all observers, all unblocks, and all waits.

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

    Driver.paused() (and the other parking imperatives) stage their
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
            d.scan(); d()
            self.assertIs(tx.state, State.BLOCKED)
            d.paused()
            self.assertIs(tx.state, State.BLOCKED)
            disp = s.Dispatch()
            disp.add(d)
            self.assertIs(tx.state, State.BLOCKED)
            disp.remove(d)
            self.assertIs(tx.state, State.BLOCKED)
            # Re-add and iterate: now the staged lazy fires.
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIs(tx.state, State.PAUSED)

        # --- Phase 2: Driver via Chain into Dispatch. ---
        s, lock, worker = make()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            d.scan(); d()
            self.assertIs(tx.state, State.BLOCKED)
            d.paused()
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
            self.assertIs(yielded.state, d.success)
            self.assertIs(tx.state, State.PAUSED)


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
        d.scan()
        d()
        return lock, t, d

    def _cleanup(self, s, *triples):
        """Release the worker threads.  Each triple is (lock, thread,
        driver).  Drivers not already in a terminal state get
        close()'d to release their score-slot; then each thread's
        lock.acquire is unblocked so it can exit.
        """
        for lock, t, d in triples:
            if Terminated(t) not in d.motivation:  # coverage: test-defensive branch
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
            d.paused()
            disp = s.Dispatch()
            disp.add(chain)
            next(iter(disp))
            # After the yielded driver leaves the active set,
            # advance_chain_after clears current; pending was already
            # empty; chain is falsy again.
            self.assertFalse(chain)

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
            d.paused()
            chain = s.Chain(d)
            disp = s.Dispatch()
            disp.add(chain)
            self.assertEqual(chain.pending, (d,))
            yielded = next(iter(disp))
            self.assertIs(yielded, d)
            # After the yielded driver is handed to the user,
            # advance_chain_after has cleared current.
            self.assertEqual(chain.pending, ())

    def test_two_drivers_serialized_through_dispatch(self):
        # A chain of two drivers iterated through a Dispatch yields
        # them in pending-order: d1 first, then d2 after d1 reaches
        # terminal.
        s = Scenario()
        with s:
            lock1, t1, d1 = self._make_thread(s)
            lock2, t2, d2 = self._make_thread(s)
            d1.paused()
            d2.paused()
            chain = s.Chain(d1, d2)
            disp = s.Dispatch()
            disp.add(chain)
            it = iter(disp)
            first = next(it)
            self.assertIs(first, d1)
            self.assertIs(first.state, d1.success)
            second = next(it)
            self.assertIs(second, d2)
            self.assertIs(second.state, d2.success)


class TestModuleHelpersAndImportBranches(unittest.TestCase):
    def test_do_nothing_sentinel_is_callable(self):
        self.assertIsNone(primitives_module._do_nothing(None))

    def test_state_helpers_and_scenario_properties(self):
        s = Scenario()
        self.assertEqual(s.name, '')
        self.assertIs(s.apis, s._core.apis_proxy)
        self.assertIs(s.raws, s._core.raws_proxy)
        self.assertIs(s.log, s._core.log_proxy)
        self.assertEqual(list(s.log), s._core.log)
        with self.assertRaises(AttributeError):
            s.log.append(None)
        self.assertIs(s.managed, s._core.managed_proxy)
        self.assertIsNone(s.transaction(threading.current_thread()))
        self.assertEqual(State.BLOCKED.index, 100)
        self.assertEqual(State.BLOCKED.name, 'BLOCKED')

    def test_scenario_wait_requires_items(self):
        s = Scenario()
        with self.assertRaises(ValueError):
            s.wait()

    def test_transaction_wait_state(self):
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lambda: lock.acquire())
            s.wait(Call(t, lock.acquire, State.BLOCKED))
            tx = s.transaction(t)
            self.assertIs(tx.wait(State.BLOCKED), State.BLOCKED)
            with self.assertRaises(TypeError):
                tx.wait("BLOCKED")
            api = s.api(lock)
            api.unblock(lock.acquire, t)
            self.assertIs(tx.wait(), State.RETURNED)

    def test_transaction_visited_queries_state_log(self):
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lambda: lock.acquire())
            s.wait(Call(t, lock.acquire, State.BLOCKED))
            tx = s.transaction(t)
            self.assertTrue(tx.visited(State.BLOCKED))
            self.assertFalse(tx.visited(State.PAUSED))
            with self.assertRaises(ValueError):
                tx.visited()
            with self.assertRaises(TypeError):
                tx.visited("BLOCKED")

            d = s.Driver(t)
            d.scan(); d()
            d.paused(); d()
            self.assertTrue(tx.visited(State.BLOCKED, State.PAUSED))
            self.assertFalse(tx.visited(State.BLOCKED, State.RETURNED))
            d.finish(); d()

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

    def test_context_manager_repr_reports_entered_state(self):
        cm = self.core.ContextManager()
        self.assertIn("not entered", repr(cm))
        with self.scenario:
            self.assertIn("entered", repr(self.scenario._context_manager))

    def test_reset_clears_terminated_tx_debris_from_signaling(self):
        """reset() drops accumulated state.  Self-reporting tx and
        Signaling items report their own state directly; the per-score
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
            self.assertTrue(tx_api.sample(scenario))
            scenario.reset()
            self.assertTrue(tx_api.sample(scenario))
            self.assertEqual(len(self.core.log), 0)
            self.assertTrue(Terminated(t).sample(scenario))

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
        self.assertEqual(len(self.core.log), 0)
        self.assertEqual(len(self.core.waiters), 0)

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
            self.assertTrue(tx_api.sample(scenario))
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
        # tx.unpause is meaningful only for a PAUSED transaction with
        # the scheduler's pause bit set.  The old strict-state RuntimeError
        # path is gone, so there's nothing to assert here.
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
            _ = tx.kwargs; _ = tx.pause; _ = tx.pausing
            tx.pause = True
            self.assertTrue(tx.pause)
            self.assertTrue(tx.pausing)
            tx.pause = False
            self.assertFalse(tx.pause)
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
        # tx.api.unpause is permissive: it clears the scheduler pause bit,
        # and is a no-op when that bit was already clear.
        tx.api.unpause()


    def test_non_timeout_transaction_api_timeout_and_acquire_repr(self):
        def worker():
            self.lock.acquire(); self.lock.release()
        with self.scenario:
            t = self.scenario.thread(worker)
            self.scenario.wait(t)
            tx = self.scenario.transaction(t)
            self.assertIn('Lock.acquire', repr(tx._core))
            self.scenario.skip(t, self.lock.acquire)
            self.scenario.wait(t)
            rtx = self.scenario.transaction(t)
            self.assertIsNone(rtx.timeout.value)

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
            except RuntimeError:  # coverage: thread-defensive cleanup
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
        self.assertEqual(FakeActual()._recursion_count(), 0)
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
        if hasattr(actual, 'locked'):  # coverage: version-specific branch
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
            s.skip(holder, lock.acquire)
            # Waiter blocks on lock.acquire with a long timeout.
            result = []
            t = s.thread(lambda: result.append(lock.acquire(timeout=NEVER)))
            s.wait(lock.acquire, t)
            # Expire the waiter's acquire: settings-only, just marks
            # _timeout=0.
            api.expire(lock.acquire, t)
            # Drive past commit; actual.acquire(True, 0) returns
            # False because the lock is held.
            s.skip(t, lock.acquire)
            # Release the held lock so scenario can exit cleanly.
            s.raw(lock).release()
        self.assertEqual(result, [False])

    def test_transfer_release_raised(self):
        s = Scenario(); lock = s.Lock(); api = s.api(lock)
        def bad_releaser():
            try:
                lock.release()
            except RuntimeError:  # coverage: thread-defensive cleanup
                pass
        def acquirer():
            lock.acquire()
        with s:
            r = s.thread(bad_releaser)
            a = s.thread(acquirer)
            with self.assertRaises(RuntimeError):
                api.assign(r, a)
            api.unblock(lock.acquire, a)


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

    def test_relay_waits_for_later_arrivals_lazily(self):
        """relay is progressive, not a snapshot: it must not wait for a
        later chain participant to reach its tx until the chain advances
        to it.  Here C is gated behind an event and has NOT called
        lock.acquire() when relay() is invoked; relay drives B then A,
        and only once we advance the iterator past A does it wait for C
        (after we release the gate).  A snapshot relay would fail at
        call time because C isn't parked at acquire/BLOCKED."""
        s = Scenario(); lock = s.Lock(); api = s.api(lock); order = []
        gate = threading.Event()
        def a(): lock.acquire(); order.append('A'); lock.release()
        def b(): lock.acquire(); order.append('B'); lock.release()
        def c():
            gate.wait()
            lock.acquire(); order.append('C'); lock.release()
        with s:
            ta = s.thread(a); tb = s.thread(b); tc = s.thread(c)
            api.assign(tb)                      # B holds the lock, at release/BLOCKED
            it = api.relay(tb, ta, tc)
            self.assertIs(next(it), ta)         # drives B->release, A->acquire
            # C has not been waited for yet: it is still gated, not on a tx.
            self.assertNotIn(tc, s.transactions)
            gate.set()                          # now let C head toward acquire
            self.assertIs(next(it), tc)         # relay waits for C, then drives it
            with self.assertRaises(StopIteration):
                next(it)
            api.unblock(lock.release, tc)
        self.assertEqual(order, ['B', 'A', 'C'])

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
            m = Call(t, lock.acquire)
            self.assertIsInstance(m, Call)
            self.assertIsInstance(m, tuple)
            self.assertEqual(m.thread, t)
            self.assertEqual(m.method, lock.acquire)
            self.assertIsNone(m.state)
            m2 = Call(t, lock.acquire)
            self.assertEqual(m, m2)

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

    def test_not_normalized_shorthand_forms_stay_native(self):
        """Not(thread/method/primitive) normalizes to native objects, not boxes."""
        s = Scenario()
        lock = s.Lock()
        raw = s.raws[lock]

        t = threading.Thread(target=lambda: None)
        not_thread = Not(t)
        self.assertIs(not_thread.normalized(), not_thread)
        self.assertIs(not_thread.normalized().wrapped, t)

        not_lock = Not(lock)
        self.assertIs(not_lock.normalized(), not_lock)
        self.assertIs(not_lock.normalized().wrapped, lock)
        self.assertIs(Not(raw).normalized().wrapped, lock)

        not_acquire = Not(lock.acquire)
        self.assertIs(not_acquire.normalized(), not_acquire)
        lock_acquire = not_acquire.normalized().wrapped
        raw_acquire = Not(raw.acquire).normalized().wrapped
        self.assertIs(lock_acquire.__self__, lock)
        self.assertIs(raw_acquire.__self__, lock)
        self.assertEqual(lock_acquire.__name__, 'acquire')
        self.assertEqual(raw_acquire.__name__, 'acquire')

        for signal in (Not(t), Not(lock), Not(raw), Not(lock.acquire), Not(raw.acquire)):
            with self.subTest(signal=signal):
                wrapped = signal.normalized().wrapped
                self.assertNotIsInstance(wrapped, primitives_module.Signaling)

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

    def _wait_until_parked(self, scenario, key, message):
        deadline = time.perf_counter() + 1
        while True:
            with scenario._core.lock:
                parked = bool(scenario._core.waiters.get(key))
            if parked:
                return
            if time.perf_counter() > deadline:
                self.fail(message)
            time.sleep(0.001)

    def _assert_not_shorthand_wakes_on_close(self, signal_factory):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            signal = signal_factory(t, lock)
            self.assertFalse(s.wait(signal, timeout=0))

            results = []
            def wait_for_signal():
                results.append(s.wait(signal, timeout=NEVER))

            waiter = threading.Thread(target=wait_for_signal)
            waiter.start()
            self._wait_until_parked(
                s, signal.normalized(),
                f"{signal!r} waiter did not park")

            s.skip(t, lock.acquire)

            waiter.join(5)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(results, [{signal}])
            return True

    def test_not_thread_wait_wakes_when_thread_goes_idle(self):
        self.assertTrue(self._assert_not_shorthand_wakes_on_close(lambda t, lock: Not(t)))

    def test_not_thread_bearing_signals_reject_unstarted_thread(self):
        s = Scenario()
        lock = s.Lock()
        t = threading.Thread(target=lambda: None)
        signals = [
            Not(t),
            Not(Terminated(t)),
            Not(Call(t, lock.acquire)),
            Not(Use(t, lock)),
        ]
        for signal in signals:
            with self.subTest(signal=signal):
                with self.assertRaises(ValueError) as cm:
                    s.wait(signal, timeout=0)
                self.assertIn("unstarted thread", str(cm.exception))



    def test_not_bound_method_wait_wakes_when_method_goes_low(self):
        self.assertTrue(self._assert_not_shorthand_wakes_on_close(lambda t, lock: Not(lock.acquire)))

    def test_not_primitive_wait_wakes_when_primitive_goes_low(self):
        self.assertTrue(self._assert_not_shorthand_wakes_on_close(lambda t, lock: Not(lock)))

    def test_not_transaction_state_wait_wakes_when_state_changes(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            signal = Not(Blocked(tx))
            self.assertFalse(s.wait(signal, timeout=0))

            results = []
            def wait_for_signal():
                results.append(s.wait(signal, timeout=NEVER))

            waiter = threading.Thread(target=wait_for_signal)
            waiter.start()
            self._wait_until_parked(
                s, signal.normalized(),
                "Not(Blocked(tx)) waiter did not park")

            tx.unblock()

            waiter.join(5)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(results, [{signal}])
            s.wait(tx)

    def test_not_nested_wait_wakes_when_child_transaction_closes(self):
        s = Scenario()
        rlock = s.RLock()
        condition = s.Condition(rlock)

        def predicate():
            return condition.wait_for(lambda: True, timeout=-1)

        def worker():
            rlock.acquire()
            condition.wait_for(predicate, timeout=-1)
            rlock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, rlock.acquire)
            s.wait(Call(t, condition.wait_for, State.BLOCKED))
            parent = s.transaction(t)
            parent.unblock()
            s.wait(Call((t, parent), condition.wait_for, State.BLOCKED))
            child = s.transaction(t)

            signal = Not(Nested(parent))
            self.assertFalse(s.wait(signal, timeout=0))

            results = []
            def wait_for_signal():
                results.append(s.wait(signal, timeout=NEVER))

            waiter = threading.Thread(target=wait_for_signal)
            waiter.start()
            self._wait_until_parked(
                s, signal.normalized(),
                "Not(Nested(tx)) waiter did not park")

            child.unblock()

            waiter.join(5)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(results, [{signal}])
            s.wait(parent)
            s.skip(t, rlock.release)

class TestScenarioWaitAll(unittest.TestCase):

    def test_wait_all_accumulates_under_one_wait(self):
        s = Scenario()
        a_lock = s.Lock()
        b_lock = s.Lock()

        def a():
            a_lock.acquire()

        def b():
            b_lock.acquire()

        with s:
            a_thread = s.thread(a)
            b_thread = s.thread(b)
            signaled = s.wait(a_thread, b_thread, all=True)
            self.assertIsInstance(signaled, frozenset)
            self.assertEqual(signaled, frozenset({a_thread, b_thread}))
            self.assertEqual(
                s.wait(a_thread, b_thread, all=True, timeout=0),
                frozenset({a_thread, b_thread}))
            s.skip(a_thread, a_lock.acquire)
            s.skip(b_thread, b_lock.acquire)

    def test_wait_all_timeout_returns_partial_set(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            terminated = Terminated(t)
            signaled = s.wait(t, terminated, all=True, timeout=IMMEDIATELY)
            self.assertIsInstance(signaled, frozenset)
            self.assertEqual(signaled, frozenset({t}))
            self.assertEqual(
                s.wait(t, terminated, all=True, timeout=0),
                frozenset({t}))
            s.skip(t, lock.acquire)

    def test_wait_all_waits_without_timeout_for_remaining_items(self):
        s = Scenario()
        first_lock = s.Lock()
        second_lock = s.Lock()
        gate = threading.Event()

        def first():
            first_lock.acquire()

        def second():
            gate.wait()
            second_lock.acquire()

        with s:
            first_thread = s.thread(first)
            second_thread = s.thread(second)
            s.wait(first_thread)
            timer = threading.Timer(0.01, gate.set)
            timer.start()
            try:
                self.assertEqual(
                    s.wait(first_thread, second_thread, all=True),
                    frozenset({first_thread, second_thread}))
            finally:
                timer.cancel()
            s.skip(first_thread, first_lock.acquire)
            s.skip(second_thread, second_lock.acquire)


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

    def test_block_without_enter_raises(self):
        s = Scenario()
        lock = s.Lock()
        t = threading.Thread(target=lambda: None)
        with self.assertRaises(RuntimeError) as cm:
            s.block(t, lock.acquire)
        self.assertIn("scenario not entered", str(cm.exception))


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

    def test_signaling_base_sample_must_be_overridden(self):
        signal = primitives_module.Signaling()
        with self.assertRaises(NotImplementedError):
            signal.sample(Scenario())

    def test_transaction_signal_token_thread_property(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            nested = Nested(tx)
            self.assertIs(nested.thread, tx.thread)
        finally:
            self._finish_tx(s, tx)

    def test_use_rejects_non_primitive(self):
        t = threading.Thread(target=lambda: None)
        with self.assertRaises(TypeError) as cm:
            Use(t, object())
        self.assertIn("primitive", str(cm.exception))

    def test_use_scoped_properties_and_repr(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            use = Use((tx.thread, tx), lock)
            self.assertIs(use.thread, tx.thread)
            self.assertIs(use.base_tx, tx)
            self.assertIs(use.primitive, lock)
            self.assertEqual(use.thread_scope, (tx.thread, tx))
            self.assertIn(tx.thread.name, repr(use))
            self.assertIn("Use(", repr(use))
        finally:
            self._finish_tx(s, tx)

    def test_call_rejects_unregulated_callable(self):
        t = threading.Thread(target=lambda: None)
        with self.assertRaises(TypeError) as cm:
            Call(t, lambda: None)
        self.assertIn("bound method", str(cm.exception))

    def test_call_scoped_properties_and_repr(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            call = Call((tx.thread, tx), lock.acquire, State.BLOCKED)
            self.assertIs(call.thread, tx.thread)
            self.assertIs(call.base_tx, tx)
            self.assertIs(call.method.__self__, lock)
            self.assertIs(call.state, State.BLOCKED)
            self.assertEqual(call.thread_scope, (tx.thread, tx))
            self.assertIn(tx.thread.name, repr(call))
            self.assertIn("BLOCKED", repr(call))

            class FakeTx:
                def __init__(self, parent, methods=()):
                    self.parent = parent
                    self._methods = set(methods)
                def call_methods(self):
                    return self._methods

            middle = FakeTx(tx._core)
            child = FakeTx(middle, {lock.acquire})
            self.assertTrue(Call._scoped_tx_selected(child, tx._core, lock.acquire))
        finally:
            self._finish_tx(s, tx)

    def test_reached_thread_property(self):
        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            self.assertIs(Reached(tx, State.BLOCKED).thread, tx.thread)
        finally:
            self._finish_tx(s, tx)

    def test_action_and_predicate_validation_and_repr(self):
        with self.assertRaises(TypeError) as cm:
            Action("not a tx")
        self.assertIn("Transaction", str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            Predicate("not a tx")
        self.assertIn("Transaction", str(cm.exception))

        s = Scenario()
        lock = s.Lock()
        tx = self._make_tx(s, lock)
        try:
            self.assertTrue(repr(Action(tx)).startswith("Action("))
            self.assertTrue(repr(Predicate(tx)).startswith("Predicate("))
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

    def test_park_skips_over_nonmatching(self):
        """park steps over (drives to terminal) any tx that isn't the
        named method, until it appears.  Here the worker calls acquire
        then release; park(release) drives past acquire and parks the
        release at BLOCKED."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            r = s.park(t, lock.release)
            self.assertEqual(r[t].method, lock.release)
            self.assertEqual(r[t].state, State.BLOCKED)
            s.skip(t, lock.release)

    def test_park_skip_raises_on_missing_method(self):
        """park raises if the worker terminates before reaching the method."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Drive acquire to completion so thread is about to terminate.
            s.skip(t, lock.acquire)
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

    def test_skip_drives_past(self):
        """skip drives the named call to terminal, so the worker has
        progressed past it when skip returns."""
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
            r = s.skip(t, lock.acquire)
            self.assertTrue(r[t].done)
            # After the skip, worker has progressed past lock.acquire.
            # Drive lock.release to let worker finish.
            s.skip(t, lock.release)
        self.assertEqual(order, ['acq', 'rel'])

    def test_skip_concurrent_across_threads(self):
        """skip drives all named threads concurrently; threads whose
        targets depend on each other must not deadlock."""
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
            s.skip(B, lock.acquire)
            # Now A is blocked on lock.acquire (actual lock held by B).
            s.wait(Call(A, lock.acquire, State.BLOCKED))
            # skip(A, lock.acquire, B, lock.release): A's target
            # is lock.acquire and can only complete after B releases.
            # Sequential per-thread drive would deadlock; concurrent must not.
            r = s.skip(A, lock.acquire, B, lock.release)
            self.assertTrue(r[A].done)
            self.assertTrue(r[B].done)
        self.assertEqual(order, ['B', 'A'])

    def test_park_after_completed_skip_tolerates_inflight_tx(self):
        """park after skip tolerates whatever transient state the worker
        is in mid-tx-pop."""
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
            s.skip(t, lock.acquire, lock.release)
            # park() scans for and selects the next acquire transaction.
            r = s.park(t, lock.acquire)
            self.assertEqual(r[t].method, lock.acquire)
            r[t].unblock(); s.wait(r[t])
            s.skip(t, lock.release)

    def test_park_skips_over_blocked_nonmatching(self):
        """park steps over a non-matching tx even when the thread is
        already parked (BLOCKED) on it: it drives that tx to terminal
        and keeps looking for the named method."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            # worker parked at acquire BLOCKED; park asks for release,
            # which is further along -- park drives over acquire.
            s.wait(Call(t, lock.acquire, State.BLOCKED))
            r = s.park(t, lock.release)
            self.assertEqual(r[t].method, lock.release)
            self.assertEqual(r[t].state, State.BLOCKED)
            s.skip(t, lock.release)


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
        """skip drives several threads in one call, concurrently, even
        when they contend for the same lock.  (Arg order groups a
        thread with its methods; it does not impose an A-before-B
        ordering -- the threads are driven in parallel, so each must be
        able to make progress, e.g. both release the lock here.)"""
        s = Scenario()
        lock = s.Lock()
        def worker_a():
            lock.acquire()
            lock.release()
        def worker_b():
            lock.acquire()
            lock.release()
        with s:
            A = s.thread(worker_a)
            B = s.thread(worker_b)
            r = s.skip(A, lock.acquire, lock.release,
                       B, lock.acquire, lock.release)
            self.assertEqual(r[A].method, lock.release)
            self.assertEqual(r[B].method, lock.release)
            self.assertTrue(r[A].done)
            self.assertTrue(r[B].done)

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
                       t, lock.release)
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
            s.skip(t, lock.acquire)

    def test_skip_raises_on_missing_method(self):
        """If the worker terminates before reaching the method, raise."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            # Wait for the worker to reach and enter lock.acquire.
            s.skip(t, lock.acquire)  # unblock the acquire
            # Now thread is post-acquire, about to terminate.
            with self.assertRaises(RuntimeError) as cm:
                s.skip(t, lock.release)  # never called
            self.assertIn('terminated', str(cm.exception))

    def test_skip_does_not_deadlock_across_threads(self):
        """skip drives every named thread CONCURRENTLY, so a thread
        whose target can't complete until another is driven won't
        deadlock.

        Setup: B acquires the lock first.  A then calls lock.acquire,
        which blocks on the actual underlying lock.  We then call
        skip(A, lock.acquire, B, lock.release).  A serial,
        one-thread-at-a-time drive would deadlock (driving A.acquire to
        completion waits on B's release, but B isn't driven yet).
        Driving both concurrently lets B.release free the lock so
        A.acquire can complete."""
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
            s.skip(B, lock.acquire)
            # Now A calls lock.acquire and will be BLOCKED at the
            # scheduler level.  Wait for A to reach BLOCKED.
            s.wait(Call(A, lock.acquire, State.BLOCKED))
            # Drive both concurrently: unblock A.acquire (A's commit
            # will then sit waiting on the actual lock until B releases)
            # AND drive B.release.  Both must complete.
            r = s.skip(A, lock.acquire,
                       B, lock.release)
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
            s.skip(t, lock.acquire, lock.release)
            # By the time skip returns, the worker has completed the
            # first pair and is heading to the next lock.acquire.
            s.skip(t, lock.acquire, lock.release)
            self.assertEqual(s.wait(Terminated(t), timeout=IMMEDIATELY), {Terminated(t)})



class TestConditionCycleBasic(unittest.TestCase):

    def park_waiter(self, scenario, condition, lock, thread):
        scenario.skip(thread, lock.acquire)
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
            scenario.skip(x, lock.acquire)
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
            scenario.skip(a, lock.release)

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
            scenario.skip(x, lock.acquire)
            scenario.wait(x)

            with capi.cycle(a, b, x) as cycle:
                scenario.skip(x, lock.release)
                self.assertEqual(cycle.wake(b), (b,))
                scenario.skip(b, lock.release)
            # __exit__ closed the cycle and woke A.
            scenario.skip(a, lock.release)

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
            scenario.skip(n, lock.acquire)
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
            scenario.skip(b, lock.release)

        self.assertEqual(log, ['C', 'A', 'B'])

    def test_cycle_immediate_success_among_waiters(self):
        """Comprehensive immediate-success: a wait_for whose predicate is
        true on its first check (A) sits mid-spec among four plain
        cond.wait waiters (B, C, D, E) plus a notify_all waker (F).

        A never waits -- it parks at PAUSED holding UL -- so the resumable
        processor surfaces it as ready straight out of construction (Stage
        1 returns the moment it's reached).  Waking A lets it run on; the
        rest are then driven through F's notify_all by close(), which
        relays each thread's lock.release as the next is woken and leaves
        only the final thread parked at its release for us to drive.  All
        six threads complete, waiters in spec order, A first."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter(name, predicate=None):
            def fn():
                lock.acquire()
                condition.wait() if predicate is None else condition.wait_for(predicate)
                log.append(name)
                lock.release()
            return fn

        def notifier():
            lock.acquire(); condition.notify_all(); lock.release()

        with scenario:
            b = scenario.thread(waiter('B'))
            c = scenario.thread(waiter('C'))
            d = scenario.thread(waiter('D'))
            e = scenario.thread(waiter('E'))
            a = scenario.thread(waiter('A', lambda: True))  # immediate success
            f = scenario.thread(notifier)
            # Park the plain waiters first: each acquires UL, waits (which
            # releases UL), so the next can acquire.  Only then can A
            # acquire the now-free UL and sit at its wait_for -- an
            # immediate-success waiter holds UL until woken, so at most one
            # can be pending at construction.
            for t in (b, c, d, e):
                self.park_waiter(scenario, condition, lock, t)
            scenario.skip(a, lock.acquire)
            scenario.wait(a)

            # A is mid-spec; it is nonetheless ready immediately.
            cycle = capi.cycle(b, a, c, d, e, f)
            self.assertEqual(cycle.ready, (a,))

            self.assertEqual(cycle.wake(a), (a,))
            # close() drives B, C, D, E through F's notify_all in spec
            # order, relaying releases; the last is left at its release.
            woke = cycle.close()
            self.assertEqual(woke, (b, c, d, e))
            self.assertTrue(cycle.closed)
            scenario.skip(woke[-1], lock.release)

        self.assertEqual(log, ['A', 'B', 'C', 'D', 'E'])

    def test_cycle_wait_rejects_plain_waiter_and_preserves_ready(self):
        """wait() is only valid for wait_for waiters that can re-wait.

        A plain cond.wait waiter should remain ready after the error so
        the caller can still wake or pause it.
        """
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter():
            lock.acquire(); condition.wait(); log.append('A'); lock.release()

        def notifier():
            lock.acquire(); condition.notify(); lock.release()

        with scenario:
            a = scenario.thread(waiter)
            n = scenario.thread(notifier)
            self.park_waiter(scenario, condition, lock, a)
            cycle = capi.cycle(a, n)
            self.assertEqual(cycle.ready, (a,))

            with self.assertRaisesRegex(ValueError, "plain cond.wait"):
                cycle.wait(a)
            self.assertEqual(cycle.ready, (a,))
            self.assertFalse(cycle.closed)
            self.assertEqual(log, [])

            scenario.skip(n, lock.release)
            self.assertEqual(cycle.wake(a), (a,))
            scenario.skip(a, lock.release)

        self.assertEqual(log, ['A'])


    def test_cycle_pause_plain_waiter(self):
        """pause() on a plain cond.wait waiter parks it at PAUSED (user
        pause flag) instead of running it on; unpause() resumes it."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter():
            lock.acquire(); condition.wait(); log.append('A'); lock.release()

        def notifier():
            lock.acquire(); condition.notify_all(); lock.release()

        with scenario:
            a = scenario.thread(waiter)
            n = scenario.thread(notifier)
            self.park_waiter(scenario, condition, lock, a)
            cycle = capi.cycle(a, n)
            self.assertEqual(cycle.ready, (a,))
            self.assertEqual(cycle.pause(a), (a,))
            tx = scenario.transaction(a)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(log, [])
            tx.unpause()
            scenario.wait(tx)
            scenario.skip(a, lock.release)

        self.assertEqual(log, ['A'])

    def test_cycle_pause_immediate_success_waiter(self):
        """pause() on an immediate-success wait_for (parked at PAUSED
        holding UL) hands the cycle's blanket pause bit to the scheduler's
        pause bit; the thread stays at PAUSED until unpause()."""
        scenario = Scenario()
        lock = scenario.Lock()
        condition = scenario.Condition(lock)
        capi = scenario.api(condition)
        log = []

        def waiter():
            lock.acquire(); condition.wait_for(lambda: True); log.append('A'); lock.release()

        def notifier():
            lock.acquire(); condition.notify(); lock.release()

        with scenario:
            a = scenario.thread(waiter)
            n = scenario.thread(notifier)
            scenario.skip(a, lock.acquire)
            scenario.wait(a)
            cycle = capi.cycle(a, n)
            self.assertEqual(cycle.ready, (a,))
            self.assertEqual(cycle.pause(a), (a,))
            tx = scenario.transaction(a)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(log, [])
            tx.unpause()
            scenario.wait(tx)
            scenario.skip(a, lock.release)
            cycle.close()   # drain the notifier (its notify is a no-op)

        self.assertEqual(log, ['A'])

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
            scenario.skip(a, lock.release)

        self.assertEqual(log, ['A_acq', 'N_acq', 'N_notified', 'N_rel', 'A_woke', 'A_rel'])


class TestCycleIteratorProtocol(unittest.TestCase):
    """The cycle object is itself an iterator: __iter__ returns self,
    __next__ wakes the next remaining waiter (spec order) and yields
    its thread, raising StopIteration on empty remaining or closed."""

    def park_waiter(self, scenario, condition, lock, thread):
        scenario.skip(thread, lock.acquire)
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
            s.skip(n, lock.acquire)
            s.wait(n)
            cy = capi.cycle(a, n)
            s.skip(n, lock.release)
            self.assertIs(iter(cy), cy)
            list(cy)  # drain so scenario exits cleanly
            s.skip(a, lock.release)

    def test_for_loop_drains_in_spec_order(self):
        # The standard with-for idiom: drain all remaining waiters
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
            s.skip(n, lock.acquire)
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
            s.skip(n, lock.acquire)
            s.wait(n)
            cy = capi.cycle(a, b, n)
            s.skip(n, lock.release)
            self.assertIs(next(cy), a)
            s.skip(a, lock.release)
            self.assertIs(next(cy), b)
            s.skip(b, lock.release)
            with self.assertRaises(StopIteration):
                next(cy)

    def test_stopiteration_on_drained(self):
        # After draining via iteration, further next() raises StopIteration.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            s.skip(n, lock.acquire)
            s.wait(n)
            cy = capi.cycle(a, n)
            s.skip(n, lock.release)
            list(cy)
            s.skip(a, lock.release)
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
            s.skip(n, lock.acquire)
            s.wait(n)
            cy = capi.cycle(a, b, n)
            s.skip(n, lock.release)
            self.assertIs(next(cy), a)
            s.skip(a, lock.release)
            # close() drains b; lock is free so b's cond.wait reacquires.
            cy.close()
            self.assertTrue(cy.closed)
            s.skip(b, lock.release)
            with self.assertRaises(StopIteration):
                next(cy)

    def test_iter_in_with_block(self):
        # Standard idiom: with-as cycle, for-in drain.
        s, lock, cond, capi, log, waiter, notifier = self._setup()
        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            n = s.thread(notifier)
            self.park_waiter(s, cond, lock, a)
            self.park_waiter(s, cond, lock, b)
            s.skip(n, lock.acquire)
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
            with self.assertRaisesRegex(ValueError, 'specified more than once'):
                capi.cycle(t, t)
            s.skip(t, lock.acquire, lock.release)

    def test_cycle_notify_fewer_than_waiters_raises(self):
        """notify(n) with MORE cycle waiters than n is a spec error: the
        notify can't wake them all.  (notify(n) with FEWER waiters than n
        is fine -- the surplus is a no-op; see the immediate-success
        tests.)  The check is in drive_waker, so it fires at construction.

        The scenario it guards is genuinely deadlocked -- a never-notified
        cond.wait can't be unblocked -- so teardown needs a helper
        notify_all to release the stranded waiter."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        log = []

        def waiter(name):
            def fn():
                lock.acquire(); condition.wait(); log.append(name); lock.release()
            return fn

        def notifier():
            lock.acquire(); condition.notify(1); lock.release()

        def relief():  # teardown helper: frees whichever waiter notify(1) couldn't
            lock.acquire(); condition.notify_all(); lock.release()

        with s:
            b = s.thread(waiter('B'))
            c = s.thread(waiter('C'))
            f = s.thread(notifier)
            g = s.thread(relief)
            for t_ in (b, c):
                s.skip(t_, lock.acquire); s.wait(t_)
                wtx = s.transaction(t_); wtx.unblock(); s.wait(Waiting(wtx))
            with self.assertRaisesRegex(
                    ValueError,
                    r'holding 2 waiters but notify\(1\)'):
                capi.cycle(b, c, f)
            # Teardown: fire the notify(1) (wakes one), then the helper's
            # notify_all (wakes the other), draining both.
            s.skip(f, condition.notify, lock.release)
            s.skip(b, condition.wait, lock.release)
            s.skip(g, lock.acquire, condition.notify_all, lock.release)
            s.skip(c, condition.wait, lock.release)

        self.assertEqual(sorted(log), ['B', 'C'])

    def test_cycle_notify_count_excludes_immediate_success(self):
        """The waiter-count check is late-bound: a wait_for whose predicate
        passes on the first try never becomes a waiter, so it doesn't count
        against notify(n).  Here notify(1) is fine even though the cycle
        names two would-be waiters, because one of them (A) succeeds
        immediately and the only genuine waiter is B."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        log = []

        def plain():
            lock.acquire(); condition.wait(); log.append('B'); lock.release()

        def immediate():
            lock.acquire(); condition.wait_for(lambda: True); log.append('A'); lock.release()

        def notifier():
            lock.acquire(); condition.notify(1); lock.release()

        with s:
            b = s.thread(plain)
            a = s.thread(immediate)
            f = s.thread(notifier)
            s.skip(b, lock.acquire); s.wait(b)        # B genuinely waits
            btx = s.transaction(b); btx.unblock(); s.wait(Waiting(btx))
            s.skip(a, lock.acquire)                   # A sits at its wait_for
            s.wait(a)
            # A is excluded from the count, so 1 waiter == notify(1): no raise.
            cycle = capi.cycle(a, b, f)
            self.assertEqual(cycle.ready, (a,))
            self.assertEqual(cycle.wake(a), (a,))
            woke = cycle.close()
            self.assertEqual(woke, (b,))
            s.skip(woke[-1], lock.release)

        self.assertEqual(log, ['A', 'B'])

    def test_cycle_wait_for_first_predicate_success(self):
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
            s.skip(w, lock.acquire)
            s.wait(w)
            # Immediate predicate success: the wait_for returns without
            # ever calling wait.  The cycle parks the waiter at PAUSED
            # (holding UL) and makes it ready -- no longer an error.
            cycle = capi.cycle(w, n)
            self.assertEqual(cycle.ready, (w,))
            self.assertEqual(cycle.wake(w), (w,))
            s.skip(w, lock.release)
            # The notifier acquires and notifies into a now-empty wait
            # set (the waiter already left): a no-op.  Drive it directly.
            s.skip(n, lock.acquire, condition.notify, lock.release)


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
            s.skip(t, lock.acquire)
            s.wait(t)
            tx = s.transaction(t)
            self.assertEqual(tx.timeout.value, -1)
            tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            tx.unstall()
            s.wait(tx)
            self.assertTrue(tx.timeout.timed_out)
            s.skip(t, lock.release)

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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)

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
            s.skip(w, lock.acquire)
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
            s.skip(n, lock.acquire)
            s.wait(n)

            cycle = capi.cycle(w, n)
            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(), w)

            # Drain waiter's lock.release.
            s.skip(w, lock.release)

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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)

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
            s.skip(n, lock.acquire)
            s.wait(n)
            s.transaction(n).unblock()
            s.skip(n, lock.release)
            s.wait(Call(t, condition.wait, State.STALLED))
            child.unstall()
            s.wait(child)
            return child

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)


class TestConditionCycleScheduler(unittest.TestCase):
    """Condition.cycle(scheduler=) drives wait_for waiters through
    their predicate.  wait_for sits at COMMIT and emits Predicate,
    then (Nested, Predicate)*; the cycle drives the single inner
    wait.  When the predicate runs the Driver yields REENTERED, the
    cycle calls scheduler(wait_for_tx) so the caller can drive any
    regulated tx the predicate spawns, then the one inner cond.wait
    is driven to WAITING."""

    def test_scheduler_called_with_wait_for_tx(self):
        """The scheduler is handed the wait_for tx, and the waiter
        ends up parked at the inner cond.wait WAITING even though it
        was not pre-driven past its predicate."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        results = iter([False, True])
        seen = []

        def predicate():
            return next(results)

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        def scheduler(wf):
            seen.append(wf)

        with s:
            w = s.thread(waiter)
            # Waiter holds UL, blocked at cond.wait_for (NOT pre-driven
            # past the predicate -- the scheduler= path does that).
            s.skip(w, lock.acquire)
            s.wait(w)
            # Notifier blocked at UL.acquire (UL held by waiter); the
            # cycle frees it in Phase 4 after the waiter releases UL.
            n = s.thread(notifier)
            s.wait(n)

            cycle = capi.cycle(w, n, scheduler=scheduler)

            # Scheduler was invoked exactly once, with the wait_for tx.
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].method, condition.wait_for)
            # Waiter parked at the inner cond.wait WAITING.
            self.assertEqual(s.transaction(w).method, condition.wait)
            self.assertEqual(s.transaction(w).state, State.STALLED)

            # Wake: inner wait returns, predicate re-checks True,
            # wait_for exits.
            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(), w)
            s.skip(w, lock.release)

    def test_scheduler_tx_disambiguates_multiple_wait_for_waiters(self):
        """scheduler(wait_for_tx) identifies which waiter reentered.

        A single Condition.cycle may manage multiple wait_for callers;
        the tx argument lets the scheduler distinguish them by tx.thread.
        """
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        predicate_calls = {}
        seen = []
        log = []

        def predicate():
            thread = threading.current_thread()
            predicate_calls[thread] = predicate_calls.get(thread, 0) + 1
            return predicate_calls[thread] >= 2

        def waiter(name):
            def fn():
                lock.acquire()
                condition.wait_for(predicate)
                log.append(name)
                lock.release()
            return fn

        def notifier():
            lock.acquire()
            condition.notify_all()
            lock.release()

        def scheduler(wf):
            seen.append((wf.thread, wf.method))

        with s:
            a = s.thread(waiter('A')); a.name = 'A'
            b = s.thread(waiter('B')); b.name = 'B'
            n = s.thread(notifier); n.name = 'N'

            s.skip(a, lock.acquire)
            s.wait(a)      # A is at wait_for, holding UL.
            s.wait(b)      # B is at UL.acquire.
            s.wait(n)      # N is also waiting for UL.

            cycle = capi.cycle(a, b, n, scheduler=scheduler)
            ready = cycle.ready

            s.skip(n, lock.release)
            woke_a = cycle.wake(a)
            s.skip(a, lock.release)
            woke_b = cycle.wake(b)
            s.skip(b, lock.release)

        self.assertEqual(ready, (a, b))
        self.assertEqual(woke_a, (a,))
        self.assertEqual(woke_b, (b,))
        self.assertEqual([thread for thread, _ in seen], [a, b])
        self.assertEqual([method for _, method in seen], [condition.wait_for, condition.wait_for])
        self.assertEqual(log, ['A', 'B'])

    def test_scheduler_drives_predicate_child(self):
        """A predicate that spawns a regulated child: the scheduler
        drives that child to terminal while the predicate runs, then
        the cycle drives the inner cond.wait to WAITING."""
        s = Scenario()
        lock = s.Lock()
        gate = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        results = iter([False, True])
        children = []

        def predicate():
            r = next(results)
            if not r:
                # Spawn a regulated child for the scheduler to drive.
                gate.acquire()
            return r

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        def scheduler(wf):
            # The predicate spawns gate.acquire as a nested child of
            # the wait_for; wait for it, then drive it to terminal.
            s.wait(Nested(wf), Terminated(w))
            child = s.transaction(w)
            children.append(child.method)
            self.assertEqual(child.method, gate.acquire)
            child.unblock()
            s.wait(child)

        with s:
            w = s.thread(waiter)
            s.skip(w, lock.acquire)
            s.wait(w)
            n = s.thread(notifier)
            s.wait(n)

            cycle = capi.cycle(w, n, scheduler=scheduler)

            self.assertEqual(children, [gate.acquire])
            self.assertEqual(s.transaction(w).method, condition.wait)
            self.assertEqual(s.transaction(w).state, State.STALLED)

            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(), w)
            s.skip(w, lock.release)

    def test_scheduler_predicate_succeeds_immediately(self):
        """If the predicate returns truthy on its first check, the
        wait_for exits without ever calling wait.  The cycle parks that
        waiter at PAUSED and makes it ready (it never waited, so the
        notifier's notify lands in an empty wait set -- driven here
        directly rather than through the cycle)."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: True)
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        scheduler_calls = []
        scheduler = scheduler_calls.append

        with s:
            w = s.thread(waiter)
            s.skip(w, lock.acquire)
            s.wait(w)
            n = s.thread(notifier)
            s.wait(n)

            cycle = capi.cycle(w, n, scheduler=scheduler)
            self.assertEqual(cycle.ready, (w,))
            with self.assertRaisesRegex(ValueError, "already succeeded"):
                cycle.wait(w)
            self.assertEqual(cycle.ready, (w,))
            self.assertEqual(cycle.wake(w), (w,))
            s.skip(w, lock.release)
            s.skip(n, lock.acquire, condition.notify, lock.release)

    def test_cycle_wait_wait_for_waiter_rewaits(self):
        """wait() on a wait_for waiter re-runs a false predicate and parks
        the waiter at a fresh inner cond.wait, ready for a later cycle."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        log = []
        results = iter([False, False, True])
        seen = []

        def predicate():
            seen.append('predicate')
            return next(results)

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            log.append('A')
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        with s:
            w = s.thread(waiter)
            n1 = s.thread(notifier)
            s.skip(w, lock.acquire)
            s.wait(w)
            s.wait(n1)

            cycle = capi.cycle(w, n1)
            self.assertEqual(s.transaction(w).state, State.STALLED)

            self.assertEqual(cycle.wait(), w)
            self.assertTrue(cycle.closed)
            self.assertEqual(s.transaction(w).method, condition.wait)
            self.assertEqual(s.transaction(w).state, State.WAITING)
            self.assertEqual(log, [])

            # A later cycle can now notify the fresh wait.  The third
            # predicate call succeeds and the waiter exits wait_for.
            n2 = s.thread(notifier)
            s.wait(n2)
            cycle2 = capi.cycle(w, n2)
            s.skip(n2, lock.release)
            self.assertEqual(cycle2.wake(w), (w,))
            s.skip(w, lock.release)

        self.assertEqual(log, ['A'])
        self.assertEqual(seen, ['predicate', 'predicate', 'predicate'])

    def test_cycle_wait_exiting_wait_for_preserves_relay(self):
        """If wait() expected a re-wait but the predicate succeeds,
        the cycle raises while preserving enough relay state to drain
        later waiters."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        results = {
            'A': iter([False, True]),
            'B': iter([False, True]),
        }
        log = []

        def predicate(name):
            return next(results[name])

        def waiter(name):
            def fn():
                lock.acquire()
                condition.wait_for(lambda: predicate(name))
                log.append(f'{name}_after')
                lock.release()
            return fn

        def notifier():
            lock.acquire()
            condition.notify_all()
            lock.release()

        with s:
            a = s.thread(waiter('A'))
            b = s.thread(waiter('B'))
            n = s.thread(notifier)
            s.skip(a, lock.acquire)
            s.wait(a)          # A is at wait_for, holding the underlying lock.
            s.wait(b)          # B is blocked at the underlying lock acquire.
            s.wait(n)          # notifier is blocked behind both waiters.

            cycle = capi.cycle(a, b, n)
            s.skip(n, lock.release)

            with self.assertRaisesRegex(RuntimeError, "wait_for exited"):
                cycle.wait(a)
            self.assertEqual(s.transaction(a).method, lock.release)

            # Without the relay repair, this would try to free the
            # underlying lock using the old notifier Driver instead of A.
            self.assertEqual(cycle.wake(b), (b,))
            self.assertEqual(log, ['A_after', 'B_after'])
            s.skip(b, lock.release)


    def test_cycle_pause_wait_for_waiter(self):
        """pause() on a wait_for waiter: the cycle re-runs the predicate
        through the scheduler loop (as for wake) but parks the thread at
        PAUSED with the scheduler pause flag instead of letting it exit.  The
        thread holds at PAUSED until unpause(), then runs its body."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        log = []
        results = iter([False, True])
        seen = []

        def predicate():
            return next(results)

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            log.append('A')
            lock.release()

        def notifier():
            lock.acquire()
            condition.notify()
            lock.release()

        def scheduler(wf):
            seen.append(wf)

        with s:
            w = s.thread(waiter)
            n = s.thread(notifier)
            s.skip(w, lock.acquire)   # w at wait_for, holding UL, predicate unrun
            s.wait(w)
            s.wait(n)                 # n at UL.acquire, blocked
            cycle = capi.cycle(w, n, scheduler=scheduler)
            self.assertEqual(s.transaction(w).state, State.STALLED)
            s.skip(n, lock.release)   # let the notifier reach release for the relay

            self.assertEqual(cycle.pause(w), (w,))
            tx = s.transaction(w)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(log, [])         # body not run while paused
            self.assertEqual(len(seen), 1)    # predicate driven once via scheduler

            tx.unpause()
            s.wait(tx)
            s.skip(w, lock.release)

        self.assertEqual(log, ['A'])


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
            self.assertEqual(s.wait(Terminated(t), timeout=IMMEDIATELY), {Terminated(t)})


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
            self.assertEqual(s.wait(Terminated(t), timeout=IMMEDIATELY), {Terminated(t)})


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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.acquire)
            s.wait(t)
            wait_tx = s.transaction(t)
            self.assertIn('Condition.wait', repr(wait_tx))
            self.assertIn('Condition.wait', repr(wait_tx._core))
            # Drive wait through WAITING (timeout will fire) then STALLED.
            wait_tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            wait_tx.unstall()
            s.wait(wait_tx)
            s.skip(t, lock.release)

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
            s.skip(t1, lock.acquire)
            s.wait(t1)
            notify_tx = s.transaction(t1)
            self.assertIn('Condition.notify', repr(notify_tx))
            self.assertIn('Condition.notify', repr(notify_tx._core))
            self.assertEqual(notify_tx.n, 2)
            self.assertEqual(notify_tx._core.n, 2)
            s.skip(t1, condition.notify, lock.release)

            t2 = s.thread(notifier_all)
            s.skip(t2, lock.acquire)
            s.wait(t2)
            notify_all_tx = s.transaction(t2)
            self.assertIn('Condition.notify_all', repr(notify_all_tx))
            self.assertIn('Condition.notify_all', repr(notify_all_tx._core))
            import math
            self.assertEqual(notify_all_tx._core.n, math.inf)
            s.skip(t2, condition.notify_all, lock.release)

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
            s.skip(t, lock.acquire)
            s.wait(t)
            wait_tx = s.transaction(t)
            wait_tx.unblock()
            # Timeout causes cond.wait to proceed to STALLED.
            s.wait(Call(t, condition.wait, State.STALLED))
            self.assertEqual(wait_tx._core.state, State.STALLED)
            wait_tx.unstall()
            s.wait(wait_tx)  # tx now finishes naturally
            self.assertEqual(wait_tx._core.state, State.RETURNED)
            s.skip(t, lock.release)

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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)

    def test_cond_api_unstall_single_thread(self):
        """cond_api.unstall(thread) works for a single parked thread."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait(timeout=IMMEDIATELY))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            wait_tx = s.transaction(t)
            wait_tx.unblock()
            s.wait(Call(t, condition.wait, State.STALLED))
            capi.unstall(condition.wait, t)
            s.wait(wait_tx)
            s.skip(t, lock.release)
            self.assertEqual(results, [False])

    def test_cond_api_unstall_multi_thread(self):
        """cond_api.unstall(*threads) works for multiple parked threads."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        capi = s.api(condition)
        results = []

        def waiter():
            lock.acquire()
            results.append(condition.wait(timeout=IMMEDIATELY))
            lock.release()

        def park(thread):
            s.skip(thread, lock.acquire)
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
            s.skip(a, lock.release)
            s.wait(Call(b, condition.wait, State.STALLED))
            capi.unstall(condition.wait, b)
            s.wait(b_tx)
            s.skip(b, lock.release)
            self.assertEqual(results, [False, False])

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
            s.skip(t, lock.acquire)

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
            s.skip(t, lock.acquire)
            s.wait(t)
            # State is BLOCKED, not STALLED.
            with self.assertRaisesRegex(ValueError, "STALLED"):
                capi.unstall(condition.wait, t)
            # Drain the wait: expire (settings-only timeout=0) then
            # drive worker through commit, STALLED, and to terminal.
            wait_tx = s.transaction(t)
            wait_tx.expire()
            s.skip(t, condition.wait)


class TestTxUnpark(unittest.TestCase):
    """The public tx.unpark() force cascade was removed.

    Scenario teardown still has an internal unstick path, but ordinary
    users release individual parking states through unblock / unstall /
    unpause or by driving with Driver.
    """

    def test_public_unpark_removed(self):
        s = Scenario()
        lock = s.Lock()

        def w():
            lock.acquire()

        with s:
            t = s.thread(w)
            s.wait(lock.acquire, t)
            tx = s.transaction(t)
            self.assertFalse(hasattr(tx, 'unpark'))


class TestTransactionPauseFlag(unittest.TestCase):

    def test_pause_property_prearms_and_unpause_releases(self):
        s = Scenario()
        lock = s.Lock()
        order = []

        def worker():
            lock.acquire()
            order.append('ran')

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            tx.pause = True
            self.assertTrue(tx.pause)
            self.assertTrue(tx.pausing)
            tx.unblock()
            s.wait(Paused(tx))
            self.assertEqual(tx.state, State.PAUSED)
            tx.unpause()
            s.wait(tx)
            self.assertFalse(tx.pause)
            self.assertFalse(tx.pausing)
            self.assertEqual(order, ['ran'])

    def test_unpause_reports_blanket_pause(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            d = s.Driver(t)
            d.scan(); d()
            tx = d.tx
            d._core.blanket_paused(); d()
            self.assertTrue(tx.pausing)
            self.assertFalse(tx.pause)
            with self.assertRaisesRegex(RuntimeError, 'still paused'):
                tx.unpause()


class TestWaitForInsidePredicate(unittest.TestCase):
    """Tests for the in_predicate flag on cond.wait_for txs."""

    def test_in_predicate_flag_toggles(self):
        """in_predicate is False outside the predicate call, True
        inside it."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        observed = []
        def predicate():
            # Peek at our own tx's in_predicate while running.
            tx = s.transaction(threading.current_thread())
            observed.append(tx._core.in_predicate)
            return True  # terminate immediately

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            wf_tx = s.transaction(t)
            # Outside: False.
            self.assertFalse(wf_tx._core.in_predicate)
            # Drive through.  Predicate True on first call -> no wait.
            wf_tx.unblock()
            s.wait(wf_tx)
            # Predicate was called exactly once.
            self.assertEqual(wf_tx.iterations, 1)
            # During predicate: True observed.
            self.assertEqual(observed, [True])
            # After: False again.
            self.assertFalse(wf_tx._core.in_predicate)
            s.skip(t, lock.release)

    def test_predicate_releasing_ul_does_not_corrupt_parent_state(self):
        """If user's predicate itself releases and unstalls the UL
        (weird but legal), the shim's in_predicate check prevents
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
            s.skip(t, lock.acquire)
            s.wait(t)
            wf_tx = s.transaction(t)
            wf_tx.unblock()
            s.wait(wf_tx)
            # Parent never transitioned through WAITING/STALLED.
            self.assertEqual(wf_tx._core.state, State.RETURNED)
            s.skip(t, lock.release)

    def test_not_predicate_wakes_when_predicate_exits(self):
        s = Scenario()
        lock = s.RLock()
        condition = s.Condition(lock)
        child_lock = s.Lock()

        def predicate():
            child_lock.locked()
            return True

        def waiter():
            lock.acquire()
            condition.wait_for(predicate)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(Call(t, condition.wait_for, State.BLOCKED))
            wf_tx = s.transaction(t)
            wf_tx.unblock()
            s.wait(Predicate(wf_tx))
            s.wait(Call((t, wf_tx), child_lock.locked, State.BLOCKED))

            signal = Not(Predicate(wf_tx))
            self.assertFalse(s.wait(signal, timeout=0))
            results = []

            def wait_for_signal():
                results.append(s.wait(signal, timeout=NEVER))

            parked = threading.Thread(target=wait_for_signal)
            parked.start()
            deadline = time.perf_counter() + 1
            while True:
                with s._core.lock:
                    is_parked = bool(s._core.waiters.get(signal.normalized()))
                if is_parked:
                    break
                if time.perf_counter() > deadline:
                    self.fail("Not(Predicate(tx)) waiter did not park")
                time.sleep(0.001)

            child = s.transaction(t)
            child.unblock()
            parked.join(1)
            self.assertFalse(parked.is_alive())
            self.assertEqual(results, [{signal}])
            s.wait(wf_tx)
            s.skip(t, lock.release)


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
            # We can't easily install a state_observer from outside,
            # so just verify that the tx's final state is terminal.
            s.skip(t, lock.acquire, lock.release)
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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)



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

    def test_wait_positive_timeout_returns_empty_set(self):
        s = Scenario()
        gate = threading.Event()
        t = threading.Thread(target=gate.wait, name='positive-timeout-target')
        t.start()
        try:
            with s:
                start = time.perf_counter()
                self.assertEqual(s.wait(Terminated(t), timeout=IMMEDIATELY), set())
                self.assertGreaterEqual(time.perf_counter() - start, IMMEDIATELY)
                gate.set()
        finally:
            gate.set()
            t.join(1)


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
        self.assertEqual(plan, [(t, None, [lock.acquire])])

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
        self.assertEqual(plan, [(t, None, [lock.acquire, lock.release])])
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
            with self.assertRaisesRegex(TypeError, r"expected thread, \(thread, base_tx\), or method"):
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

    def test_thread_base_tuple_parser(self):
        """Base txs are supplied only as strict (thread, base_tx) tuples."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def worker():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire)
            base = s.transaction(t)

            plan = s._core.parse_park_skip_args(
                ((t, base), lock.release), 'skip')
            self.assertEqual(plan, [(t, base._core, [lock.release])])

            pairs = s._core.parse_thread_base_pairs(((t, base),), 'cycle')
            self.assertEqual(pairs, [(t, base._core)])

            with self.assertRaisesRegex(
                    TypeError, r"expected thread, \(thread, base_tx\), or method"):
                s._core.parse_park_skip_args((t, base, lock.release), 'skip')
            with self.assertRaisesRegex(
                    TypeError, r"expected thread or \(thread, base_tx\)"):
                s._core.parse_thread_base_pairs((t, base), 'cycle')

    def test_thread_base_tuple_parser_rejects_bad_tuples(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def worker():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            other = threading.Thread(target=lambda: None, name='other')

            with self.assertRaisesRegex(TypeError, "2-tuple"):
                s._core.parse_thread_base_pairs(((t, base, object()),), 'cycle')
            with self.assertRaisesRegex(TypeError, "first item"):
                s._core.parse_thread_base_pairs(((object(), base),), 'cycle')
            with self.assertRaisesRegex(TypeError, "second item"):
                s._core.parse_thread_base_pairs(((t, object()),), 'cycle')
            with self.assertRaisesRegex(ValueError, "belongs to thread"):
                s._core.parse_thread_base_pairs(((other, base),), 'cycle')


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
            s.skip(t, lock.acquire)

    def test_transaction_returns_none_when_thread_has_no_active_tx(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.acquire, lock.release)
            # By the time skip returns, lock.release tx has been
            # unblocked + waited.  The thread is now on its way to the
            # next lock.acquire.  A second skip must settle on the
            # *next* fresh acquire, not get confused by the stale state.
            s.skip(t, lock.acquire, lock.release)
            self.assertEqual(s.wait(Terminated(t), timeout=IMMEDIATELY), {Terminated(t)})


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
            # Drain.  The scheduler pause remains owned by the user.
            tx.pause = False

    def test_park_raises_when_method_never_reached(self):
        """park steps over every tx looking for its method; if the
        thread terminates first, it raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        other = s.Lock()

        def worker():
            lock.acquire()
            lock.release()

        with s:
            t = s.thread(worker)
            s.wait(t)
            with self.assertRaisesRegex(RuntimeError, "terminated before reaching"):
                # other.acquire is never called -> park skips to the end.
                s.park(t, other.acquire)



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
            s.skip(t, lock.acquire)
            s.wait(t)
            parent = s.transaction(t)
            parent.unblock()

            s.wait(Call(t, lock.locked, State.BLOCKED))
            child = s.transaction(t)
            self.assertEqual(child.method, lock.locked)
            self.assertIs(child.parent, parent)
            child.unblock(); s.wait(child)

            s.wait(parent)
            s.skip(t, lock.release)



    def test_wait_for_nested_same_primitive_different_method_restores_parent(self):
        """A nested tx on the same primitive but a different method
        uses the outer wait_for as parent, and does not corrupt the
        active parent tx between nested calls."""
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
            s.skip(t, lock.acquire)
            s.wait(t)
            parent = s.transaction(t)
            self.assertEqual(parent.method, condition.wait_for)
            use = Use(t, condition)
            signaled = s.wait(use)
            self.assertIn(use, signaled)

            parent.unblock()

            s.wait(Call(t, condition.notify, State.BLOCKED))
            child1 = s.transaction(t)
            self.assertEqual(child1.method, condition.notify)
            self.assertIs(child1.parent, parent)
            signaled = s.wait(use)
            self.assertIn(use, signaled)
            child1.unblock()
            s.wait(child1)

            s.wait(Call(t, condition.notify, State.BLOCKED))
            child2 = s.transaction(t)
            self.assertEqual(child2.method, condition.notify)
            self.assertIs(child2.parent, parent)
            self.assertIsNot(child2.parent, child1)
            signaled = s.wait(use)
            self.assertIn(use, signaled)
            child2.unblock()
            s.wait(child2)

            s.wait(parent)
            s.skip(t, lock.release)

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
            def drive_child(tx):
                # The scheduler now receives the opener's tx directly.
                self.assertEqual(tx.method, barrier.wait)
                s.wait(Action(tx))
                child = s.transaction(x)
                self.assertEqual(child.method, lock.locked)
                # Under the new (no-implicit-stash) design, the child
                # is a proper child of opener_tx via tx.parent.
                self.assertIs(child.parent, opener_tx)
                self.assertTrue(tx.ran_action)
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

    def test_wait_for_recursive_base_scopes_are_navigable(self):
        """Recursive wait_for calls can be selected with (thread, base_tx)."""
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
            s.skip(t, rlock.acquire)

            txs = []
            scope = t
            for depth in range(5):
                s.wait(Call(scope, condition.wait_for, State.BLOCKED))
                tx = s.transaction(t)
                self.assertEqual(tx.method, condition.wait_for)

                if depth == 0:
                    self.assertIsNone(tx.parent)
                else:
                    self.assertIs(tx.parent, txs[-1])
                    scoped_call = Call((t, txs[-1]), condition.wait_for)
                    self.assertIn(scoped_call, s.wait(scoped_call, timeout=0))

                if depth >= 2:
                    # The current wait_for is shadowed from the grandparent
                    # scope by the same-method child immediately under it.
                    shadowed = Call((t, txs[-2]), condition.wait_for, State.BLOCKED)
                    self.assertFalse(s.wait(shadowed, timeout=0))

                txs.append(tx)
                tx.unblock()
                scope = (t, tx)

            # The innermost predicate returns True; every parent then
            # unwinds and returns True too.  The original worker finally
            # releases the RLock.
            s.wait(txs[0])
            s.skip(t, rlock.release)

        self.assertEqual(len(predicate_calls), 5)
        self.assertTrue(all(tx.done for tx in txs))

    def test_not_scoped_call_wakes_on_state_and_close(self):
        """Not(Call((thread, base), ...)) wakes when the selected call goes low."""
        s = Scenario()
        rlock = s.RLock()
        condition = s.Condition(rlock)

        def predicate():
            return condition.wait_for(lambda: True, timeout=-1)

        def worker():
            rlock.acquire()
            condition.wait_for(predicate, timeout=-1)
            rlock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, rlock.acquire)
            s.wait(Call(t, condition.wait_for, State.BLOCKED))
            outer = s.transaction(t)
            outer.unblock()
            s.wait(Call((t, outer), condition.wait_for, State.BLOCKED))
            inner = s.transaction(t)

            not_inner_blocked = Not(Call((t, outer), condition.wait_for, State.BLOCKED))
            not_inner_active = Not(Call((t, outer), condition.wait_for))
            results = []

            def wait_for_signal(signal):
                results.append(s.wait(signal, timeout=NEVER))

            waiters = [
                threading.Thread(target=wait_for_signal, args=(not_inner_blocked,)),
                threading.Thread(target=wait_for_signal, args=(not_inner_active,)),
            ]
            for waiter in waiters:
                waiter.start()

            keys = [not_inner_blocked.normalized(), not_inner_active.normalized()]
            deadline = time.perf_counter() + 1
            while True:
                with s._core.lock:
                    parked = all(s._core.waiters.get(key) for key in keys)
                if parked:
                    break
                if time.perf_counter() > deadline:
                    self.fail("Not(Call(...)) waiter threads did not park")
                time.sleep(0.001)

            inner.unblock()

            for waiter in waiters:
                waiter.join(1)
                self.assertFalse(waiter.is_alive())

            self.assertIn({not_inner_blocked}, results)
            self.assertIn({not_inner_active}, results)
            s.wait(outer)
            s.skip(t, rlock.release)


    def test_not_use_wakes_when_thread_stops_using_primitive(self):
        """Not(Use(thread, primitive)) wakes when the selected Use goes low."""
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            use = Use(t, lock)
            s.wait(use)
            not_use = Not(use)
            self.assertFalse(s.wait(not_use, timeout=0))
            results = []

            def wait_for_signal():
                results.append(s.wait(not_use, timeout=NEVER))

            waiter = threading.Thread(target=wait_for_signal)
            waiter.start()

            key = not_use.normalized()
            deadline = time.perf_counter() + 1
            while True:
                with s._core.lock:
                    parked = bool(s._core.waiters.get(key))
                if parked:
                    break
                if time.perf_counter() > deadline:
                    self.fail("Not(Use(thread, primitive)) waiter did not park")
                time.sleep(0.001)

            s.skip(t, lock.acquire)

            waiter.join(1)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(results, [{not_use}])

    def test_scoped_use_signal_and_not_wakeup(self):
        """Use((thread, base_tx), primitive) signals for descendants only.

        The positive signal wakes when a child transaction under the
        base starts using the primitive; the negated signal wakes when
        that selected child transaction closes.
        """
        s = Scenario()
        rlock = s.RLock()
        condition = s.Condition(rlock)

        def predicate():
            return condition.wait_for(lambda: True, timeout=-1)

        def worker():
            rlock.acquire()
            condition.wait_for(predicate, timeout=-1)
            rlock.release()

        with s:
            t = s.thread(worker)
            s.skip(t, rlock.acquire)
            s.wait(Call(t, condition.wait_for, State.BLOCKED))
            outer = s.transaction(t)

            scoped_use = Use((t, outer), condition)
            self.assertFalse(s.wait(scoped_use, timeout=0))

            results = []

            def wait_for_signal(signal):
                results.append(s.wait(signal, timeout=NEVER))

            waiter = threading.Thread(target=wait_for_signal, args=(scoped_use,))
            waiter.start()

            key = scoped_use.normalized()
            deadline = time.perf_counter() + 1
            while True:
                with s._core.lock:
                    parked = bool(s._core.waiters.get(key))
                if parked:
                    break
                if time.perf_counter() > deadline:
                    self.fail("Use((thread, base_tx), primitive) waiter did not park")
                time.sleep(0.001)

            outer.unblock()

            waiter.join(1)
            self.assertFalse(waiter.is_alive())
            self.assertIn({scoped_use}, results)

            inner = s.transaction(t)
            self.assertIsNot(inner, outer)

            not_scoped_use = Not(scoped_use)
            results.clear()
            waiter = threading.Thread(target=wait_for_signal, args=(not_scoped_use,))
            waiter.start()

            key = not_scoped_use.normalized()
            deadline = time.perf_counter() + 1
            while True:
                with s._core.lock:
                    parked = bool(s._core.waiters.get(key))
                if parked:
                    break
                if time.perf_counter() > deadline:
                    self.fail("Not(Use((thread, base_tx), primitive)) waiter did not park")
                time.sleep(0.001)

            inner.unblock()

            waiter.join(1)
            self.assertFalse(waiter.is_alive())
            self.assertIn({not_scoped_use}, results)

            s.wait(outer)
            s.skip(t, rlock.release)

    def test_use_signal_refcounted_through_recursion(self):
        """Use(thread, primitive) stays high through recursive wait_for.

        Use is self-reporting: it walks the active transaction chain, so
        it remains high while either the outer or inner wait_for is using
        the condition.
        """
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
            s.skip(t, rlock.acquire)

            call0 = Call(t, condition.wait_for, State.BLOCKED)
            use = Use(t, condition)
            signaled = s.wait(call0, use)
            self.assertIn(use, signaled)
            outer_tx = s.transaction(t)
            outer_tx.unblock()

            s.wait(Call((t, outer_tx), condition.wait_for, State.BLOCKED))
            # Both outer and inner wait_for txs are in the thread's
            # transaction chain, so Use(t, condition) remains high.
            signaled = s.wait(use)
            self.assertIn(use, signaled)

            # Drive both back to RETURNED.  The inner predicate returns
            # True, then unwinds; the outer returns True too.
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
            s.skip(t, rlock.release)

            # Now nothing is in flight; both Use signals are gone.
            self.assertFalse(s.wait(use, timeout=0))
            self.assertFalse(s.wait(Use(t, rlock), timeout=0))

    def test_raw_form_signals_for_condition_family(self):
        """Inside a Lock-family member's tx, every family member
        (lock, conditions, and their raws) signals as bare primitive forms.
        Raw and cooked primitive forms normalize to the same key, so waiting
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
                lock, lock_raw, cond, cond_raw,
                Use(t, lock), Use(t, lock_raw), Use(t, cond), Use(t, cond_raw),
                )
            # All four primitive forms signal (raw normalizes to cooked,
            # so each spelling comes back).
            self.assertIn(lock, signaled)
            self.assertIn(lock_raw, signaled)
            self.assertIn(cond, signaled)
            self.assertIn(cond_raw, signaled)

            # All four Use forms also signal.
            self.assertIn(Use(t, lock), signaled)
            self.assertIn(Use(t, lock_raw), signaled)
            self.assertIn(Use(t, cond), signaled)
            self.assertIn(Use(t, cond_raw), signaled)


    def test_bare_primitive_not_signal_and_no_public_wrapper(self):
        """Primitive signals are internal; users wait on primitives directly."""
        self.assertFalse(hasattr(blanket, 'Primitive'))
        self.assertNotIn('Primitive', blanket.__all__)

        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            self.assertIn(lock, s.wait(lock))
            self.assertFalse(s.wait(Not(lock), timeout=0))
            s.skip(t, lock.acquire)
            self.assertIn(Not(lock), s.wait(Not(lock)))


    def test_raw_method_call_signals_alongside_primitive(self):
        """Call(t, raw_method, ...) signals whenever Call(t, primitive_method, ...)
        does.  Raw and cooked method forms normalize to the same key."""
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
            s.skip(t, lock.acquire)
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
            s.skip(t, lock.release)

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
            s.skip(n, lock.acquire)
            s.wait(n)

            cycle = capi.cycle(a, b, c, n)
            s.skip(n, lock.release)
            self.assertEqual(cycle.wake(b), (b,))
            s.skip(b, lock.release)
            self.assertEqual(cycle.wake(a), (a,))
            s.skip(a, lock.release)
            self.assertEqual(cycle.close(), (c,))
            s.skip(c, lock.release)

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

    def test_call_thread_scope_validation_errors(self):
        scenario = Scenario()
        lock = scenario.Lock()
        with self.assertRaisesRegex(TypeError, "thread or"):
            Call(object(), lock.acquire)
        with self.assertRaisesRegex(TypeError, "2-tuple"):
            Call((threading.current_thread(), object(), object()), lock.acquire)
        with self.assertRaisesRegex(TypeError, "first item"):
            Call((object(), object()), lock.acquire)
        with self.assertRaisesRegex(TypeError, "second item"):
            Call((threading.current_thread(), object()), lock.acquire)

    def test_read_only_list_index_with_explicit_stop(self):
        scenario = Scenario()
        proxy = scenario._core.ReadOnlyListProxy([1, 2, 3, 2])
        self.assertEqual(proxy.index(2, 2, 4), 3)

    def test_park_raises_when_waited_thread_never_reaches_method(self):
        scenario = Scenario()
        lock = scenario.Lock()

        def worker():
            lock.locked()

        with scenario:
            thread = scenario.thread(worker)
            with self.assertRaisesRegex(RuntimeError, "terminated before reaching"):
                scenario.park(thread, lock.acquire)  # never called

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

    def test_call_thread_scope_rejects_mismatched_base_thread(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()

        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            other = threading.Thread(target=lambda: None, name='other')
            with self.assertRaisesRegex(ValueError, "belongs to thread"):
                Call((other, tx), lock.acquire)
            tx.unblock()

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
        self.assertFalse(lock.locked())
        self.assertIn('RLock', repr(rlock))

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

    def test_barrier_cycle_with_pause(self):
        """pause() on a barrier cycle parks a triggered waiter at PAUSED
        with the scheduler pause flag (mirrors test_event_cycle_with_pause);
        unpause() resumes it, and close() drains the rest."""
        s = Scenario()
        barrier = s.Barrier(2)
        api = s.api(barrier)
        log = []

        def worker(name):
            def fn():
                barrier.wait(); log.append(name)
            return fn

        with s:
            a = s.thread(worker('a'))
            b = s.thread(worker('b'))
            s.wait(a)
            s.wait(b)
            cycle = api.cycle(a, b)
            self.assertEqual(cycle.pause(a), (a,))
            tx = s.transaction(a)
            self.assertEqual(tx.state, State.PAUSED)
            self.assertEqual(log, [])
            tx.unpause()
            s.wait(tx)
            cycle.close()

        self.assertEqual(sorted(log), ['a', 'b'])

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
            self.assertTrue(r.startswith('<ModuleImpersonator '))
            # Names the module it stands in for...
            self.assertIn("'threading'", r)
            # ...and includes the scenario's repr so two concurrent
            # stand-ins over different scenarios are distinguishable.
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
        many CPython versions and is the standard pattern-2 use
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
        bound to the scenario.  This is the standard pattern-1 use
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

    # ---- module passthrough handles ------------------------------

    def test_threading_handle_primitives_and_fallthrough(self):
        scenario = Scenario()
        for n in self.PRIMITIVE_NAMES:
            with self.subTest(primitive=n):
                self.assertIs(getattr(scenario.threading, n),
                              getattr(scenario, n))
        # everything else falls through to the real module
        self.assertIs(scenario.threading.Thread, threading.Thread)
        self.assertIs(scenario.threading.current_thread,
                      threading.current_thread)

    def test_queue_handle_primitives_and_fallthrough(self):
        scenario = Scenario()
        self.assertIs(scenario.queue.Queue, scenario.Queue)
        self.assertIs(scenario.queue.LifoQueue, scenario.LifoQueue)
        self.assertIs(scenario.queue.PriorityQueue, scenario.PriorityQueue)
        self.assertEqual(hasattr(scenario.queue, 'SimpleQueue'), hasattr(queue, 'SimpleQueue'))
        if hasattr(queue, 'SimpleQueue'):
            self.assertIs(scenario.queue.SimpleQueue, scenario.SimpleQueue)
        self.assertIs(scenario.queue.Empty, queue.Empty)
        self.assertIs(scenario.queue.Full, queue.Full)

    def test_handles_are_cached(self):
        scenario = Scenario()
        self.assertIs(scenario.threading, scenario.threading)
        self.assertIs(scenario.queue, scenario.queue)
        self.assertIsNot(scenario.threading, scenario.queue)

    def test_handle_repr_names_module_and_scenario(self):
        scenario = Scenario()
        r = repr(scenario.queue)
        self.assertTrue(r.startswith('<ModuleImpersonator '))
        self.assertIn("'queue'", r)
        self.assertIn(repr(scenario), r)

    # ---- queue injection -----------------------------------------

    def test_pattern1_queue_from_import(self):
        """from queue import Queue -> scenario.Queue."""
        scenario = Scenario()
        target = self.make_module(Queue=queue.Queue)
        with scenario.inject(target):
            self.assertIs(target.Queue, scenario.Queue)
            self.assertIs(bound_to(type(target.Queue())), scenario)
        self.assertIs(target.Queue, queue.Queue)

    def test_pattern2_queue_module_attr_replaced_with_standin(self):
        """import queue -> the attribute becomes the queue impersonator."""
        scenario = Scenario()
        target = self.make_module(queue=queue)
        with scenario.inject(target):
            self.assertIs(target.queue, scenario.queue)
            self.assertIs(target.queue.Queue, scenario.Queue)
            self.assertIs(target.queue.Empty, queue.Empty)  # fallthrough
        self.assertIs(target.queue, queue)

    def test_inject_patches_threading_and_queue_together(self):
        """A single inject handles references to both modules."""
        scenario = Scenario()
        target = self.make_module(
            Lock=threading.Lock,
            Queue=queue.Queue,
            threading=threading,
            queue=queue,
        )
        with scenario.inject(target):
            self.assertIs(target.Lock, scenario.Lock)
            self.assertIs(target.Queue, scenario.Queue)
            self.assertIs(target.threading, scenario.threading)
            self.assertIs(target.queue, scenario.queue)
        self.assertIs(target.Lock, threading.Lock)
        self.assertIs(target.Queue, queue.Queue)
        self.assertIs(target.threading, threading)
        self.assertIs(target.queue, queue)



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
        # Out of scenario now.
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            d.finish()

    def test_scenario_methods_outside_scenario_raise(self):
        """scenario.skip / park / pause raise if not entered."""
        s = Scenario()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.skip()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.park()
        with self.assertRaisesRegex(RuntimeError, "scenario not entered"):
            s.pause()

    def test_two_drivers_dont_compete_until_drive(self):
        """Relaxed ownership: a Driver holds the score slot only while
        actively driving.  Two Driver(t) for one thread coexist; once
        d1 drives and yields (releasing the slot), d2 can drive the
        same thread without competing."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            s.wait(t)             # t parked at lock.acquire
            d1 = s.Driver(t)
            d2 = s.Driver(t)      # no raise -- both inert
            d1.scan(); d1()       # explicit scan discovers acquire
            d1.finish()             # claims slot; skip acquire
            d1()                  # SUCCESS at lock.release -- slot released
            # d1 has yielded, so d2 may now drive the same thread.
            d2.scan(); d2()       # explicit scan discovers release
            d2.finish()             # no CompetingDriversError
            d2()                  # skip release; thread has no current tx
            self.assertIs(d2.state, d2.success)

    def test_driver_init_preserves_scheduler_pause_bit(self):
        """Driver initialization does not clear a scheduler pause."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            tx.pause = True
            d = s.Driver(t)
            d.scan()
            d()
            self.assertTrue(tx.pausing)
            tx.pause = False
            d.finish()
            d()


class TestDriverImperatives(unittest.TestCase):

    def assertDriverSignalSets(self, d):
        self.assertIsInstance(d.waited, frozenset)
        self.assertIsInstance(d.signaled, frozenset)
        self.assertIsInstance(d.motivation, frozenset)
        self.assertLessEqual(d.motivation, d.signaled)
        self.assertLessEqual(d.signaled, d.waited)

    def test_driver_skip_is_removed(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            self.assertFalse(hasattr(d, 'skip'))
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_finish(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertNotIn(Terminated(yielded.thread), yielded.motivation)

    def test_raised_is_recoverable(self):
        """An unanticipated raise lands the Driver in `raised`, which is NOT
        terminal.  The worker raised but caught it and kept running, so the
        Driver can scan() and drive the thread's next transaction.  Only a
        gone thread (`terminated`) is a true dead-end."""
        s = Scenario()
        sem = s.BoundedSemaphore(1)   # already at max: release() raises ValueError
        lock = s.Lock()
        log = []

        def worker():
            try:
                sem.release()
            except ValueError:
                log.append("caught")
            with lock:                # worker continues into a new transaction
                log.append("locked")

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            self.assertEqual(d.tx.method.__name__, "release")
            d.finish(); d()
            self.assertIs(d.state, d.unexpected)
            self.assertNotIn(Terminated(d.thread), d.motivation)  # the whole point: still drivable
            # Recover off the live published tx the caught-and-continued worker made.
            d.scan(); d()
            self.assertIsNotNone(d.tx)
            self.assertEqual(d.tx.method.__name__, "acquire")
            d.finish(); d()
            self.assertIs(d.state, d.success)
        self.assertEqual(log, ["caught", "locked"])

    def test_block(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.blocked()
            d()  # drives d to a successful BLOCKED park
            self.assertIs(d.state, d.success)
            self.assertIs(d.tx.state, State.BLOCKED)
            self.assertNotIn(Terminated(d.thread), d.motivation)
            # Release worker so cleanup joins.
            s.api(lock).unblock(lock.acquire, t)

    def test_commit(self):
        """Driver.commit() succeeds when the tx reaches COMMIT.

        COMMIT is externally controlled, so the tx may naturally race
        onward after that without contradicting the directive.
        """
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.commit()
            # Drive to the COMMIT parking point.
            d.finish(); d()
            self.assertIs(d.state, d.success)
            self.assertIn(d.tx.state, (State.COMMIT, State.RETURNED))
            d.close()

    def test_pause(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            d.scan(); d()
            d.paused()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIs(tx.state, State.PAUSED)
            self.assertTrue(tx.pausing)
            disp.close()

    def test_wait(self):
        s = Scenario()
        gate = threading.Event()
        def waiter():
            gate.wait()
        with s:
            w = s.thread(waiter)
            dw = s.Driver(w)
            target = Terminated(w)
            dw.wait(target)
            helper = threading.Thread(target=gate.set)
            helper.start()
            dw()
            helper.join()
            self.assertIs(dw.state, dw.success)
            self.assertIn(target, dw.waited)
            self.assertIn(target, dw.signaled)
            self.assertIn(target, dw.motivation)
            self.assertDriverSignalSets(dw)

    def test_stall(self):
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)
        def waiter():
            with lock:
                cond.wait(timeout=0)
        with s:
            w = s.thread(waiter)
            s.skip(w, lock.acquire)
            s.wait(w)
            d = s.Driver(w)
            d.scan(); d()
            d.stalled()
            # Drive d to park at STALLED.  The test only needs the parked
            # state; scenario exit cleanup releases the worker afterwards.
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIs(d.tx.state, State.STALLED)

    def test_commit_validates_type(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            s.skip(t, lock.acquire)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            with self.assertRaisesRegex(RuntimeError, "park in COMMIT"):
                d.commit()
            d.close()

    def test_wait_requires_signal(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            with self.assertRaisesRegex(ValueError, "at least one signal"):
                d.wait()
            d.close()

    def test_public_driver_status_and_log_record_completed_directives(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            self.assertIsNone(d.status)
            self.assertEqual(d.log, ())
            self.assertIsNone(d.directive)
            self.assertEqual(d.args, ())

            d.scan()
            # Staging alone is not a completed Driver outcome.
            self.assertIsNone(d.status)
            self.assertIsNone(d.directive)
            self.assertEqual(d.args, ())

            d()
            self.assertIsInstance(d.status, DriverStatus)
            self.assertIn("DriverStatus", repr(d.status))
            with self.assertRaisesRegex(ValueError, "motivation <= signaled <= waited"):
                DriverStatus(None, (), d.success, None, (), (), (Terminated(t),))
            raw = DriverStatus(None, (), d.success, None, (), (), ())
            self.assertIsNone(d._public_status(raw).directive)
            self.assertEqual(d.directive, d.scan)
            self.assertEqual(d.args, (None,))
            self.assertEqual(len(d.log), 1)
            self.assertIs(d.log[-1].state, d.success)
            self.assertEqual(d.log[-1].directive, d.scan)

            d.finish()
            # Until the next drive, status continues to describe the last
            # completed outcome.
            self.assertEqual(d.directive, d.scan)
            d()
            self.assertEqual(d.directive, d.finish)
            self.assertEqual(d.args, ())
            self.assertEqual(len(d.log), 2)
            self.assertLessEqual(d.status.motivation, d.status.signaled)
            self.assertLessEqual(d.status.signaled, d.status.waited)
            d.close()

    def test_stall_validates_type(self):
        s = Scenario()
        ev = s.Event()
        def waiter():
            ev.wait()
        with s:
            w = s.thread(waiter)
            s.wait(w)
            d = s.Driver(w)
            d.scan(); d()
            with self.assertRaisesRegex(RuntimeError, "park in STALLED"):
                d.stalled()
            # stall() raised before staging; d is still at success
            # and owns w.  Set the underlying event so when the
            # commit's actual.wait runs it returns immediately, then
            # close d and finish w.
            s.raw(ev).set()
            d.close()

    def test_imperative_raises_if_not_active(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            list(disp)  # drive d through the current tx
            # A successful finish leaves no next tx selected; a new
            # driving directive must be preceded by scan().
            with self.assertRaisesRegex(RuntimeError, "scan first"):
                d.finish()


class TestDriverRoutes(unittest.TestCase):

    def test_constructor_route_scans_then_drives_until_route_returns(self):
        s = Scenario()
        lock = s.Lock()
        seen = []

        def worker():
            lock.acquire()
            lock.release()

        def route(d):
            seen.append(("start", d.state, d.tx))
            d.scan()
            yield
            seen.append(("scan", d.state, d.tx.method))
            d.finish()
            yield
            seen.append(("skip", d.state, d.tx.method))

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, lock.acquire)
            self.assertEqual(seen, [
                ("start", d.undirected, None),
                ("scan", d.success, lock.acquire),
                ("skip", d.success, lock.acquire),
            ])
            d.scan(); d()
            d.finish(); d()

    def test_dispatch_route_drives_until_route_returns(self):
        s = Scenario()
        lock = s.Lock()

        def worker():
            lock.acquire()
            lock.release()

        def route(d):
            d.scan()
            yield
            d.finish()
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded, d)
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, lock.acquire)
            d.scan(); d()
            d.finish(); d()

    def test_staged_route_install_can_be_overwritten(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        ran = []

        def route(d):
            ran.append(True)  # pragma: no cover - this route is intentionally overwritten
            return            # pragma: no cover
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.route(route)
            d.paused()
            d()
            self.assertEqual(ran, [])
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertIs(d.tx.state, State.PAUSED)
            d.finish(); d()

    def test_route_imperatives_must_come_from_route_thread(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        errors = []

        def route(d):
            def outsider():
                try:
                    d.scan()
                except RuntimeError as e:
                    errors.append(str(e))
            thread = threading.Thread(target=outsider)
            thread.start()
            thread.join()
            d.scan()
            yield
            d.finish()
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertEqual(len(errors), 1)
            self.assertIn("being driven right now", errors[0])
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertTrue(d.tx.done)

    def test_route_bare_yield_raises(self):
        s, lock, worker = _make_scenario_with_lock_worker()

        def route(d):
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            with self.assertRaisesRegex(RuntimeError, "without issuing"):
                d()
            self.assertIs(d.state, d.undirected)
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_route_close_blocks_finalizer_imperatives(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        errors = []

        def route(d):
            try:
                yield
            finally:
                try:
                    d.finish()
                except RuntimeError as e:
                    errors.append(str(e))

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            with self.assertRaisesRegex(RuntimeError, "without issuing"):
                d()
            self.assertEqual(len(errors), 1)
            self.assertIn("being driven right now", errors[0])
            self.assertIs(d.state, d.undirected)
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_route_return_after_staging_directive_is_honored(self):
        s, lock, worker = _make_scenario_with_lock_worker()

        def route(d):
            d.scan()
            return
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, lock.acquire)
            d.finish(); d()

    def test_route_continues_after_successful_finish(self):
        s = Scenario()
        lock = s.Lock()
        seen = []

        def worker():
            lock.acquire()

        def route(d):
            d.scan()
            yield
            d.finish()
            yield
            seen.append(d.state)

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(seen, [d.success])

    def test_route_waits_for_named_signal(self):
        s = Scenario()
        gate = threading.Event()
        seen = []

        def waiter():
            gate.wait()

        def route(d):
            target = Terminated(t)
            d.wait(target)
            yield
            seen.append((d.signaled, d.motivation))

        with s:
            t = s.thread(waiter)
            d = s.Driver(t, route=route)
            helper = threading.Thread(target=gate.set)
            helper.start()
            d()
            helper.join()
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(seen, [(frozenset({Terminated(t)}),
                                     frozenset({Terminated(t)}))])

    def test_routes_compose_with_yield_from(self):
        s = Scenario()
        lock = s.Lock()
        seen = []

        def worker():
            lock.acquire()
            lock.release()

        def skip_one(d):
            d.finish()
            yield
            d.scan()
            yield
            seen.append(d.tx.method)

        def route(d):
            d.scan()
            yield
            yield from skip_one(d)
            self.assertEqual(d.tx.method, lock.release)
            d.finish()
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertEqual(seen, [lock.release])

    def test_route_rejects_bad_installations(self):
        s, lock, worker = _make_scenario_with_lock_worker()

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            with self.assertRaisesRegex(TypeError, "route must be callable"):
                d.route(None)
            with self.assertRaisesRegex(TypeError, "route must return an iterator"):
                d.route(lambda driver: object())
            with self.assertRaisesRegex(TypeError, "route must return an iterator"):
                d.route(lambda driver: [])
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_route_bare_iterator_without_close_raises(self):
        s, lock, worker = _make_scenario_with_lock_worker()

        class BareIterator:
            def __next__(self):
                return None

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=lambda driver: BareIterator())
            with self.assertRaisesRegex(RuntimeError, "without issuing"):
                d()
            self.assertIs(d.state, d.undirected)
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_route_calling_route_tail_calls_successor(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        seen = []

        def route2(d):
            seen.append("route2")
            d.scan()
            yield
            d.finish()
            yield

        def route1(d):
            seen.append("route1")
            d.route(route2)
            return
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route1)
            d()
            self.assertEqual(seen, ["route1", "route2"])
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertTrue(d.tx.done)

    def test_route_route_then_other_directive_clears_route(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        ran = []

        def replacement(d):
            ran.append("replacement")  # pragma: no cover - this route is intentionally overwritten
            return                     # pragma: no cover
            yield

        def route(d):
            d.scan()
            yield
            d.route(replacement)
            d.paused()
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertEqual(ran, [])
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.success)
            self.assertIs(d.tx.state, State.PAUSED)
            d.finish(); d()

    def test_route_exception_raises_and_closes_route(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        closed = []

        def route(d):
            try:
                raise ValueError("route blew up")
                yield
            finally:
                closed.append(True)

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            with self.assertRaisesRegex(ValueError, "route blew up"):
                d()
            self.assertIs(d.state, d.unexpected)
            self.assertEqual(closed, [True])
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_route_may_return_at_first_ask(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        seen = []

        def route(d):
            seen.append(d.state)
            return
            yield

        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t, route=route)
            d()
            self.assertEqual(seen, [d.undirected])
            self.assertFalse(d.routed)
            self.assertIs(d.state, d.undirected)
            d.close()
            s.api(lock).unblock(lock.acquire, t)

    def test_wait_rejects_pending_or_invalid_states(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            tx = d.tx
            d.finish()
            # Last-wins staging lets wait() overwrite the pending finish;
            # wait() itself is passive, so an explicit out-of-band release
            # makes the tx exit while the Driver observes it.
            d.wait(tx)
            disp = s.Dispatch()
            disp.add(d)
            s.api(lock).unblock(lock.acquire, t)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIn(tx, d.signaled)
            self.assertIn(tx, d.motivation)
            with self.assertRaisesRegex(RuntimeError, "scan first|currently in"):
                d.finish()

    def test_wait_from_idle_handles_thread_start_and_termination(self):
        s = Scenario()

        def worker():
            return None

        with s:
            t = s.thread(worker)
            d = s.Driver(t)
            d.wait(Terminated(t))
            d()
            self.assertIs(d.state, d.success)
            self.assertEqual(set(d.signaled), {Terminated(t)})
            self.assertEqual(set(d.motivation), {Terminated(t)})

    def test_wait_is_passive_at_stalled_and_paused_states(self):
        s = Scenario()
        lock = s.Lock()
        cond = s.Condition(lock)

        def stalled_worker():
            with lock:
                cond.wait(timeout=0)

        with s:
            t = s.thread(stalled_worker)
            s.skip(t, lock.acquire)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.stalled()
            d()
            self.assertIs(d.tx.state, State.STALLED)
            tx = d.tx
            d.wait(tx)
            disp = s.Dispatch(); disp.add(d)
            tx.unstall()
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIn(tx, d.signaled)

        s = Scenario()
        lock = s.Lock()

        def paused_worker():
            lock.acquire()

        with s:
            t = s.thread(paused_worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.paused()
            d()
            self.assertIs(d.tx.state, State.PAUSED)
            tx = d.tx
            d.wait(tx)
            disp = s.Dispatch(); disp.add(d)
            tx.unpause()
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIn(tx, d.signaled)


class TestDriverCascade(unittest.TestCase):

    def test_terminated_signal_in_idle(self):
        s = Scenario()
        with s:
            t = s.thread(lambda: None)
            t.join()
            s.wait(Terminated(t))
            d = s.Driver(t)
            d.scan()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.unexpected)
            self.assertIn(Terminated(yielded.thread), yielded.motivation)

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
            d.scan()
            disp = s.Dispatch()
            disp.add(d)
            gate.set()
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            d.close()

    def test_wait_does_not_treat_tx_exit_as_intrinsic_failure(self):
        s = Scenario()
        sem = s.Semaphore(value=1)
        def worker():
            sem.acquire()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            # If the author cares about tx-exit, they list the tx.
            d.wait(tx)
            disp = s.Dispatch()
            disp.add(d)
            s.api(sem).unblock(sem.acquire, t)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            self.assertIn(tx, d.signaled)

    def test_legacy_cascade_state_machine_is_removed(self):
        """The tranche-2 Driver no longer exposes the old cascade/mode state machine."""
        s = Scenario()
        thread = threading.Thread(name="fake-driver-thread")
        with s:
            d = s.Driver(thread)
            self.assertFalse(hasattr(d._core, "mode"))
            self.assertFalse(hasattr(d._core, "skipping"))
            self.assertFalse(hasattr(d._core, "active"))
            d.close()


    def test_finish_route_then_scan_finds_next_tx(self):
        """After a deep finish route completes one tx, scan() is the
        explicit way to select a subsequent regulated call."""
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: True, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.route(lambda dd, core=s._core: core.route_finish_deep(dd)); d()
            d.scan(); d()
            self.assertIs(d.state, d.success)
            self.assertIsNotNone(d.tx)
            self.assertEqual(d.tx.method.__name__, 'release')
            d.close()

    def test_deep_finish_route_then_scan_finds_next_tx(self):
        """A deep finish route handles the nested wait_for chain as
        one route, and an explicit scan() selects the worker's next
        regulated call afterward."""
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
            s.skip(t, lock.acquire)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.route(lambda dd, core=s._core: core.route_finish_deep(dd)); d()
            d.scan(); d()
            self.assertIs(d.state, d.success)
            self.assertIsNotNone(d.tx)
            self.assertEqual(d.tx.method.__name__, 'release')
            d.close()

    def test_cascade_exhaust_yields_to_idle_when_worker_has_no_next_tx(self):
        """Same nested wait_for scenario as above, but the worker
        ends after the wait_for chain.  A following scan() therefore
        lands in terminated instead of selecting a next transaction."""
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
            s.skip(t, lock.acquire)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.route(lambda dd, core=s._core: core.route_finish_deep(dd)); d()
            self.assertIs(d.state, d.success)
            self.assertTrue(d.tx.done)
            d.scan(); d()
            self.assertIs(d.state, d.unexpected)
            self.assertIn(Terminated(d.thread), d.motivation)

    def test_scan_is_the_explicit_recovery_path_after_finish(self):
        """finish() handles one tx; scan() is how the Driver reaches the next one."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            first = d.tx
            d.finish(); d()
            self.assertIs(d.state, d.success)
            self.assertTrue(first.done)
            d.scan(); d()
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, lock.release)
            d.finish(); d()



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

    def test_add_done_driver_is_ready_immediately(self):
        s = Scenario()
        with s:
            t = s.thread(lambda: None)
            t.join()
            s.wait(Terminated(t))
            d = s.Driver(t)
            d.scan()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.unexpected)
            self.assertIn(Terminated(yielded.thread), yielded.motivation)

    def test_dispatch_yields_done_then_stops(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
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


class TestDriverAPI(unittest.TestCase):

    def test_repr_says_scenario_driver(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan()
            d()  # trigger explicit scan / first-use initialization
            r = repr(d)
            self.assertIn("Scenario.Driver", r)
            self.assertIn("SUCCESS", r)
            d.finish(); d()

    def test_thread_attribute(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            self.assertIs(d.thread, t)
            d.close()

    def test_base_tx_attribute(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)

            d = s.Driver(t)
            self.assertIsNone(d.base_tx)
            d.close()

            d = s.Driver(t, tx)
            self.assertIs(d.base_tx, tx)
            d.close()

            s.skip(t, lock.acquire)
            s.raw(lock).release()

    def test_txs_history(self):
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()  # explicit scan populates txs
            self.assertEqual(len(d.txs), 1)
            d.finish(); d()

    def test_state_constants_at_class_level(self):
        self.assertEqual(Scenario.Driver.undirected.name, 'UNDIRECTED')
        self.assertEqual(Scenario.Driver.driving.name, 'DRIVING')
        self.assertEqual(Scenario.Driver.success.name, 'SUCCESS')
        self.assertEqual(Scenario.Driver.unexpected.name, 'UNEXPECTED')
        self.assertEqual(Scenario.Driver.mutated.name, 'MUTATED')
        # terminated / raised / impasse are until() prediction targets now,
        # not Driver rest states.
        self.assertEqual(Scenario.Driver.terminated.name, 'terminated')
        self.assertEqual(Scenario.Driver.raised.name, 'raised')
        self.assertEqual(Scenario.Driver.impasse.name, 'impasse')
        # The divergence states are gone -- folded into `unexpected`, with the
        # signal(s) that fired carried in motivation.  `terminal_states` is
        # gone too: termination is ground truth via the Driver's done property.
        for gone in ('returned', 'overshot', 'persisted', 'nested',
                     'terminal_states'):
            self.assertFalse(hasattr(Scenario.Driver, gone),
                             f"{gone} should be gone")



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
            for d in (da, db, dc):
                d.scan()
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
            for d in (da, db):
                d.scan()
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
            for d in (da, db):
                d.scan()
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

            # Cleanup: the promoted Drivers are unowned; explicitly
            # scan each one before finishing it.
            for d in (d1, d2):
                d.scan(); d()
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
            for d in (da, db):
                d.scan()
            chain = s.Chain(da, db)

            collected = []
            d = chain.promote()
            while d is not None:
                collected.append(d)
                d.scan(); d()
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
            for d in (da, db, dc):
                d.scan()
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
            for d in (da, db, dc):
                d.scan()
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
            d.scan()
            disp = s.Dispatch()
            disp.add(d)
            yielded = next(iter(disp))
            self.assertIs(yielded.state, d.success)
            d.close()

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

    def test_driver_close_idempotent(self):
        """Driver.close is safe to call twice."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.close()
            d.close()  # no raise
            d2 = s.Driver(t)
            self.assertIs(d2.thread, t)
            d2.close()

    def test_driver_close_after_auto_close_is_noop(self):
        """A driver that auto-closed on reaching terminal can still
        have close() called without raising."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.finish()
            disp = s.Dispatch()
            disp.add(d)
            list(disp)  # drives d to a successful FINISHED internal mode
            self.assertIs(d.state, d.success)
            self.assertNotIn(t, s._core.drivers)
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
            d1.scan(); d2.scan()
            chain = s.Chain(d1, d2)
            self.assertEqual(len(chain), 2)
            chain.close()
            self.assertEqual(len(chain), 0)
            self.assertFalse(chain)
            self.assertEqual(chain.pending, ())
            # Drivers were closed; can create new ones.

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
            d1.scan(); d2.scan()
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

    def test_chain_close_idempotent(self):
        s = Scenario()
        with s:
            chain = s.Chain()
            chain.close()
            chain.close()  # no raise
            self.assertFalse(chain)

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
            d1.scan()
            d2.scan()
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

    def test_dispatch_close_idempotent(self):
        s = Scenario()
        with s:
            disp = s.Dispatch()
            disp.close()
            disp.close()  # no raise
            self.assertIn('Scenario.Dispatch', repr(disp))

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
        # WaitTransaction(score, items): bare primitives are auto-boxed.
        item = ev
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

    def test_threads_to_txs_self_thread_raises(self):
        """primitive core threads_to_txs raises ValueError on caller."""
        s = Scenario()
        lock = s.Lock()
        core = s.api(lock)._core
        with s:
            with self.assertRaisesRegex(ValueError, "calling thread"):
                core.threads_to_txs((threading.current_thread(),), 'threads_to_txs_self_thread_raises')

    def test_staged_directives_are_last_wins(self):
        """A second directive before driving overwrites the first."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.paused()
            d.finish()
            d()
            self.assertIs(d.state, d.success)
            self.assertIs(d.tx.state, State.RETURNED)

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
            d.scan(); d()
            d.paused()
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
            s.skip(w, lock.acquire)
            s.wait(w)
            d = s.Driver(w)
            d.scan(); d()
            d.stalled()
            d()
            self.assertIs(d.tx.state, State.STALLED)
            d.close()
            # Exit: auto-unpark transitions STALLED -> RESUMED so
            # the waiter runs _acquire_restore and finishes cond.wait.
        self.assertEqual(ran, ['woke'])

    def test_scenario_exit_handles_mid_tx_worker(self):
        """scenario exit handles a worker that has advanced past
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
            tx = s.transaction(w)
            d = s.Driver(w)
            d.scan(); d()
            with d._lock:
                d._core.waiting()
            d()
            self.assertIs(d.tx.state, State.WAITING)
            d.close()  # release the slot
            # Release the setter so it fires the event; scenario exit
            # then joins both managed threads.
            gate.set()

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

    def test_tx_unblock_raises_in_non_blocked_state(self):
        """tx.unblock raises RuntimeError if the tx isn't at BLOCKED."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            tx = s.transaction(t)
            d = s.Driver(t)
            d.scan(); d()
            d.paused()
            d()
            # Tx is now at PAUSED; unblock should refuse.
            with self.assertRaisesRegex(RuntimeError, "can't unblock"):
                tx.unblock()
            disp = s.Dispatch()
            d.finish()
            disp.add(d)
            list(disp)

    def test_tx_pause_past_paused_raises(self):
        """tx.pause = True raises if tx has advanced past PAUSED."""
        s = Scenario()
        ev = s.Event()
        def w():
            ev.set()
        with s:
            t = s.thread(w)
            s.skip(t, ev.set)
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
            s.skip(t, ev.set)
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
        self.assertIn('RLock.raw', repr(raw))

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
            d1.scan()
            d2.scan()
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

    def test_driver_callable(self):
        """Driver.__call__ runs the driver synchronously."""
        s, lock, worker = _make_scenario_with_lock_worker()
        with s:
            t = s.thread(worker)
            s.wait(t)
            d = s.Driver(t)
            d.scan(); d()
            d.finish()
            d()  # advance synchronously
            self.assertIs(d.state, d.success)
            self.assertTrue(d.tx.done)
            d.close()



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

            def cond_waiter():
                cond_lock.acquire()
                cond.wait(timeout=NEVER)
                cond_lock.release()
            t_cond = s.thread(cond_waiter)
            s.skip(t_cond, cond_lock.acquire)
            s.wait(cond.wait, t_cond)
            tx_cond = s.transactions[t_cond]
            tx_cond.disregard()
            self.assertIsNone(tx_cond._core._timeout)
            self.assertIsNone(tx_cond._core.no_timeout)
            # Drive the wait to terminal so the scenario exits cleanly:
            # expire forces actual.wait(0) on commit.
            tx_cond.expire()
            s.skip(t_cond, cond.wait)

    def test_timeout_getter_returns_remaining(self):
        # The getter does the live math: tx._timeout (duration from
        # start_time) minus elapsed, clamped at 0.
        s = Scenario()
        lock = s.Lock()
        with s:
            # Hold the lock externally so the worker blocks.
            holder = s.thread(lambda: lock.acquire())
            s.skip(holder, lock.acquire)
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
            s.skip(t, lock.acquire)
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
            except RuntimeError:  # coverage: thread-defensive cleanup
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



class TestAction(unittest.TestCase):
    """Push/pop, Action signal, and cycle-validation coverage.

    Push/pop is an internal mechanism (barrier
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

    def test_action_signal_low_before_push(self):
        """Action(tx).sample returns False when tx isn't pushed."""
        s = Scenario()
        lock = s.Lock()
        with s:
            t = s.thread(lock.locked)
            s.wait(t)
            tx = s.transaction(t)
            self.assertFalse(Action(tx).sample(s))

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

            self.assertFalse(Action(opener).sample(s))

            def drive_child(tx):
                # Block until the worker enters the action.  The
                # scheduler runs concurrently with the worker thread
                # leaving WAITING and entering the action, so
                # Action(tx) is initially low; we wait until it
                # fires.  tx is the opener's tx, passed by the cycle.
                s.wait(Action(tx))
                observed['during_action'] = Action(tx).sample(s)
                child = s.transaction(x)
                self.assertEqual(child.method, lock.locked)
                # Under the new design the child is a proper child of
                # opener; the framework does not detach it.
                self.assertIs(child.parent, opener)
                child.unblock()
                s.wait(child)

            cycle = bapi.cycle(a, x, scheduler=drive_child)
            cycle.close()

            # After the action returns, Action(opener) goes low.
            self.assertFalse(Action(opener).sample(s))

        self.assertTrue(observed['during_action'])

    def test_not_action_wakes_when_barrier_action_exits(self):
        s = Scenario()
        lock = s.Lock()
        results = []

        def action(tx):
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

            def drive_child(tx):
                s.wait(Action(tx))
                signal = Not(Action(tx))
                self.assertFalse(s.wait(signal, timeout=0))

                def wait_for_signal():
                    results.append(s.wait(signal, timeout=NEVER))

                parked = threading.Thread(target=wait_for_signal)
                parked.start()
                deadline = time.perf_counter() + 1
                while True:
                    with s._core.lock:
                        is_parked = bool(s._core.waiters.get(signal.normalized()))
                    if is_parked:
                        break
                    if time.perf_counter() > deadline:
                        self.fail("Not(Action(tx)) waiter did not park")
                    time.sleep(0.001)

                child = s.transaction(x)
                child.unblock()
                s.wait(child)
                observed_waiter[0] = parked

            observed_waiter = [None]
            cycle = bapi.cycle(a, x, scheduler=drive_child)
            cycle.close()

            parked = observed_waiter[0]
            parked.join(1)
            self.assertFalse(parked.is_alive())
            self.assertEqual(results, [{Not(Action(opener))}])

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

            def scheduler(tx):
                pass  # coverage: intentionally never invoked

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
            self.assertEqual(cycle.close(), (a, x))

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
            s.skip(holder, lock.acquire)
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
            s.skip(t, lock.acquire)
            s.raw(lock).release()
        self.assertEqual(result, [False])


class TestDriverNested(unittest.TestCase):
    """Driver handling for nested transactions under the tranche-2 model."""

    def test_driver_internal_old_nesting_modes_are_removed(self):
        s = Scenario()
        self.assertFalse(hasattr(s.Driver, 'nesting'))
        self.assertFalse(hasattr(s.Driver, 'reentered'))
        self.assertFalse(hasattr(s._core.Driver, 'nesting'))
        self.assertFalse(hasattr(s._core.Driver, 'reentered'))
        self.assertFalse(hasattr(s._core.Driver, 'active'))

    def test_finish_surfaces_nested_child(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            parent = s.transaction(t)
            d = s.Driver(t)
            d.scan(); d()
            self.assertIs(d.tx, parent)
            d.finish(); d()
            # finish predicts the tx exits; it's instead blocked on a child,
            # so it rests `unexpected` with the Nested edge in motivation and
            # leaves the tx as the parent (it doesn't select the child).
            self.assertIs(d.state, d.unexpected)
            self.assertTrue(any(isinstance(m, Nested) for m in d.motivation))
            self.assertIs(d.tx, parent)
            # descend explicitly: push makes the parent the scope floor,
            # scan finds the inner cond.wait child.
            d.push()
            d.scan(); d()
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, condition.wait)
            self.assertIs(d.tx.parent, parent)
            s._core.driver_finish_deep(d)
            d.pop()
            s.skip(t, lock.release)

    def test_scan_watch_target_surfaces_child_under_parent(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, condition.wait, State.BLOCKED))
            d = s.Driver(t)
            d.scan(base); d()
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, condition.wait)
            self.assertIs(d.tx.parent, base)
            s._core.driver_finish_deep(d)
            d.scan(); d()
            self.assertIs(d.state, d.success)
            d.finish(); d()

    def test_base_tx_driver_impasse_on_blanket_parked_base(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            base = s.transaction(t)
            d = s.Driver(t, base)
            # scan below a blanket-parked floor isn't an error -- it's a valid
            # observation (the floor parked), reported as unexpected.
            d.scan(); d()
            self.assertIs(d.state, d.unexpected)
            d.close()

            d2 = s.Driver(t)
            d2.scan(); d2()
            self.assertIs(d2.state, d2.success)
            self.assertIs(d2.tx, base)
            d2.close()

    def test_base_tx_driver_surfaces_and_drives_child(self):
        s = Scenario()
        lock = s.Lock()
        condition = s.Condition(lock)
        seen = []

        def waiter():
            lock.acquire()
            seen.append(condition.wait_for(lambda: False, timeout=-1))
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            s.wait(t)
            base = s.transaction(t)
            d = s.Driver(t, base)
            base.unblock()
            s.wait(Call(t, condition.wait, State.BLOCKED))
            d.scan(); d()
            self.assertIs(d.state, d.success)
            self.assertEqual(d.tx.method, condition.wait)
            s._core.driver_finish_deep(d)
            self.assertTrue(base.done)
            s.skip(t, lock.release)

        self.assertEqual(seen, [False])


    def test_skip_with_base_tx_strict(self):
        """s.skip((t, base), m1, m2) is strict on base's children: base's
        next child must be m1 (driven to terminal), then the next must
        be m2.  Here base is a wait_for whose predicate calls
        lockB.acquire() then lockB.release(); naming both drives both to
        terminal.  skip never touches base."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire()
                lockB.release()
                return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)        # the wait_for tx

            # Put the worker inside base's body (test stands in for
            # whatever owns base); wait until the first nested child
            # appears so base is method-controlled, not blanket-parked.
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.skip((t, base), lockB.acquire, lockB.release)
            self.assertEqual(r[t].method, lockB.release)
            self.assertTrue(r[t].done)


    def test_skip_with_base_tx_strict_rejects_intervening_child(self):
        """skip is strict on base's children too: skipping straight to
        lockB.release raises, because base's next child is lockB.acquire.
        To step past an intervening child you use park, not skip."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire()
                lockB.release()
                return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(RuntimeError, "expected"):
                s.skip((t, base), lockB.release)


    def test_skip_with_base_tx_impasse_when_base_blanket_parked(self):
        """If base is blanket-parked (BLOCKED) when skip starts, the
        nested method is unreachable -- the Driver lands IMPASSE and
        skip raises RuntimeError rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        lockC = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)        # wait_for, BLOCKED (blanket-parked)

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.skip((t, base), lockC.acquire)


    def test_park_with_base_tx_parks_nested_method(self):
        """s.park((t, base), method) skips over base's children until one
        matches `method`, then leaves that child parked at BLOCKED.
        This is how you park in a child tx (skip is strict and can't
        step past the intervening lockB.acquire)."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire()
                lockB.release()
                return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.park((t, base), lockB.release)
            self.assertEqual(r[t].method, lockB.release)
            self.assertEqual(r[t].state, State.BLOCKED)

    def test_park_with_base_tx_first_child(self):
        """park(t, base, method) where method IS base's first child: no
        skipping needed, the child is left parked at BLOCKED.  Releasing
        it (here via __exit__) lets the worker finish its body."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)
        order = []

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); order.append('pred'); return True
            condition.wait_for(pred, timeout=-1)
            order.append('body')
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.park((t, base), lockB.acquire)
            self.assertEqual(r[t].method, lockB.acquire)
            self.assertEqual(r[t].state, State.BLOCKED)
            self.assertEqual(order, [])   # body not run while parked

        # __exit__ drove the parked child and the rest of base to terminal.
        self.assertEqual(order, ['pred', 'body'])

    def test_park_with_base_tx_base_exits_before_method(self):
        """park is lenient -- it skips base's children looking for the
        named method.  If base produces all its children and terminates
        without ever yielding that method, park raises rather than
        hanging (the named foreign lock is never a child of base)."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        lockZ = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(
                    RuntimeError, "base tx ended before reaching"):
                s.park((t, base), lockZ.acquire)

    def test_park_with_base_tx_impasse_when_base_blanket_parked(self):
        """If base is blanket-parked (BLOCKED) when park starts, it can
        produce no children, so the nested method is unreachable: the
        Driver lands IMPASSE and park raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        lockC = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)   # wait_for, BLOCKED (blanket-parked)

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.park((t, base), lockC.acquire)

    def test_park_with_base_tx_rejects_multiple_methods(self):
        """park takes exactly one method per thread, base or not."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(ValueError, "exactly one method"):
                s.park((t, base), lockB.acquire, lockB.release)
            # leave t parked at a real child so __exit__ can drain it.
            s.park((t, base), lockB.release)



class TestPause(unittest.TestCase):
    """Tests for Scenario.pause -- the PAUSED-state sibling of skip."""

    def test_pause_drives_to_paused(self):
        """pause runs the named call but lands it at PAUSED with the
        scheduler pause flag set, so it can be released later."""
        s = Scenario()
        lock = s.Lock()
        order = []
        def worker():
            lock.acquire()
            order.append('a')
            lock.release()
        with s:
            t = s.thread(worker)
            r = s.pause(t, lock.acquire)
            self.assertEqual(r[t].method, lock.acquire)
            self.assertEqual(r[t].state, State.PAUSED)
            # nothing has run past the pause yet
            self.assertEqual(order, [])
            r[t].unpause()            # release the scheduler pause
        self.assertEqual(order, ['a'])

    def test_pause_is_strict_on_divergence(self):
        """pause, like skip, is strict: the named call must be next."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire()
            lock.release()
        with s:
            t = s.thread(worker)
            with self.assertRaisesRegex(RuntimeError, "expected"):
                s.pause(t, lock.release)   # next is acquire, not release

    def test_pause_raises_when_method_never_reached(self):
        """pause is strict, so a thread that pushes no transaction at
        all terminates before its named call is reached, and pause
        raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            pass   # never calls anything regulated
        with s:
            t = s.thread(worker)
            with self.assertRaisesRegex(RuntimeError, "terminated before reaching"):
                s.pause(t, lock.acquire)

    def test_pause_multi_thread_concurrent(self):
        """pause drives several threads concurrently, each to PAUSED."""
        s = Scenario()
        lockA = s.Lock()
        lockB = s.Lock()
        def a():
            lockA.acquire(); lockA.release()
        def b():
            lockB.acquire(); lockB.release()
        with s:
            A = s.thread(a)
            B = s.thread(b)
            r = s.pause(A, lockA.acquire, B, lockB.acquire)
            self.assertEqual(r[A].state, State.PAUSED)
            self.assertEqual(r[B].state, State.PAUSED)
            r[A].unpause()
            r[B].unpause()

    def test_pause_with_base_tx(self):
        """pause(t, base, method) is strict on base's children: drives
        base's next child to PAUSED, never touching base."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire()
                lockB.release()
                return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.pause((t, base), lockB.acquire)
            self.assertEqual(r[t].method, lockB.acquire)
            self.assertEqual(r[t].state, State.PAUSED)
            r[t].unpause()

    def test_pause_with_base_tx_resume_completes(self):
        """pause(t, base, method) lands base's next child at PAUSED;
        clearing the scheduler pause flag lets it run on and the worker
        finishes its body within base."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)
        order = []

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); order.append('pred'); return True
            condition.wait_for(pred, timeout=-1)
            order.append('body')
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.pause((t, base), lockB.acquire)
            self.assertEqual(r[t].state, State.PAUSED)
            self.assertEqual(order, [])
            r[t].unpause()

        self.assertEqual(order, ['pred', 'body'])

    def test_pause_with_base_tx_rejects_intervening_child(self):
        """pause is strict on base's children too: the named call must
        be base's NEXT child.  Asking for lockB.release while the next
        child is lockB.acquire raises (use park to step past)."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(RuntimeError, "next child was"):
                s.pause((t, base), lockB.release)
            # leave t parked so __exit__ can drain it.
            s.park((t, base), lockB.release)

    def test_pause_with_base_tx_impasse_when_base_blanket_parked(self):
        """base blanket-parked at BLOCKED -> nested method unreachable ->
        IMPASSE -> pause raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        lockC = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.pause((t, base), lockC.acquire)

    def test_pause_with_base_tx_rejects_multiple_methods(self):
        """pause takes exactly one method per thread, base or not."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(ValueError, "exactly one method"):
                s.pause((t, base), lockB.acquire, lockB.release)
            s.park((t, base), lockB.release)


class TestBlock(unittest.TestCase):
    """Tests for Scenario.block -- the BLOCKED-state sibling of skip and
    pause.  Strict like them (the named call must be next) but leaves the
    call parked at BLOCKED, un-driven, rather than driving it to terminal
    (skip) or PAUSED (pause)."""

    def test_block_drives_to_blocked(self):
        """block runs nothing: it leaves the named call parked at BLOCKED.
        Unblocking it lets the worker run on."""
        s = Scenario()
        lock = s.Lock()
        order = []
        def worker():
            lock.acquire(); order.append('a'); lock.release()
        with s:
            t = s.thread(worker)
            r = s.block(t, lock.acquire)
            self.assertEqual(r[t].method, lock.acquire)
            self.assertEqual(r[t].state, State.BLOCKED)
            self.assertEqual(order, [])      # nothing ran -- acquire is still blocked
            s.skip(t, lock.acquire, lock.release)
            self.assertEqual(order, ['a'])   # driven on from the block

    def test_block_is_strict_on_divergence(self):
        """block, like skip and pause, is strict: the named call must be
        the thread's next transaction."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(worker)
            with self.assertRaisesRegex(RuntimeError, "next tx was"):
                s.block(t, lock.release)   # next is acquire, not release
            s.skip(t, lock.acquire, lock.release)

    def test_block_raises_when_method_never_reached(self):
        """A thread that pushes no transaction terminates before its
        named call is reached, so block raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            pass
        with s:
            t = s.thread(worker)
            with self.assertRaisesRegex(RuntimeError, "terminated before reaching"):
                s.block(t, lock.acquire)

    def test_block_multi_thread_concurrent(self):
        """block drives several threads concurrently, each to BLOCKED."""
        s = Scenario()
        lockA = s.Lock()
        lockB = s.Lock()
        def a():
            lockA.acquire(); lockA.release()
        def b():
            lockB.acquire(); lockB.release()
        with s:
            A = s.thread(a)
            B = s.thread(b)
            r = s.block(A, lockA.acquire, B, lockB.acquire)
            self.assertEqual(r[A].state, State.BLOCKED)
            self.assertEqual(r[B].state, State.BLOCKED)
            s.skip(A, lockA.acquire, lockA.release)
            s.skip(B, lockB.acquire, lockB.release)

    def test_block_rejects_multiple_methods(self):
        """block takes exactly one method per thread."""
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(worker)
            with self.assertRaisesRegex(ValueError, "exactly one method"):
                s.block(t, lock.acquire, lock.release)
            s.skip(t, lock.acquire, lock.release)

    def test_block_with_base_tx(self):
        """block(t, base, method) is strict on base's children: it leaves
        base's next child parked at BLOCKED.  Resuming runs the worker on
        through the rest of base."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)
        order = []

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); order.append('pred'); return True
            condition.wait_for(pred, timeout=-1)
            order.append('body')
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            r = s.block((t, base), lockB.acquire)
            self.assertEqual(r[t].method, lockB.acquire)
            self.assertEqual(r[t].state, State.BLOCKED)
            self.assertEqual(order, [])

        self.assertEqual(order, ['pred', 'body'])

    def test_block_with_base_tx_rejects_intervening_child(self):
        """block is strict on base's children: the named call must be
        base's NEXT child."""
        s = Scenario()
        lock = s.Lock()
        lockB = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            def pred():
                lockB.acquire(); lockB.release(); return True
            condition.wait_for(pred, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)
            base.unblock()
            s.wait(Call(t, lockB.acquire, State.BLOCKED))

            with self.assertRaisesRegex(RuntimeError, "next child was"):
                s.block((t, base), lockB.release)
            s.park((t, base), lockB.release)

    def test_block_with_base_tx_impasse_when_base_blanket_parked(self):
        """base blanket-parked at BLOCKED -> nested method unreachable ->
        IMPASSE -> block raises rather than hanging."""
        s = Scenario()
        lock = s.Lock()
        lockC = s.Lock()
        condition = s.Condition(lock)

        def waiter():
            lock.acquire()
            condition.wait_for(lambda: False, timeout=-1)
            lock.release()

        with s:
            t = s.thread(waiter)
            s.skip(t, lock.acquire)
            base = s.transaction(t)

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.block((t, base), lockC.acquire)


class TestAssignBaseTx(unittest.TestCase):
    """base_tx support for Lock.assign: each thread (releaser and/or
    acquirer) may be followed by a base tx, scoping that thread's driver
    to its subtree under base -- the release / acquire must then be
    base's next surfaced child."""

    def test_assign_base_tx_on_acquirer(self):
        """The acquirer takes the handed-off lock inside its base tx (a
        wait_for predicate); the releaser releases at top level."""
        s = Scenario()
        L = s.Lock()
        lockA = s.Lock()
        condA = s.Condition(lockA)
        order = []
        def releaser():
            L.acquire(); order.append('R-acq')
            L.release(); order.append('R-rel')
        def acquirer():
            lockA.acquire()
            def pred():
                L.acquire(); order.append('A-acq'); return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
            L.release(); order.append('A-rel')
        with s:
            R = s.thread(releaser)
            A = s.thread(acquirer)
            s.skip(R, L.acquire)                 # R holds L
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A)             # the wait_for tx
            baseA.unblock()
            s.wait(Call(A, L.acquire, State.BLOCKED))

            r = s.api(L).assign(R, (A, baseA))
            self.assertEqual(list(r), [A])
            self.assertEqual(order, ['R-acq', 'R-rel', 'A-acq'])
            s.skip(A, lockA.release, L.release)
        self.assertEqual(order, ['R-acq', 'R-rel', 'A-acq', 'A-rel'])

    def test_assign_base_tx_on_releaser(self):
        """The releaser releases the lock inside its base tx; the
        acquirer takes it at top level."""
        s = Scenario()
        L = s.Lock()
        lockR = s.Lock()
        condR = s.Condition(lockR)
        order = []
        def releaser():
            L.acquire()
            lockR.acquire()
            def pred():
                L.release(); order.append('R-rel'); return True
            condR.wait_for(pred, timeout=-1)
            lockR.release()
        def acquirer():
            L.acquire(); order.append('A-acq'); L.release()
        with s:
            R = s.thread(releaser)
            A = s.thread(acquirer)
            s.skip(R, L.acquire)                 # R holds L
            s.skip(R, lockR.acquire)
            baseR = s.transaction(R)
            baseR.unblock()
            s.wait(Call(R, L.release, State.BLOCKED))

            r = s.api(L).assign((R, baseR), A)
            self.assertEqual(order, ['R-rel', 'A-acq'])
            s.skip(R, lockR.release)
            s.skip(A, L.release)

    def test_assign_base_tx_on_both(self):
        """Both releaser and acquirer act within their own base txs."""
        s = Scenario()
        L = s.Lock()
        lockR = s.Lock(); condR = s.Condition(lockR)
        lockA = s.Lock(); condA = s.Condition(lockA)
        order = []
        def releaser():
            L.acquire()
            lockR.acquire()
            def predR():
                L.release(); order.append('R-rel'); return True
            condR.wait_for(predR, timeout=-1)
            lockR.release()
        def acquirer():
            lockA.acquire()
            def predA():
                L.acquire(); order.append('A-acq'); return True
            condA.wait_for(predA, timeout=-1)
            lockA.release()
            L.release()
        with s:
            R = s.thread(releaser)
            A = s.thread(acquirer)
            s.skip(R, L.acquire)
            s.skip(R, lockR.acquire)
            baseR = s.transaction(R); baseR.unblock()
            s.wait(Call(R, L.release, State.BLOCKED))
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A); baseA.unblock()
            s.wait(Call(A, L.acquire, State.BLOCKED))

            r = s.api(L).assign((R, baseR), (A, baseA))
            self.assertEqual(order, ['R-rel', 'A-acq'])
            s.skip(R, lockR.release)
            s.skip(A, lockA.release, L.release)

    def test_assign_base_tx_lone_acquirer(self):
        """A lone acquirer (no releaser, lock unheld) may take the lock
        inside its base tx."""
        s = Scenario()
        L = s.Lock()
        lockA = s.Lock()
        condA = s.Condition(lockA)
        order = []
        def acquirer():
            lockA.acquire()
            def pred():
                L.acquire(); order.append('A-acq'); return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
            L.release()
        with s:
            A = s.thread(acquirer)
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A); baseA.unblock()
            s.wait(Call(A, L.acquire, State.BLOCKED))

            r = s.api(L).assign((A, baseA))
            self.assertEqual(list(r), [A])
            self.assertEqual(order, ['A-acq'])
            s.skip(A, lockA.release, L.release)

    def test_assign_base_tx_impasse_when_base_blanket_parked(self):
        """If the acquirer's base is blanket-parked, its acquire is
        unreachable -- assign raises rather than hanging."""
        s = Scenario()
        L = s.Lock()
        lockA = s.Lock()
        condA = s.Condition(lockA)
        def releaser():
            L.acquire(); L.release()
        def acquirer():
            lockA.acquire()
            condA.wait_for(lambda: False, timeout=-1)
            lockA.release()
        with s:
            R = s.thread(releaser)
            A = s.thread(acquirer)
            s.skip(R, L.acquire)
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A)      # wait_for, blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.api(L).assign(R, (A, baseA))

    def test_assign_too_many_threads(self):
        """assign accepts at most a releaser and an acquirer."""
        s = Scenario()
        L = s.Lock()
        def w():
            L.acquire(); L.release()
        with s:
            a = s.thread(w); b = s.thread(w); c = s.thread(w)
            with self.assertRaisesRegex(ValueError, "at most a releaser"):
                s.api(L).assign(a, b, c)
            s.skip(a, L.acquire, L.release)
            s.skip(b, L.acquire, L.release)
            s.skip(c, L.acquire, L.release)


class TestDriverEquality(unittest.TestCase):
    """Driver.__eq__ / __ne__ / __hash__: two Drivers wrapping the same
    thread compare and hash equal; drivers for different threads do not;
    comparison with a non-Driver falls back to identity (not equal)."""

    def test_driver_equality_and_hash(self):
        s = Scenario()
        lock = s.Lock()
        def worker():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(worker)
            u = s.thread(worker)
            Core = s._core
            d1 = Core.Driver(t)
            d2 = Core.Driver(t)        # same thread, distinct object
            d3 = Core.Driver(u)        # different thread
            self.assertEqual(d1, d2)
            self.assertFalse(d1 != d2)
            self.assertEqual(hash(d1), hash(d2))
            self.assertNotEqual(d1, d3)
            self.assertTrue(d1 != d3)
            self.assertNotEqual(d1, t)              # non-Driver -> not equal
            self.assertFalse(d1 == object())        # NotImplemented -> identity
            self.assertEqual(len({d1, d2, d3}), 2)  # set dedups by thread
            s.skip(t, lock.acquire, lock.release)
            s.skip(u, lock.acquire, lock.release)

    def test_cycle_rejects_duplicate_thread_deterministically(self):
        """A cycle naming the same thread twice raises ValueError up
        front (the CycleBase seen-set guard), never the internal Dispatch
        signal-collision assert.  Repeated to guard against the prior
        load-dependent race."""
        for _ in range(20):
            s = Scenario()
            ev = s.Event()
            def waiter():
                ev.wait()
            def setter():
                ev.set()
            with s:
                a = s.thread(waiter)
                x = s.thread(setter)
                with self.assertRaises(ValueError):
                    s.api(ev).cycle(a, a)
                c = s.api(ev).cycle(a, x)
                c.wake(a)
                c.close()


class TestRelayBaseTx(unittest.TestCase):
    """base_tx support for Lock.relay: each participant thread may be
    followed by a base tx, scoping its driver to its subtree under base.
    A base applies to any position -- the hot-start initial-releaser,
    the last acquirer, AND a middle thread that both acquires and
    releases.  A middle thread's acquire and release happen in
    consecutive relay iterations; its base stays live across the relay
    yield (the predicate is paused mid-flight) and its reused driver
    finds the release as the next child on reactivation."""

    def test_relay_base_tx_on_acquirer(self):
        """The lone acquirer takes the lock inside its base tx."""
        s = Scenario()
        L = s.Lock()
        lockA = s.Lock()
        condA = s.Condition(lockA)
        order = []
        def initial_fn():
            L.acquire(); order.append('I-acq')
            L.release(); order.append('I-rel')
        def acq():
            lockA.acquire()
            def pred():
                L.acquire(); order.append('A-acq'); return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
            L.release(); order.append('A-rel')
        with s:
            I = s.thread(initial_fn)
            A = s.thread(acq)
            s.skip(I, L.acquire)
            s.block(I, L.release)                 # hot-start: parked at release
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A); baseA.unblock()
            s.wait(Call(A, L.acquire, State.BLOCKED))

            got = list(s.api(L).relay(I, (A, baseA)))
            self.assertEqual(got, [A])
            self.assertEqual(order, ['I-acq', 'I-rel', 'A-acq'])
            s.skip(A, lockA.release, L.release)
        self.assertEqual(order, ['I-acq', 'I-rel', 'A-acq', 'A-rel'])

    def test_relay_base_tx_on_initial_releaser(self):
        """The hot-start initial releases the lock inside its base tx."""
        s = Scenario()
        L = s.Lock()
        lockI = s.Lock()
        condI = s.Condition(lockI)
        order = []
        def initial_fn():
            L.acquire()
            lockI.acquire()
            def pred():
                L.release(); order.append('I-rel'); return True
            condI.wait_for(pred, timeout=-1)
            lockI.release()
        def acq():
            L.acquire(); order.append('A-acq'); L.release()
        with s:
            I = s.thread(initial_fn)
            A = s.thread(acq)
            s.skip(I, L.acquire)
            s.skip(I, lockI.acquire)
            baseI = s.transaction(I); baseI.unblock()
            s.wait(Call(I, L.release, State.BLOCKED))
            s.block(A, L.acquire)

            got = list(s.api(L).relay((I, baseI), A))
            self.assertEqual(got, [A])
            self.assertEqual(order, ['I-rel', 'A-acq'])
            s.skip(I, lockI.release)

    def test_relay_base_tx_on_both_endpoints(self):
        """A 3-hop relay with bases on the initial-releaser and the last
        acquirer; the middle thread runs at top level."""
        s = Scenario()
        L = s.Lock()
        lockI = s.Lock(); condI = s.Condition(lockI)
        lockB = s.Lock(); condB = s.Condition(lockB)
        order = []
        def initial_fn():
            L.acquire()
            lockI.acquire()
            def pred():
                L.release(); order.append('I-rel'); return True
            condI.wait_for(pred, timeout=-1)
            lockI.release()
        def mid():
            L.acquire(); order.append('M-acq')
            L.release(); order.append('M-rel')
        def last():
            lockB.acquire()
            def pred():
                L.acquire(); order.append('B-acq'); return True
            condB.wait_for(pred, timeout=-1)
            lockB.release()
            L.release(); order.append('B-rel')
        with s:
            I = s.thread(initial_fn)
            M = s.thread(mid)
            B = s.thread(last)
            s.skip(I, L.acquire)
            s.skip(I, lockI.acquire)
            baseI = s.transaction(I); baseI.unblock()
            s.wait(Call(I, L.release, State.BLOCKED))
            s.block(M, L.acquire)
            s.skip(B, lockB.acquire)
            baseB = s.transaction(B); baseB.unblock()
            s.wait(Call(B, L.acquire, State.BLOCKED))

            got = list(s.api(L).relay((I, baseI), M, (B, baseB)))
            self.assertEqual(got, [M, B])
            self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'B-acq'])
            s.skip(B, lockB.release, L.release)
        self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'B-acq', 'B-rel'])

    def test_relay_base_tx_impasse_when_base_blanket_parked(self):
        """If the last acquirer's base is blanket-parked, its acquire is
        unreachable -- relay raises rather than hanging."""
        s = Scenario()
        L = s.Lock()
        lockB = s.Lock()
        condB = s.Condition(lockB)
        def initial_fn():
            L.acquire(); L.release()
        def last():
            lockB.acquire()
            condB.wait_for(lambda: False, timeout=-1)
            lockB.release()
        with s:
            I = s.thread(initial_fn)
            B = s.thread(last)
            s.skip(I, L.acquire)
            s.block(I, L.release)
            s.skip(B, lockB.acquire)
            baseB = s.transaction(B)      # blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                list(s.api(L).relay(I, (B, baseB)))

    def test_relay_base_tx_on_middle_thread(self):
        """A 3-hop relay where the MIDDLE thread runs its acquire AND
        release inside one base tx.  The acquire and release land in
        consecutive relay iterations; the base stays live across the
        relay yield, so the reused driver finds the release as the next
        child."""
        s = Scenario()
        L = s.Lock()
        lockM = s.Lock(); condM = s.Condition(lockM)
        order = []
        def initial_fn():
            L.acquire(); L.release(); order.append('I-rel')
        def mid():
            lockM.acquire()
            def pred():
                L.acquire(); order.append('M-acq')
                L.release(); order.append('M-rel')
                return True
            condM.wait_for(pred, timeout=-1)
            lockM.release()
        def last():
            L.acquire(); order.append('C-acq'); L.release()
        with s:
            I = s.thread(initial_fn)
            M = s.thread(mid)
            C = s.thread(last)
            s.skip(I, L.acquire)
            s.block(I, L.release)
            s.skip(M, lockM.acquire)
            baseM = s.transaction(M); baseM.unblock()
            s.wait(Call(M, L.acquire, State.BLOCKED))
            s.block(C, L.acquire)

            got = list(s.api(L).relay(I, (M, baseM), C))
            self.assertEqual(got, [M, C])
            self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'C-acq'])
            s.skip(M, lockM.release)
            s.skip(C, L.release)
        self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'C-acq'])

    def test_relay_base_tx_all_three_under_base(self):
        """A 3-hop relay with every participant -- initial releaser,
        middle, and last acquirer -- under its own base tx."""
        s = Scenario()
        L = s.Lock()
        lockI = s.Lock(); condI = s.Condition(lockI)
        lockM = s.Lock(); condM = s.Condition(lockM)
        lockC = s.Lock(); condC = s.Condition(lockC)
        order = []
        def initial_fn():
            L.acquire()
            lockI.acquire()
            def pred():
                L.release(); order.append('I-rel'); return True
            condI.wait_for(pred, timeout=-1)
            lockI.release()
        def mid():
            lockM.acquire()
            def pred():
                L.acquire(); order.append('M-acq')
                L.release(); order.append('M-rel')
                return True
            condM.wait_for(pred, timeout=-1)
            lockM.release()
        def last():
            lockC.acquire()
            def pred():
                L.acquire(); order.append('C-acq'); L.release(); return True
            condC.wait_for(pred, timeout=-1)
            lockC.release()
        with s:
            I = s.thread(initial_fn)
            M = s.thread(mid)
            C = s.thread(last)
            s.skip(I, L.acquire)
            s.skip(I, lockI.acquire)
            baseI = s.transaction(I); baseI.unblock()
            s.wait(Call(I, L.release, State.BLOCKED))
            s.skip(M, lockM.acquire)
            baseM = s.transaction(M); baseM.unblock()
            s.wait(Call(M, L.acquire, State.BLOCKED))
            s.skip(C, lockC.acquire)
            baseC = s.transaction(C); baseC.unblock()
            s.wait(Call(C, L.acquire, State.BLOCKED))

            got = list(s.api(L).relay((I, baseI), (M, baseM), (C, baseC)))
            self.assertEqual(got, [M, C])
            self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'C-acq'])
            s.skip(I, lockI.release)
            s.skip(M, lockM.release)
            s.skip(C, L.release)        # final acquirer's release (base child) is undriven
            s.skip(C, lockC.release)
        self.assertEqual(order, ['I-rel', 'M-acq', 'M-rel', 'C-acq'])


class TestAllocateBaseTx(unittest.TestCase):
    """base_tx support for Semaphore.allocate: each participant thread
    may be followed by a base tx, scoping its driver to its subtree
    under base.  Also covers the deterministic duplicate-thread guard."""

    def test_allocate_base_tx_on_acquirer(self):
        """A lone acquirer takes the semaphore inside its base tx."""
        s = Scenario()
        sem = s.Semaphore(2)
        lockA = s.Lock()
        condA = s.Condition(lockA)
        order = []
        def acq():
            lockA.acquire()
            def pred():
                sem.acquire(); order.append('acq'); return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
            sem.release(); order.append('rel')
        with s:
            A = s.thread(acq)
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A); baseA.unblock()
            s.wait(Call(A, sem.acquire, State.BLOCKED))

            got = list(s.api(sem).allocate((A, baseA)))
            self.assertEqual(got, [A])
            self.assertEqual(order, ['acq'])
            s.skip(A, lockA.release, sem.release)
        self.assertEqual(order, ['acq', 'rel'])

    def test_allocate_base_tx_mixed_batch(self):
        """A batch mixing a release thread (base) and acquirers (one
        with a base, one at top level), driven in spec order."""
        s = Scenario()
        sem = s.Semaphore(1)
        lockR = s.Lock(); condR = s.Condition(lockR)
        lockB = s.Lock(); condB = s.Condition(lockB)
        order = []
        def rel():
            lockR.acquire()
            def pred():
                sem.release(); order.append('R-rel'); return True
            condR.wait_for(pred, timeout=-1)
            lockR.release()
        def a():
            sem.acquire(); order.append('A-acq'); sem.release()
        def b():
            lockB.acquire()
            def pred():
                sem.acquire(); order.append('B-acq'); return True
            condB.wait_for(pred, timeout=-1)
            lockB.release()
            sem.release(); order.append('B-rel')
        with s:
            R = s.thread(rel); A = s.thread(a); B = s.thread(b)
            s.skip(R, lockR.acquire)
            baseR = s.transaction(R); baseR.unblock()
            s.wait(Call(R, sem.release, State.BLOCKED))
            s.block(A, sem.acquire)
            s.skip(B, lockB.acquire)
            baseB = s.transaction(B); baseB.unblock()
            s.wait(Call(B, sem.acquire, State.BLOCKED))

            got = list(s.api(sem).allocate((R, baseR), A, (B, baseB)))
            self.assertEqual(got, [A, B])
            self.assertEqual(order, ['R-rel', 'A-acq', 'B-acq'])
            s.skip(B, lockB.release, sem.release)
        self.assertEqual(order, ['R-rel', 'A-acq', 'B-acq', 'B-rel'])

    def test_allocate_base_tx_impasse_when_base_blanket_parked(self):
        """An acquirer whose base is blanket-parked can't reach its
        acquire -- allocate raises rather than hanging."""
        s = Scenario()
        sem = s.Semaphore(2)
        lockA = s.Lock()
        condA = s.Condition(lockA)
        def acq():
            lockA.acquire()
            condA.wait_for(lambda: False, timeout=-1)
            lockA.release()
        with s:
            A = s.thread(acq)
            s.skip(A, lockA.acquire)
            baseA = s.transaction(A)      # blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                list(s.api(sem).allocate((A, baseA)))

    def test_allocate_allows_duplicate_thread_specs(self):
        """A later occurrence of the same thread starts after the
        earlier occurrence finishes its semaphore transaction."""
        s = Scenario()
        sem = s.Semaphore(1)
        order = []
        def w():
            sem.acquire(); order.append('acquire')
            sem.release(); order.append('release')
        with s:
            a = s.thread(w)
            self.assertEqual(list(s.api(sem).allocate(a, a)), [a])

        self.assertEqual(order, ['acquire', 'release'])


class TestEventCycleBaseTx(unittest.TestCase):
    """base_tx support for Event.cycle: a participant (waiter or setter)
    may be followed by a base tx, scoping its driver to its subtree
    under base -- its event.wait / event.set must surface as base's
    child."""

    def test_event_cycle_base_tx_on_waiter(self):
        """The waiter does event.wait inside its base tx."""
        s = Scenario()
        ev = s.Event()
        lockW = s.Lock()
        condW = s.Condition(lockW)
        order = []
        def waiter():
            lockW.acquire()
            def pred():
                ev.wait(); order.append('W-wait'); return True
            condW.wait_for(pred, timeout=-1)
            lockW.release(); order.append('W-done')
        def setter():
            ev.set(); order.append('S-set')
        with s:
            W = s.thread(waiter)
            S = s.thread(setter)
            s.skip(W, lockW.acquire)
            baseW = s.transaction(W); baseW.unblock()
            s.wait(Call(W, ev.wait, State.BLOCKED))

            c = s.api(ev).cycle((W, baseW), S)
            self.assertEqual(set(c.ready), {W, S})
            c.wake(W)
            c.close()
            self.assertEqual(order, ['W-wait', 'S-set'])
            s.skip(W, lockW.release)
        self.assertEqual(order, ['W-wait', 'S-set', 'W-done'])

    def test_event_cycle_base_tx_on_setter(self):
        """The setter does event.set inside its base tx."""
        s = Scenario()
        ev = s.Event()
        lockS = s.Lock()
        condS = s.Condition(lockS)
        order = []
        def waiter():
            ev.wait(); order.append('W-wait')
        def setter():
            lockS.acquire()
            def pred():
                ev.set(); order.append('S-set'); return True
            condS.wait_for(pred, timeout=-1)
            lockS.release(); order.append('S-done')
        with s:
            W = s.thread(waiter)
            S = s.thread(setter)
            s.skip(S, lockS.acquire)
            baseS = s.transaction(S); baseS.unblock()
            s.wait(Call(S, ev.set, State.BLOCKED))

            c = s.api(ev).cycle(W, (S, baseS))
            c.wake(W)
            c.close()
            self.assertEqual(order, ['W-wait', 'S-set'])
            s.skip(S, lockS.release)
        self.assertEqual(order, ['W-wait', 'S-set', 'S-done'])

    def test_event_cycle_base_tx_impasse_when_base_blanket_parked(self):
        """A waiter whose base is blanket-parked can't reach its
        event.wait -- cycle construction raises rather than hanging."""
        s = Scenario()
        ev = s.Event()
        lockW = s.Lock()
        condW = s.Condition(lockW)
        def waiter():
            lockW.acquire()
            condW.wait_for(lambda: False, timeout=-1)
            lockW.release()
        def setter():
            ev.set()
        with s:
            W = s.thread(waiter)
            S = s.thread(setter)
            s.skip(W, lockW.acquire)
            baseW = s.transaction(W)      # blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.api(ev).cycle((W, baseW), S)


class TestBarrierCycleBaseTx(unittest.TestCase):
    """base_tx support for Barrier.cycle: a participant may be followed
    by a base tx; its barrier.wait must surface as base's child."""

    def test_barrier_cycle_base_tx_on_waiter(self):
        """A waiter does barrier.wait inside its base tx."""
        s = Scenario()
        br = s.Barrier(2)
        lockW = s.Lock()
        condW = s.Condition(lockW)
        order = []
        def w1():
            lockW.acquire()
            def pred():
                br.wait(); order.append('w1'); return True
            condW.wait_for(pred, timeout=-1)
            lockW.release(); order.append('w1-done')
        def w2():
            br.wait(); order.append('w2')
        with s:
            W1 = s.thread(w1)
            W2 = s.thread(w2)
            s.skip(W1, lockW.acquire)
            baseW = s.transaction(W1); baseW.unblock()
            s.wait(Call(W1, br.wait, State.BLOCKED))

            c = s.api(br).cycle((W1, baseW), W2)
            self.assertEqual(set(c.ready), {W1, W2})
            c.wake(W1)
            c.close()
            self.assertEqual(sorted(order), ['w1', 'w2'])
            s.skip(W1, lockW.release)
        self.assertEqual(sorted(order), ['w1', 'w1-done', 'w2'])

    def test_barrier_cycle_base_tx_impasse_when_base_blanket_parked(self):
        """A waiter whose base is blanket-parked can't reach its
        barrier.wait -- cycle construction raises rather than hanging.
        (barrier.wait sits inside the base so __exit__ can still drive
        both threads to the barrier afterward.)"""
        s = Scenario()
        br = s.Barrier(2)
        lockW = s.Lock()
        condW = s.Condition(lockW)
        def w1():
            lockW.acquire()
            def pred():
                br.wait(); return True
            condW.wait_for(pred, timeout=-1)
            lockW.release()
        def w2():
            br.wait()
        with s:
            W1 = s.thread(w1)
            W2 = s.thread(w2)
            s.skip(W1, lockW.acquire)
            baseW = s.transaction(W1)      # blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.api(br).cycle((W1, baseW), W2)


class TestConditionCycleBaseTx(unittest.TestCase):
    """base_tx support for Condition.cycle: a participant may run its
    whole UL participation -- lock.acquire, cond.wait / cond.notify,
    lock.release -- inside a base tx (here, a wait_for predicate on a
    separate condition, the 'frame of reference').  The cycle drives
    that participation as base's children."""

    def park_waiter(self, scenario, condition, lock, thread):
        scenario.skip(thread, lock.acquire)
        scenario.wait(thread)
        tx = scenario.transaction(thread)
        tx.unblock()
        scenario.wait(Waiting(tx))
        return tx

    def into_base(self, scenario, thread, outer_lock, lock):
        """Drive `thread` into its wait_for base, parked at the cycle's
        underlying-lock acquire (the first child of base).  Returns the
        base tx."""
        scenario.skip(thread, outer_lock.acquire)
        base = scenario.transaction(thread)
        base.unblock()
        scenario.wait(Call(thread, lock.acquire, State.BLOCKED))
        return base

    def test_condition_cycle_base_tx_on_waiter(self):
        """The waiter runs lock.acquire / cond.wait / lock.release inside
        its base; the waker is a plain top-level notifier."""
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockA = s.Lock(); condA = s.Condition(lockA)
        log = []
        def waiter():
            lockA.acquire()
            def pred():
                lock.acquire()
                condition.wait(); log.append('A-woke')
                lock.release()
                return True
            condA.wait_for(pred, timeout=-1)
            lockA.release(); log.append('A-done')
        def notifier():
            lock.acquire()
            condition.notify(); log.append('X-notify')
            lock.release()
        with s:
            a = s.thread(waiter); x = s.thread(notifier)
            baseA = self.into_base(s, a, lockA, lock)
            # cycle drives a from UL.acquire through wait, then x notifies
            c = s.api(condition).cycle((a, baseA), x)
            self.assertEqual(list(c.ready), [a])
            self.assertEqual(c.wake(a), (a,))
            c.close()
            self.assertEqual(log, ['X-notify', 'A-woke'])
        self.assertEqual(log, ['X-notify', 'A-woke', 'A-done'])

    def test_condition_cycle_base_tx_on_waker(self):
        """The waker runs lock.acquire / cond.notify / lock.release inside
        its base; the waiter is a plain top-level waiter parked at
        WAITING."""
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockX = s.Lock(); condX = s.Condition(lockX)
        log = []
        def waiter():
            lock.acquire()
            condition.wait(); log.append('A-woke')
            lock.release()
        def notifier():
            lockX.acquire()
            def pred():
                lock.acquire()
                condition.notify(); log.append('X-notify')
                lock.release()
                return True
            condX.wait_for(pred, timeout=-1)
            lockX.release(); log.append('X-done')
        with s:
            a = s.thread(waiter); x = s.thread(notifier)
            self.park_waiter(s, condition, lock, a)
            baseX = self.into_base(s, x, lockX, lock)
            c = s.api(condition).cycle(a, (x, baseX))
            self.assertEqual(list(c.ready), [a])
            self.assertEqual(log, ['X-notify'])
            self.assertEqual(c.wake(a), (a,))
            c.close()
            self.assertEqual(log, ['X-notify', 'A-woke'])
        self.assertEqual(sorted(log), ['A-woke', 'X-done', 'X-notify'])

    def test_condition_cycle_base_tx_both_participants(self):
        """Both waiter and waker run their full participation under
        their own base."""
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockA = s.Lock(); condA = s.Condition(lockA)
        lockX = s.Lock(); condX = s.Condition(lockX)
        log = []
        def waiter():
            lockA.acquire()
            def pred():
                lock.acquire()
                condition.wait(); log.append('A-woke')
                lock.release()
                return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
        def notifier():
            lockX.acquire()
            def pred():
                lock.acquire()
                condition.notify(); log.append('X-notify')
                lock.release()
                return True
            condX.wait_for(pred, timeout=-1)
            lockX.release()
        with s:
            a = s.thread(waiter); x = s.thread(notifier)
            baseA = self.into_base(s, a, lockA, lock)
            baseX = self.into_base(s, x, lockX, lock)
            c = s.api(condition).cycle((a, baseA), (x, baseX))
            self.assertEqual(list(c.ready), [a])
            self.assertEqual(log, ['X-notify'])
            self.assertEqual(c.wake(a), (a,))
            c.close()
        self.assertEqual(log, ['X-notify', 'A-woke'])

    def test_condition_cycle_base_tx_impasse_when_base_blanket_parked(self):
        """A waker whose base is blanket-parked can't reach its
        participation -- cycle construction raises rather than hanging.
        The waiter is left genuinely WAITING, so (as with any aborted
        Condition cycle) teardown is a manual, deterministic drain:
        unblock the waker's base, sync on its UL acquire, then drive its
        notify to free the waiter."""
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockX = s.Lock(); condX = s.Condition(lockX)
        log = []
        def waiter():
            lock.acquire(); condition.wait(); log.append('A'); lock.release()
        def notifier():
            lockX.acquire()
            def pred():
                lock.acquire(); condition.notify(); lock.release(); return True
            condX.wait_for(pred, timeout=-1)
            lockX.release()
        with s:
            a = s.thread(waiter); x = s.thread(notifier)
            self.park_waiter(s, condition, lock, a)   # waiter genuinely WAITING
            s.skip(x, lockX.acquire)
            baseX = s.transaction(x)      # blanket-parked

            with self.assertRaisesRegex(RuntimeError, "blanket-parked"):
                s.api(condition).cycle(a, (x, baseX))

            # Manual drain: free the stranded waiter deterministically.
            baseX.unblock()
            s.wait(Call(x, lock.acquire, State.BLOCKED))
            s.skip(x, lock.acquire, condition.notify, lock.release)
            s.skip(x, lockX.release)
            s.skip(a, condition.wait, lock.release)
        self.assertEqual(log, ['A'])


class TestConditionCycleWaitingEntry(unittest.TestCase):
    """Condition.cycle handles a waiter passed in ALREADY at WAITING --
    whether it got there via a direct cond.wait or via cond.wait_for's
    nested wait, and whether or not it sits under a base tx.  (The
    scheduler/user drives the waiter to WAITING, then hands it to the
    cycle in that state.)"""

    def park_wait_for(self, s, condition, lock, t):
        """Park a top-level wait_for waiter at WAITING on its nested
        cond.wait (predicate already run, came up false)."""
        s.skip(t, lock.acquire)
        wf = s.transaction(t); wf.unblock()
        s.wait(Call(t, condition.wait, State.BLOCKED))
        nested = s.transaction(t); nested.unblock()
        s.wait(Waiting(nested))

    def test_direct_wait_no_base(self):
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        log = []
        def waiter():
            lock.acquire(); condition.wait(); log.append('A'); lock.release()
        def waker():
            lock.acquire(); condition.notify(); log.append('X'); lock.release()
        with s:
            a = s.thread(waiter); x = s.thread(waker)
            s.skip(a, lock.acquire); s.wait(a)
            wtx = s.transaction(a); wtx.unblock(); s.wait(Waiting(wtx))
            c = s.api(condition).cycle(a, x)
            self.assertEqual(c.wake(a), (a,))
            c.close()
        self.assertEqual(log, ['X', 'A'])

    def test_direct_wait_under_base(self):
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockA = s.Lock(); condA = s.Condition(lockA)
        log = []
        def waiter():
            lockA.acquire()
            def pred():
                lock.acquire(); condition.wait(); log.append('A'); lock.release()
                return True
            condA.wait_for(pred, timeout=-1)
            lockA.release()
        def waker():
            lock.acquire(); condition.notify(); log.append('X'); lock.release()
        with s:
            a = s.thread(waiter); x = s.thread(waker)
            s.skip(a, lockA.acquire)
            baseA = s.transaction(a); baseA.unblock()
            s.wait(Call(a, lock.acquire, State.BLOCKED)); s.skip(a, lock.acquire)
            s.wait(Call(a, condition.wait, State.BLOCKED))
            wtx = s.transaction(a); wtx.unblock(); s.wait(Waiting(wtx))
            c = s.api(condition).cycle((a, baseA), x)
            self.assertEqual(c.wake(a), (a,))
            c.close()
        self.assertEqual(log, ['X', 'A'])

    def test_wait_for_no_base(self):
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        log = []; flag = [False]
        def waiter():
            lock.acquire(); condition.wait_for(lambda: flag[0])
            log.append('A'); lock.release()
        def waker():
            lock.acquire(); flag[0] = True; condition.notify()
            log.append('X'); lock.release()
        with s:
            a = s.thread(waiter); x = s.thread(waker)
            self.park_wait_for(s, condition, lock, a)
            c = s.api(condition).cycle(a, x)
            self.assertEqual(c.wake(a), (a,))
            c.close()
        self.assertEqual(log, ['X', 'A'])

    def test_wait_for_under_base(self):
        s = Scenario()
        lock = s.Lock(); condition = s.Condition(lock)
        lockA = s.Lock(); condA = s.Condition(lockA)
        log = []; flag = [False]
        def waiter():
            lockA.acquire()
            def outer_pred():
                lock.acquire(); condition.wait_for(lambda: flag[0])
                log.append('A'); lock.release()
                return True
            condA.wait_for(outer_pred, timeout=-1)
            lockA.release()
        def waker():
            lock.acquire(); flag[0] = True; condition.notify()
            log.append('X'); lock.release()
        with s:
            a = s.thread(waiter); x = s.thread(waker)
            s.skip(a, lockA.acquire)
            baseA = s.transaction(a); baseA.unblock()
            s.wait(Call(a, lock.acquire, State.BLOCKED)); s.skip(a, lock.acquire)
            s.wait(Call(a, condition.wait_for, State.BLOCKED))
            wf = s.transaction(a); wf.unblock()
            s.wait(Call(a, condition.wait, State.BLOCKED))
            nested = s.transaction(a); nested.unblock(); s.wait(Waiting(nested))
            c = s.api(condition).cycle((a, baseA), x)
            self.assertEqual(c.wake(a), (a,))
            c.close()
        self.assertEqual(log, ['X', 'A'])


class TestWrongTypeRaisesTypeError(unittest.TestCase):
    """The mid- and high-level APIs consistently raise TypeError for a
    wrong-type argument, while keeping ValueError for duplicate threads."""

    def test_skip_wrong_type_after_thread(self):
        s = Scenario()
        lock = s.Lock()
        def w():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(w)
            with self.assertRaises(TypeError):
                s.skip(t, 42)
            s.skip(t, lock.acquire, lock.release)

    def test_park_wrong_type_after_thread(self):
        s = Scenario()
        lock = s.Lock()
        def w():
            lock.acquire(); lock.release()
        with s:
            t = s.thread(w)
            with self.assertRaises(TypeError):
                s.park(t, object())
            s.skip(t, lock.acquire, lock.release)

    def test_assign_wrong_type(self):
        s = Scenario()
        lock = s.Lock()
        def w():
            lock.acquire(); lock.release()
        with s:
            a = s.thread(w)
            with self.assertRaises(TypeError):
                s.api(lock).assign(42)
            s.skip(a, lock.acquire, lock.release)

    def test_allocate_wrong_type(self):
        s = Scenario()
        sem = s.Semaphore(1)
        def w():
            sem.acquire(); sem.release()
        with s:
            a = s.thread(w)
            with self.assertRaises(TypeError):
                list(s.api(sem).allocate("nope"))
            s.skip(a, sem.acquire, sem.release)

    def test_cycle_wrong_type(self):
        s = Scenario()
        ev = s.Event()
        with s:
            # parse rejects the wrong-type arg before any participant
            # state matters, so no threads need to exist.
            with self.assertRaises(TypeError):
                s.api(ev).cycle(object(), object())

    def test_allocate_duplicate_threads_are_allowed(self):
        s = Scenario()
        sem = s.Semaphore(2)
        order = []
        def w():
            sem.acquire(); order.append('first')
            sem.acquire(); order.append('second')
        with s:
            a = s.thread(w)
            self.assertEqual(list(s.api(sem).allocate(a, a)), [a, a])
        self.assertEqual(order, ['first', 'second'])


def run_tests():
    blankettestlib.run(name="blanket.primitives", module=__name__)


if __name__ == '__main__':
    run_tests()
    blankettestlib.finish()
