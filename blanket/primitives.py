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


from collections import defaultdict, deque, Counter
import math
import threading
import time
from types import MethodType, SimpleNamespace
import weakref

from big.boundinnerclass import BoundInnerClass
from big.builtin import ClassRegistry, ModuleManager
from big.itertools import iterator_context


mm = ModuleManager()
export = mm.export
delete = mm.delete

# Python scoping rules make blanket's heavy nested-classes technique clumsy.
# To get around it, we use a global dict called "base" to store base classes
# we'll want to subclass in a different class scope.
base = ClassRegistry()
delete('base')


_current_time = time.perf_counter
_rlock_provides_locked = hasattr(threading.RLock(), 'locked')



@export
class ThreadOrderingError(ValueError):
    """Raised when a scheduler API receives an impossible thread order."""
    pass

@export
class CompetingDriversError(ValueError):
    """Raised when creating a Driver(T) when T already has an active Driver."""
    pass


class unlock:
    """Unlocking context manager.  Release on entry, acquire on exit."""

    def __init__(self, lock):
        self._lock = lock

    def __enter__(self):
        self._lock.release()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.acquire()
        return False


def _do_nothing():
    pass  # pragma: no cover -- sentinel; callers do `is _do_nothing` checks


class ImmutableSequence(tuple):
    """Base class for blanket's immutable sequence types.  Subclass of tuple.

    tuple's built-in __eq__ and __hash__ basically ignores the instance's type.
    It just does what's internally called a "fast subclass check", and since
    ImmutableSequence and its descendents are subclasses of tuple, that passes,
    and so tuple ignores the fact that the types don't actually match and just
    compares the elements.  This means distinct subclasses with identical
    element values would compare equal, and hash the same (e.g. Terminated(t)
    and Nested(t) are both the tuple (t,)).  ImmutableSequence fixes this by
    implementing its own hash and equality tests.  (We have to implement all
    six rich compare operators, lest one falls through the cracks and gets
    handled by our base class, tuple.)
    """
    __slots__ = ()

    def __hash__(self):
        return hash(type(self)) ^ tuple.__hash__(self)

    def __eq__(self, other):
        return isinstance(other, type(self)) and tuple.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return NotImplemented

    def __le__(self, other):
        return NotImplemented

    def __gt__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented


@export
class TimeoutState(ImmutableSequence):
    """Tuple subclass for timeout information: (value, time, timed_out)."""
    __slots__ = ()

    def __new__(cls, value, time, timed_out):
        return tuple.__new__(cls, (value, time, timed_out))

    def __repr__(self):
        return f"TimeoutState(value={self.value}, time={self.time}, timed_out={self.timed_out})"

    @property
    def value(self):
        """The originally-requested timeout (None, 0, or positive seconds, or -1 for Lock.acquire's no-timeout sentinel)."""
        return self[0]

    @property
    def time(self):
        """The absolute time when timeout expires, or None if there is no timeout."""
        return self[1]

    @property
    def timed_out(self):
        """Did this transaction fail to complete before it timed out?"""
        return self[2]


class ImmutableSignalToken(ImmutableSequence):
    """Base class for signal token objects.

    Base class for ImmutableThreadSignalToken and
    ImmutableTransactionSignalToken.
    """
    pass


class ImmutableThreadSignalToken(ImmutableSignalToken):
    """Base class for signal token objects that wrap a thread.

    Base class for Use, Call, Terminated, and Not.

    Always has a .thread property in self[0].
    """
    @property
    def thread(self):
        return self[0]


class ImmutableTransactionSignalToken(ImmutableSignalToken):
    """Base class for signal tokens that wrap a transaction API.

    Base class for Nested and TransactionState.

    Always has the tx in self[0]; .tx, .transaction, and .thread
    properties expose it.
    """
    @property
    def tx(self):
        return self[0]

    @property
    def transaction(self):
        return self[0]

    @property
    def thread(self):
        return self.tx._core.thread


@export
class Signaling:
    """Marker base class for self-reporting signals.

    Every** signaling item in blanket is a subclass of Signaling.
    Signaling has a `signal` attribute returning True when the
    signal is currently high, False otherwise.

    - scenario.wait queries o.signal when it's fist called.  If
      o.signal is True, we return it immediately, otherwise we wait.
    - score.signal(o) confirms o.sample is True at call time, then
      wakes any waiters parked on o.

    Notably, blanket no longer tracks o anywhere between calls.
    No reference to o is kept in the score; the next call to wait(o)
    re-queries o.signal.  This means we don't keep lingering
    references to old Signaling things (Terminated(ancient_thread),
    prehistoric_tx).  When the user drops their references, the
    objec

    ** Thread objects and bound method objects obviously aren't
    subclasses of Signaling.  For a handful of exceptions like
    these, scenario.wait auto-boxes and unboxes 'em.
    """
    __slots__ = ()

    def sample(self, scenario):
        "Return True if this signal is currently high in scenario."
        raise NotImplementedError

    def normalized(self):
        """Return the normalized form of this signal: raw primitive
        and raw bound-method references are normalized to their
        cooked-primitive equivalents.  The default is identity;
        signals that wrap a primitive or bound method override.
        """
        return self


def _normalize_method(method):
    """Normalize a bound method to its normalized form: a Condition's
    lock-shared method (acquire/release/locked) collapses to the
    underlying lock's cooked method, and any raw-handle method
    collapses to its cooked-primitive method.  Condition-only methods
    (wait/wait_for/notify...) have no lock counterpart and just cook.
    """
    core = method.__self__._core
    underlying = getattr(core, 'underlying', None)
    if underlying is not None:
        # Condition: collapse to the lock's same-named method if it
        # has one.  getattr probes the lock so only genuinely shared
        # methods (acquire/release/locked) collapse; the rest fall
        # through and cook on the condition.
        lock_method = getattr(underlying.primitive, method.__name__, None)
        if lock_method is not None:
            return lock_method
    primitive = core.primitive
    if method.__self__ is primitive:
        return method
    return getattr(primitive, method.__name__)


def _normalize_primitive(primitive):
    "Normalize a raw handle to its cooked primitive; cooked stays put."
    return primitive._core.primitive


@export
class Use(Signaling, ImmutableThreadSignalToken):
    """A level signal that goes high whenever a thread uses a primitive.

    A Use signals while its 'thread' has any transaction on
    'primitive' in its call chain.

    ("Use" is the noun form here--Use rhymes with "moose", not "booze".)
    """
    __slots__ = ()

    def __new__(cls, thread, primitive):
        if not isinstance(thread, threading.Thread):
            raise TypeError(f"Use expected a thread, got {thread!r}")
        return tuple.__new__(cls, (thread, primitive))

    @property
    def primitive(self):
        return self[1]

    def sample(self, scenario):
        # Walks thread's tx chain looking for any tx whose
        # use_primitives set contains our (normalized) primitive.
        core = self.primitive._core
        outer = core.primitive
        score = scenario._core
        tx = score.transactions.get(self.thread)
        while tx is not None:
            if outer in tx.use_primitives:
                return True
            tx = tx.parent
        return False

    def __repr__(self):
        return f"Use({self.thread.name!r}, {self.primitive!r})"

    def normalized(self):
        cooked = _normalize_primitive(self.primitive)
        if cooked is self.primitive:
            return self
        return Use(self.thread, cooked)


@export
class Call(Signaling, ImmutableThreadSignalToken):
    """A level signal that goes high whenever a thread is calling a method.

    A Call signals while its 'thread' is calling 'method', optionally only
    while in transaction state 'state'.  (If 'state' is None, signals while
    the method call is in any state.)  'depth' distinguishes recursive calls
    to the same method on the same thread; the lowest one on the stack is
    at depth 0.  (The only possible *recursive* call is Condition.wait_for.)
    """
    __slots__ = ()

    def __new__(cls, thread, method, state=None, *, depth=0):
        if not isinstance(thread, threading.Thread):
            raise TypeError(f"Call expected a thread, got {thread!r}")
        if not callable(method):
            raise TypeError(f"Call expected a callable method, got {method!r}")
        if not isinstance(depth, int):
            raise TypeError(f"Call expected an integer depth, got {depth!r}")
        if depth < 0:
            raise ValueError(f"Call depth must be >= 0, got {depth!r}")
        primitive = getattr(method, '__self__', None)
        if primitive is None or not hasattr(primitive, '_core'):
            raise TypeError(
                f"Call expected a bound method on a regulated "
                f"primitive, got {method!r}")
        return tuple.__new__(cls, (thread, method, state, depth))

    @property
    def method(self):
        return self[1]

    @property
    def state(self):
        return self[2]

    @property
    def depth(self):
        return self[3]

    def sample(self, scenario):
        thread, method, state, depth = self
        score = scenario._core
        # Walk the thread's chain from leaf to root looking for a tx
        # calling this method (or a cooked alias) at recursion `depth`.
        # tx.depth is the per-method recursion depth (set at open as
        # method_parent.depth + 1), so we match on (method, depth).
        tx = score.transactions.get(thread)
        while tx is not None:
            if depth == tx.depth and method in tx.call_methods():
                if state is None:
                    return True
                return tx.state == state
            tx = tx.parent
        return False

    def __repr__(self):
        method_name = getattr(self.method, '__name__', repr(self.method))
        state_repr = "" if self.state is None else f" state={self.state.name}"
        depth_repr = "" if self.depth == 0 else f" depth={self.depth}"
        return f"Call({self.thread.name!r}, {method_name}{state_repr}{depth_repr})"

    def normalized(self):
        cooked = _normalize_method(self.method)
        if cooked is self.method:
            return self
        return Call(self.thread, cooked, self.state, depth=self.depth)


@export
class Terminated(Signaling, ImmutableThreadSignalToken):
    """A signal that goes high (and stays high) when a thread terminates.

    A Terminated signal is low while its 'thread' is alive, and goes
    high once the thread exits.  Once high, it stays high.
    """
    __slots__ = ()

    def __new__(cls, thread):
        if not isinstance(thread, threading.Thread):
            raise TypeError(f"Terminated expected a thread, got {thread!r}")
        return tuple.__new__(cls, (thread,))

    def sample(self, scenario):
        return not self.thread.is_alive()

    def __repr__(self):
        return f"Terminated({self.thread.name!r})"


@export
class Not(Signaling, ImmutableSignalToken):
    """A level signal that inverts the signaling object it wraps.

    Not wraps a signaling object signal and gives the opposite
    signal--high when the wrapped signal is low, and vice versa.

    Examples:
        Not(A) -- high iff thread A has no active transaction.
        Not(Terminated(A)) -- high iff A has not terminated.
        Not(Not(X)) -- high iff X is high.  (Literally evaluates to X.)
        Not(tx) -- high iff tx has not finished.
    """
    __slots__ = ()

    def __new__(cls, signal):
        if isinstance(signal, Not):
            return signal.wrapped
        if not isinstance(signal, (Signaling, threading.Thread, MethodType)):
            raise TypeError(
                f"Not expected a thread, a bound method, or a signal, "
                f"got {signal!r}")
        return tuple.__new__(cls, (signal,))

    @property
    def thread(self):
        wrapped = self.wrapped
        if isinstance(wrapped, threading.Thread):
            return wrapped
        return wrapped.thread

    @property
    def wrapped(self):
        """The signal this Not wraps (a thread, a bound method, or
        another Signaling like Terminated)."""
        return self[0]

    def sample(self, scenario):
        wrapped = self.wrapped
        # A bare thread or bound method inside Not is interpreted the
        # same way the top-level boxing layer would: a bare thread
        # means "has an active tx", a bare bound method means "any
        # thread is calling this method".  We invert that.
        if isinstance(wrapped, Signaling):
            return not wrapped.sample(scenario)
        if isinstance(wrapped, threading.Thread):
            return wrapped not in scenario._core.transactions
        # bare bound method: aggregate "anyone calling?" -> invert.
        return not scenario._core.BoundMethod(wrapped).sample(scenario)

    def normalized(self):
        wrapped = self.wrapped
        if isinstance(wrapped, Signaling):
            return Not(wrapped.normalized())
        # bare thread normalizes to itself; bare bound method
        # normalizes raw -> cooked.
        if isinstance(wrapped, threading.Thread):
            return self
        return Not(_normalize_method(wrapped))

    def __repr__(self):
        wrapped = self.wrapped
        if isinstance(wrapped, threading.Thread):
            return f"Not({wrapped.name!r})"
        return f"Not({wrapped!r})"


@export
class Nested(Signaling, ImmutableTransactionSignalToken):
    """A level signal that goes high while a transaction has an active child transaction.

    Self-reporting (Signaling): Nested(tx).sample returns
    tx._core.child is not None.
    """
    __slots__ = ()

    def __new__(cls, tx):
        if not isinstance(tx, Scenario._ScenarioCore.TxAPI):
            raise TypeError(f"Nested argument must be a Transaction, not {tx!r}")
        return tuple.__new__(cls, (tx,))

    def sample(self, scenario):
        return self.tx._core.child is not None

    def __repr__(self):
        return f"Nested({self.tx!r})"


@export
class Primitive(Signaling, ImmutableSignalToken):
    """An aggregate level signal: high while ANY thread has an active
    tx using the wrapped primitive (or its raw alias).

    Self-reporting: Primitive(p).signal walks all txs in p's score
    and returns True if any has p in its use_primitives set.

    Both Primitive(p) and Primitive(raw) signal together; the
    primitive form is normalized via p._core.primitive at signal
    read.  scenario.wait() auto-boxes bare primitives into
    Primitive(p) and unboxes the returned set.
    """
    __slots__ = ()

    def __new__(cls, primitive):
        # HEY API: do isinstance(Scenario.Primitive) here.
        # the _core attr check sucks, dude.
        if not hasattr(primitive, '_core'):
            raise TypeError(
                f"Primitive expected a regulated primitive (or raw), "
                f"got {primitive!r}")
        return tuple.__new__(cls, (primitive,))

    @property
    def primitive(self):
        return self[0]

    def sample(self, scenario):
        outer = self.primitive._core.primitive
        score = scenario._core
        for tx in score.transactions.values():
            cur = tx
            while cur is not None:
                if outer in cur.use_primitives:
                    return True
                cur = cur.parent
        return False

    def normalized(self):
        cooked = _normalize_primitive(self.primitive)
        if cooked is self.primitive:
            return self
        return Primitive(cooked)

    def __repr__(self):
        return f"Primitive({self.primitive!r})"


@export
class Reached(Signaling, ImmutableSignalToken):
    """A level signal that goes high once tx.state has reached
    (or surpassed) `state`.

    Self-reporting: Reached(tx, state).signal returns
    tx.state >= state on demand.  Two Reached(tx, state)
    instances compare equal and hash equal (tuple semantics),
    so they interoperate as a single key in score.waiters
    without any normalization machinery.
    """
    __slots__ = ()

    def __new__(cls, tx, state):
        if not isinstance(tx, Scenario._ScenarioCore.TxAPI):
            raise TypeError(f"Reached tx must be a transaction, not {tx!r}")
        if not isinstance(state, State):
            raise TypeError(f"Reached state must be a State, not {state!r}")
        return tuple.__new__(cls, (tx, state))

    @property
    def tx(self):
        return self[0]

    @property
    def state(self):
        return self[1]

    @property
    def thread(self):
        return self.tx._core.thread

    def sample(self, scenario):
        # Read tx._core.state directly; callers consult under score.lock.
        return self.tx._core.state.index >= self.state.index

    def __repr__(self):
        return f"Reached({self.tx!r}, {self.state!r})"


@export
class Action(Signaling, ImmutableTransactionSignalToken):
    """A level signal that goes high while a user callback associated
    with the transaction is running.  Currently the producers are
    barrier.wait around its action callback and condition.wait_for
    around its predicate callback.

    Self-reporting: Action(tx).sample returns tx._core.in_action.
    Two Action(tx) instances compare equal and hash equal (tuple
    semantics) and interoperate as a single key in score.waiters.
    """
    __slots__ = ()

    def __new__(cls, tx):
        if not isinstance(tx, Scenario._ScenarioCore.TxAPI):
            raise TypeError(f"Action argument must be a Transaction, not {tx!r}")
        return tuple.__new__(cls, (tx,))

    def sample(self, scenario):
        return self.tx._core.in_action

    def __repr__(self):
        return f"Action({self.tx!r})"


@export
class State(tuple):
    def __new__(cls, index, name):
        return tuple.__new__(cls, (index, name))

    def __repr__(self):
        return f"State({self.index!r}, {self.name!r})"

    @property
    def index(self):
        return self[0]

    @property
    def name(self):
        return self[1]


State.states = {}
State.BLOCKED   = State.states['BLOCKED']   = State(100,  "BLOCKED")
State.COMMIT    = State.states['COMMIT']    = State(101,  "COMMIT")
State.WAITING   = State.states['WAITING']   = State(102,  "WAITING")
State.STALLED   = State.states['STALLED']   = State(103,  "STALLED")
State.RESUMED   = State.states['RESUMED']   = State(104,  "RESUMED")
State.COMMITTED = State.states['COMMITTED'] = State(105,  "COMMITTED")
State.PAUSED    = State.states['PAUSED']    = State(106,  "PAUSED")
State.EXITING   = State.states['EXITING']   = State(107,  "EXITING")
State.RETURNED  = State.states['RETURNED']  = State(108,  "RETURNED")
State.RAISED    = State.states['RAISED']    = State(109,  "RAISED")

State.terminal_states = frozenset((State.RETURNED, State.RAISED))
State.by_index = {s.index: s for s in State.states.values()}


@export
class TransactionState(Signaling, ImmutableTransactionSignalToken):
    """A level signal that goes high while a transaction is in a state.

    Self-reporting (Signaling): TransactionState(tx, X).signal returns
    tx._core.state is X on demand.  No cache; instances are cheap
    tuple subclasses, equal/hashable via tuple semantics so two
    Blocked(tx) constructions interoperate as a single key in
    score.waiters.

    You should use the subclasses directly, e.g. Blocked(tx), Stalled(tx).
    But TransactionState(tx, State.BLOCKED) also works, and dispatches
    to the bound subclass for the given state.
    """
    __slots__ = ()
    _subclass_for_state = {}

    def __new__(cls, tx, state):
        if not isinstance(tx, Scenario._ScenarioCore.TxAPI):
            raise TypeError(f"tx must be a transaction, not {tx!r}")
        if not isinstance(state, State):
            raise TypeError(f"state must be a State, not {state!r}")

        if cls is TransactionState:
            # Generic call: dispatch to the bound subclass for this state.
            cls = cls._subclass_for_state.get(state)
            if cls is None:
                raise ValueError(f'unrecognized state {state!r}')

        return tuple.__new__(cls, (tx, state))

    @property
    def state(self):
        return self[1]

    def sample(self, scenario):
        return self.tx._core.state is self.state

    def __repr__(self):
        return f"{type(self).__name__}({self.tx!r})"


def transaction_state_subclass(state):
    """Bind a TransactionState subclass to a State.

    After @transaction_state_subclass(State.BLOCKED), Blocked(tx) and
    TransactionState(tx, BLOCKED) produce equivalent Blocked
    instances (equal and hash-equal via tuple semantics).  isinstance
    checks work against both Blocked and TransactionState.
    """
    def decorator(cls):
        TransactionState._subclass_for_state[state] = cls
        return cls
    return decorator

delete('transaction_state_subclass')


@export
@transaction_state_subclass(State.BLOCKED)
class Blocked(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.BLOCKED)


@export
@transaction_state_subclass(State.COMMIT)
class Commit(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.COMMIT)


@export
@transaction_state_subclass(State.WAITING)
class Waiting(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.WAITING)


@export
@transaction_state_subclass(State.STALLED)
class Stalled(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.STALLED)


@export
@transaction_state_subclass(State.RESUMED)
class Resumed(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.RESUMED)


@export
@transaction_state_subclass(State.COMMITTED)
class Committed(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.COMMITTED)


@export
@transaction_state_subclass(State.PAUSED)
class Paused(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.PAUSED)


@export
@transaction_state_subclass(State.EXITING)
class Exiting(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.EXITING)



@export
@transaction_state_subclass(State.RETURNED)
class Returned(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.RETURNED)



@export
@transaction_state_subclass(State.RAISED)
class Raised(TransactionState):
    __slots__ = ()
    def __new__(cls, tx):
        return super().__new__(cls, tx, State.RAISED)





@export
class Scenario:
    """Object managing primitives and threads for deterministic scenarios."""

    def __init__(self):
        self._core = self._ScenarioCore()
        self._context_manager = None

    def __repr__(self):
        return repr(self._core).replace("_ScenarioCore", "Scenario")

    @property
    def name(self):
        return self._core.name

    @name.setter
    def name(self, value):
        self._core.name = value

    def reset(self):
        """Clear accumulated working state from the scenario.

        Drops the waiters reverse index and the completed-tx log.
        Leaves structural state alone (registered primitives, raws,
        managed threads, family signal sets).

        Every wait item is Signaling (self-reporting), so there is
        no per-score "currently high" set to drain -- terminated
        txs continue to read True via their own .signal property
        without holding any score-level reference.

        Safe to call multiple times.  Auto-called when exiting a
        scenario context manager.
        """
        with self._core.lock:
            self._core.reset()

    @property
    def apis(self):
        return self._core.apis_proxy

    def api(self, primitive):
        return self._core.apis_proxy.get(primitive)

    @property
    def raws(self):
        return self._core.raws_proxy

    def raw(self, primitive):
        return self._core.raws_proxy.get(primitive)

    @property
    def log(self):
        """A list of completed transactions."""
        return self._core.log

    @property
    def managed(self):
        return self._core.managed_proxy

    def thread(self, target, *args, **kwargs):
        """Create and register a managed thread."""
        with self._core.lock:
            return self._core.thread(target, *args, **kwargs)

    @property
    def transactions(self):
        """Dict-like access to transactions by thread."""
        return self._core.transaction_apis_proxy

    def transaction(self, thread):
        """Returns the current transaction for this thread.
        Returns None if the thread has no active transaction.
        """
        with self._core.lock:
            tx = self._core.transaction(thread)
            if tx is not None:
                tx = tx.api
            return tx

    def wait(self, *items, timeout=None):
        """
        Block until any of the specified items signal.

        Supported items:
            bound method object on a primitive
                signals while any thread is inside a call to that method
            Scenario
                signals while a thread has entered the scenario
            Thread
                signals while the thread has an active visible blanket
                transaction.  Thread termination is represented separately
                by Terminated(thread).
            transaction
                signals once the transaction has completed
            Call, Use, Not, Terminated, Nested, and TransactionState objects
                see their docs for more information
        """
        if not items:
            raise ValueError("must specify at least one item")
        items = set(items)
        with self._core.lock:
            return self._core.wait(items, timeout=timeout)

    def park(self, *args, wait=False):
        """Wait until each named thread reaches its specified method.

        Usage:
            scenario.park(A, lock.acquire, B, lock.release)
            scenario.park(A, cond.notify)

        Arguments come in pairs: first a thread, then exactly one method
        for that thread, then optionally another thread followed by a method,
        and so on.  Each thread may appear at most once.  Every method must
        be a method from a blanket regulated synchronization primitive.

        When each thread enters a method, park checks to see if the
        method the thread called matches the method specified as an
        arg after that thread.  If it doesn't, park unblocks the method
        call and goes back to sleep.  If it does, park removes the thread
        from the list of threads it's watching (and leaves the thread
        blocked at the scheduler block).  Once this list of threads is
        empty, park exits, at which point all threads are blocked at
        the scheduler block for their respective specified methods.

        If wait=True, park will also unblock the final matching method
        call and wait for it to complete.  This is useful for driving
        worker threads through setup methods you don't care to interact
        with.  Per-thread waits happen concurrently, not sequentially,
        so threads that depend on each other won't deadlock.

        Returns a dict mapping each thread to the transaction calling
        the matching method.  (If wait is True, every transaction will
        have already completed.)  Raises RuntimeError if any thread
        diverges from the script.

        (The word "park" is borrowed from Java's LockSupport terminology:
        threads are "parked" until a "permit" allows them to resume.)
        """
        with self._core.lock:
            if not self._core.entered:
                raise RuntimeError("can't park, scenario not entered")
            tx_by_thread = self._core.park(*args, wait=wait)
        return {t: tx.api for t, tx in tx_by_thread.items()}

    def skip(self, *args, wait=False):
        """Skip one or more threads past one or more method calls.

        Usage:
            scenario.skip(A, lock.acquire, lock.release, B, lock.acquire)

        Arguments are flat: first a thread, then one or more methods
        for that thread.  You can then specify another thread, and
        one or more methods for that thread, and so on.  You can switch
        back and forth between threads as needed, and you can specify
        the same thread as many times as needed.  Every method
        must be a regulated method on a synchronization object created
        by this scenario object.

        For each (thread, method) pair: skip waits for that thread
        to enter a method call, validates it matches the method
        specified, and unblocks it.  For all methods except the last (per
        thread), skip then waits for the transaction to complete before
        proceeding.  For the last method per thread, if wait is false
        (the default), skip returns after unblocking the last thread.
        If wait is true, skip also waits for the last method call on
        every thread to finish.

        Returns a dict mapping each thread to the last transaction.
        (If wait is true, every transaction will have already completed.)
        Raises RuntimeError if any thread calls an unexpected method.
        """
        with self._core.lock:
            if not self._core.entered:
                raise RuntimeError("can't skip, scenario not entered")
            tx_by_thread = self._core.skip(*args, wait=wait)
        return {t: tx.api for t, tx in tx_by_thread.items()}

    def finish(self, *threads):
        """Best-effort drive of each named thread to terminal.

        Builds a Driver per thread, pushes finish() to drive whatever
        tx each is currently on, and drains via a Dispatch.  When a
        Driver yields without its thread being terminated yet, pushes
        another finish() and re-adds to the Dispatch.  Loops until
        every Driver reaches terminated state.

        Best-effort: if a thread is parked waiting on something the
        test setup isn't providing (a notify that never comes, a
        lock release that never happens), finish() may deadlock --
        there's no oracle to know which way to nudge.  The 5-second
        per-test timeout will catch it as a test failure rather
        than an infinite hang.

        Intended for end-of-scenario cleanup of tests that leak
        parked threads.  scenario.__exit__ does auto-unpark
        leaked parked txs (so workers can resume natively), but
        does not drive to terminal: a worker that resumes into
        another regulated primitive whose underlying actual
        primitive is unsatisfiable will block at the OS level and
        the join will hang.  finish() does the explicit driving
        when that's a concern.
        """
        with self._core.lock:
            if not self._core.entered:
                raise RuntimeError("can't finish, scenario not entered")
            self._core.finish(*threads)

    def __enter__(self):
        if self._context_manager is not None:
            raise RuntimeError("can't __enter__, already entered scenario")

        self._context_manager = cm = self._core.ContextManager()
        cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context_manager is None:
            raise RuntimeError("can't __exit__, haven't entered scenario")
        cm = self._context_manager
        self._context_manager = None
        return cm.__exit__(exc_type, exc_val, exc_tb)

    @BoundInnerClass
    class _ScenarioCore:
        """Internal state for Scenario."""

        def __init__(self, api):
            self.lock = threading.Lock()

            self._name = ''
            self.api = api
            # Per-core registry; weak so primitives drop out naturally
            # when the user releases their references.  Cores are added
            # in Core.__init__.  All public primitive→api / primitive→raw
            # lookups go through this set, so score holds nothing
            # strongly that anchors a primitive.
            self.cores = weakref.WeakSet()
            self.apis_proxy = self.CoreAttrMapProxy('api')
            self.blockers = defaultdict(self.blocker_factory)
            self.log = []
            self.log_proxy = self.LogProxy(self.log)
            self.managed = managed = {}
            self.managed_proxy = self.LockedSetProxy(managed)
            self.entered = False
            # thread-as-signal: handled by score.Thread(t) self-reporter
            # which reads score.transactions; no per-(score, thread)
            # SignalMinder dict needed.  Use signals likewise self-report
            # by walking the chain, so no use_minders dict either.
            self.monitors = {}
            self.serial_number = 1
            # Every wait-item is Signaling (self-reporting); there is
            # no persistent "currently high" set.  signal() and wait()
            # both consult item.signal directly.
            self.threads = set()
            self.transactions = {}
            self.drivers = weakref.WeakValueDictionary()  # thread -> Driver (current Driver for the thread)
            self.driver_apis = {}  # Driver core -> Driver api (back-reference for Dispatch __next__)
            self.transactions_proxy = self.ReadOnlyDictProxy(self.transactions)
            self.transaction_apis = transaction_apis = {}
            self.transaction_apis_proxy = self.LockedDictProxy(transaction_apis)
            self.raws_proxy = self.CoreAttrMapProxy('raw')
            self.waiters = defaultdict(set)
            # Aggregate usage refcount: cooked primitive / cooked bound
            # method -> number of threads currently using it.  Drives
            # Primitive(p) / BoundMethod(m) (and their Nots) wakeups
            # without an O(threads) poll at tx close.  See incref_usage.
            self.usage_counter = Counter()

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, value):
            if not isinstance(value, str):
                raise TypeError(f"name must be a string, not {type(value).__name__}")
            self._name = value

        def __repr__(self):
            name = f"{self._name} " if self._name else ""
            entered = "entered " if self.entered else ""
            locked = "locked" if self.lock.locked() else "unlocked"
            return (f"<{name}_ScenarioCore at {hex(id(self.api)).upper()}"
                    f" {entered}{locked} {len(self.cores)} primitives"
                    f" {len(self.managed)} threads"
                    f" {len(self.transactions)} transactions>")

        @BoundInnerClass
        class WaitTransaction:
            """Represents an in-process scenario.wait() call.

            Stores references to all context needed for a scenario.wait
            call: the waiting thread, the frozenset of items being
            waited on, the accumulator for items that have signaled
            during this wait, and a one-shot blocker callable.

            The blocker is the wait()-side blocker.release method,
            captured at install time.  The first signaler to find this
            wtx in score.waiters[item] calls the blocker (waking the
            waiter) and clears it; subsequent signalers see None and
            only append to wtx.signaled.

            Hash and equality are by object identity (default).  Each
            wait() call constructs a fresh wtx; no two wtxs ever need
            to compare equal.
            """

            def __init__(self, score, items):
                self.score = score
                self.items = items = frozenset(items)
                # unbox maps each normalized waiters key back to the
                # set of user-supplied originals that normalized to it.
                # Two spellings (raw_lock.acquire and lock.acquire) can
                # land on the same key; on signal we hand back every
                # original the user passed.
                self.unbox = unbox = {}

                monitors = set(score.monitors.values())
                keys = set()
                for item in items:
                    boxed = score.box_signal(item)
                    key = boxed.normalized()

                    # Any thread-bearing signal (Thread, Terminated,
                    # Use, Call) needs its thread started, non-monitor,
                    # and registered -- registration starts the monitor
                    # that strobes Terminated(thread).
                    if isinstance(key, ImmutableThreadSignalToken):
                        thread = key.thread
                        if thread in monitors:
                            raise ValueError(
                                f"can't wait on monitor thread {thread.name!r}")
                        if thread.ident is None:
                            raise ValueError(
                                f"can't wait on unstarted thread {thread.name!r}")
                        if thread not in score.threads:
                            score.register_thread(thread)

                    unbox.setdefault(key, set()).add(item)
                    keys.add(key)

                self.keys = frozenset(keys)
                self.signaled = set()
                # Set to blocker.release while parked; score.signal
                # calls it (once) to wake us, then clears it.
                self.blocker = None

            def __repr__(self):
                return (f"<WaitTransaction items={self.items!r} signaled={self.signaled!r}>")

            def wait(self, timeout=None):
                assert not self.signaled

                score = self.score
                scenario = score.api
                # keys are normalized; we unbox to originals on return.
                signaled = set()

                for key in self.keys:
                    if key.sample(scenario):
                        signaled.add(key)

                # note: if timeout is None, we want to wait for a signal
                # so, specifically, timeout != 0
                if (not signaled) and (timeout != 0):
                    thread = threading.current_thread()
                    blocker = score.blockers[thread]
                    # Arm the wakeup before installing as a waiter, all
                    # under score.lock, so no signaler can fire before
                    # blocker is set.
                    self.blocker = blocker.release

                    waiters = score.waiters
                    for key in self.keys:
                        waiters[key].add(self)

                    with unlock(score.lock):
                        timeout = -1 if timeout is None else timeout
                        blocker.acquire(timeout)

                    for key in self.keys - self.signaled:
                        waiters[key].discard(self)

                    signaled |= self.signaled

                unbox = self.unbox
                result = {orig for key in signaled for orig in unbox[key]}
                self.signaled = signaled
                return result


        def register_thread(self, thread):
            """Register a thread with the scenario.

            Signals are inert and self-reporting, so there's no initial
            signal state to set up: Thread(thread), Terminated(thread),
            and their Nots all compute their truth on demand.  And no
            waiters can be parked on this thread yet (a wait that named
            it is what triggered registration, and it installs its
            waiters only after this returns).  So registration just
            records the thread and, if it's alive, starts a monitor to
            strobe Terminated(thread) when it exits.

            Raises ValueError if thread hasn't started yet.
            A no-op if thread has already been registered.
            """
            if thread in self.threads:
                return
            if thread.ident is None:
                raise ValueError(
                    f"cannot register thread {thread.name!r}: not started yet")
            self.threads.add(thread)

            # The thread has never been seen before, so it can't have
            # an active transaction.
            assert self.transactions.get(thread) is None

            if thread.is_alive():
                monitor = threading.Thread(
                    target=self.monitor,
                    args=(thread,),
                    daemon=True,
                    name=thread.name.replace("Thread-", "Monitor-"),
                )
                self.monitors[thread] = monitor
                monitor.start()
            # If already dead, Terminated(thread) self-reports high on
            # demand; no monitor needed.

        def monitor(self, thread):
            "Sets Terminated(thread) when the thread terminates."
            thread.join()
            with self.lock:
                # Thread cleanup.
                del self.monitors[thread]
                self.managed.pop(thread, None)
                self.signal(Terminated(thread))
                # Not(Terminated(t)) goes low automatically (its
                # .signal = not Terminated(t).signal = t.is_alive());
                # no wakeup needed since Signaling high->low is silent.

                # Stuck-tx cleanup: if the thread died with an active tx,
                # walk the chain and abort each so observers fire and
                # Not(thread) goes high.
                tx = self.transactions.get(thread)
                if tx is not None:
                    tx.aborted()  # pragma: no cover -- defensive: workers normally exit cleanly after their last tx terminates


        @base()
        @BoundInnerClass
        class Driver:
            # these are Driver states, not tx states
            idle       = State(11, 'IDLE')
            active     = State(12, 'ACTIVE')
            skipping   = State(13, 'SKIPPING')
            parking    = State(14, 'PARKING')
            finishing  = State(15, 'FINISHING')
            parked     = State(16, 'PARKED')
            finished   = State(17, 'FINISHED')
            raised     = State(18, 'RAISED')
            terminated = State(19, 'TERMINATED')
            # NESTING is the post-Nested-fire yield point: when the
            # Driver is armed via driver.nested() and a child tx
            # appears, the signal handler rotates self.tx to the
            # child, pushes the parent context onto self.stack, and
            # transitions here.  state_signals[nesting] is empty so
            # the Driver yields immediately; the caller can drive the
            # child via another imperative on the same Driver, or
            # leave the Driver alone for the cycle code to re-pursue.
            # Semantically equivalent to ACTIVE for pursue's state
            # checks; distinct so callers can tell "Nested fired"
            # apart from "fresh active driver."  The gerund form
            # avoids a name collision with the .nested() imperative.
            nesting    = State(20, 'NESTING')

            driving_states  = frozenset((skipping, parking, finishing))
            active_states   = frozenset((active, nesting)) | driving_states
            terminal_states = frozenset((parked, finished, raised, terminated))

            # Canary states are the tx states past target that the tx
            # might reach if it overshoots its intended park.  Named
            # after the canary in the coal mine: when one is signaled
            # while Driver is in PARKING, the tx walked past where we
            # wanted it and Driver raises.
            canaries = {
                State.BLOCKED: frozenset((State.COMMIT, State.COMMITTED)),
                State.COMMIT:  frozenset((State.COMMITTED,)),
                State.WAITING: frozenset((State.COMMITTED,)),
                State.STALLED: frozenset((State.COMMITTED,)),
                State.PAUSED:  frozenset((State.EXITING,)),
            }

            def __init__(self, score, thread):
                if thread is threading.current_thread():
                    raise ValueError("can't create a Driver for the current thread")

                self.score = score
                self.thread = thread

                self.owner = None

                self.base_thread_signal = {Terminated: Terminated(thread)}
                self.thread_signal = self.base_thread_signal
                terminated = self.base_thread_signal[Terminated]

                self.state_signals = {
                    self.idle:       frozenset((terminated, thread)),
                    self.active:     frozenset(),
                    self.skipping:   frozenset((terminated,)),
                    self.parking:    frozenset((terminated,)),
                    self.finishing:  frozenset((terminated,)),
                    self.parked:     frozenset(),
                    self.finished:   frozenset(),
                    self.nesting:    frozenset(),
                    self.terminated: frozenset(),
                }

                self.txs_seen = {}
                self.txs = []

                # Lazy initialization: don't claim the score's slot
                # or cache the current tx at construction time.  Two
                # Drivers for the same thread can be constructed
                # without competing -- they only compete when one of
                # them actually tries to drive (via an imperative or
                # an explicit drive()).  See initialize.
                self.initialized = False
                self.state = None
                self.tx = None
                self.target = None
                self.base = None
                self.stack = []
                self.signals = frozenset()
                self.ready = False
                self.lazy_work = None
                self.lazy_verb = None
                # If True, the next Nested fire transitions us to the
                # NESTED state (which has empty signals, so we yield
                # for caller review) instead of the default skipping/
                # delegate-parking behavior.  One-shot: cleared by the
                # signal handler that consumes it.  Set via driver.nested().
                self.nested_armed = False


            def initialize(self):
                if self.initialized:
                    return
                self.initialized = True

                if not self.score.entered:
                    raise RuntimeError(f"can't use driver, scenario not entered")

                # Claim the score's slot for this thread.  Two Drivers on one thread
                # would fight; the registry forces serialization.
                score = self.score
                thread = self.thread

                if thread in score.drivers:
                    raise CompetingDriversError(
                        f"thread {thread.name!r} already has an active Driver "
                        f"{score.drivers[thread]!r}")

                score.drivers[thread] = self

                # special sentinel value so we'll properly
                # set all the thread signals to None
                self.tx = self
                self.cache_tx()

                if self.tx is not None:
                    # Driver is taking over from the user.  Clear user
                    # pauses up the tx stack so prior user-set pauses
                    # don't interfere with imperatives.  This is the
                    # only place Driver clears tx.pause; anywhere
                    # else, assume Driver is in charge and didn't set
                    # it.  Each cleared flag matches one user
                    # contribution to pausing, so decrement.  We do
                    # NOT unpark here even if pausing hits zero on a
                    # tx parked at PAUSED: the Driver imperative
                    # about to be invoked will drive past PAUSED via
                    # the auto-advance frog-march, which clears flag
                    # and counter together.  If no imperative drives
                    # (e.g. caller raises before issuing one), the tx
                    # is left at PAUSED with counter zero -- caller
                    # cleans up via s.finish(t), not tx.unpause().
                    cursor = self.tx
                    while cursor is not None:
                        if cursor.pause:
                            cursor.pause = False
                            cursor.pausing -= 1
                        cursor = cursor.parent

                self.stack = []

                self.state = self.idle
                # target is only used for parking state, not for skipping/finishing
                self.target = None
                # base is the operational base of the current pursue
                # context -- the tx whose state/target the driver is
                # ultimately driving toward.  Maintained as an
                # invariant: set on every transition into active state
                # (initialize, idle->active), rotated to a delegate
                # child by the Nested-fire handler, and restored from
                # the 4th field of stack frames on every pop.  In
                # active state, self.base equals self.tx; the two
                # diverge only while skipping through a non-delegate
                # child (self.tx is the child, self.base is the saved
                # parent on the stack).
                self.base = None

                self.signals = frozenset()
                self.ready = False
                # Deferred work staged by pursue: a zero-argument
                # callable (typically tx.unblock as a bound method;
                # closure if it ever needs to be more) that fires
                # the moment the driver actually starts being
                # driven -- on entry into Driver.__call__, or on the
                # next Dispatch.__next__ drain of recent.  None means
                # nothing staged.  See Driver.drive.
                self.lazy_work = self.lazy_verb = None
                # Reset nested-arm: a fresh active session starts
                # unarmed.  Callers re-arm with driver.nested() after
                # reactivate if they want to be alerted on a child.
                self.nested_armed = False

                # set by cache_tx
                tx = self.tx
                if tx is None:
                    self.to(self.idle)
                    return

                self.base = tx
                self.to(self.active)

            def __repr__(self):
                state = self.state.name if self.state is not None else 'uninitialized'
                target = f' -> {self.target.name}' if self.target else ''
                tx = f' {self.tx}' if self.tx else ''
                return f"<Driver {self.thread.name!r} {state}{tx}{target} owner={self.owner!r}>"

            @property
            def done(self):
                return self.state in self.terminal_states

            def register(self, owner):
                if self.owner is not None:
                    raise RuntimeError(f"driver is already owned by {self.owner!r}")
                self.owner = owner
                return self.signals

            def drive(self):
                """Fire and clear the deferred-work callable, if any.
                Called automatically by Driver.__call__ at entry, and
                by Dispatch's drain_recent at the top of each
                Dispatch.__next__ -- the two paths through which a
                driver actually starts being driven."""
                self.initialize()
                lazy_work = self.lazy_work
                if lazy_work is None:
                    return
                self.lazy_work = self.lazy_verb = None
                lazy_work()

            def unregister(self):
                if self.owner is None:
                    raise RuntimeError(f"driver is not owned")
                self.owner = None

            def reactivate(self):
                if not self.state in self.terminal_states:
                    raise RuntimeError(f"can't reactivate, already in {self.state!r} state")
                self.initialize()

            def cache_tx(self):
                tx = self.score.transaction(self.thread)

                if self.tx == tx:
                    return

                self.tx = tx
                if tx is None:
                    self.thread_signal = self.base_thread_signal
                    return

                if tx in self.txs_seen:
                    self.thread_signal = self.txs_seen[tx]
                    return

                self.txs.append(tx)

                api = tx.api
                ts = self.thread_signal = self.base_thread_signal.copy()
                ts[State.BLOCKED]   = Blocked(api)
                ts[State.STALLED]   = Stalled(api)
                ts[State.WAITING]   = Waiting(api)
                ts[State.PAUSED]    = Paused(api)
                ts[State.COMMIT]    = Commit(api)
                ts[State.COMMITTED] = Committed(api)
                ts[State.EXITING]   = Exiting(api)
                ts[Nested]          = Nested(api)
                self.txs_seen[tx] = ts


            def close(self):
                """Cleanup on entering a terminal state.  Idempotent:
                to() calls this when transitioning into a terminal
                state, and external callers can call it too without
                worrying whether to() has already done so.  (Still
                not exposed on the Driver api object -- this is
                blanket-internal; the api never calls it.)"""
                assert self.ready is False
                self.initialized = False
                self.target = None
                self.signals = frozenset()
                # Release the score's slot.  pop-with-default makes
                # this idempotent: a second call (or a call after
                # to(terminal) already ran) is a no-op.
                self.score.drivers.pop(self.thread, None)

            def to(self, state):
                self.state = state

                if state in self.terminal_states:
                    # Clear stale signals from prior state BEFORE
                    # _update_owner, or it would call owner.listen()
                    # with the old signals right before owner.done().
                    # Tell owner we're done before close().  If there's
                    # no owner, the pending 'done' has nowhere to go;
                    # clear it ourselves to maintain close()'s invariant.
                    self.close()
                    return

                signals = self.state_signals[state]

                if state is not self.idle:
                    if state in self.driving_states:
                        def transaction_states_set(states):
                            return set(self.thread_signal[s] for s in states)
                        parking_signals = transaction_states_set(
                            (State.BLOCKED, State.STALLED, State.PAUSED))

                        signals = set(signals)
                        signals.add(self.tx)
                        signals |= parking_signals
                        signals.add(self.thread_signal[Nested])
                        if state is self.parking:
                            signals.add(self.thread_signal[self.target])
                            signals |= transaction_states_set(self.canaries[self.target])
                    elif state is self.active and self.nested_armed and self.tx is not None:
                        # Active state with the arm set: the .nested()
                        # imperative wants us to wake on the next Nested
                        # fire even though no pursue is in flight.
                        # state_signals[active] is empty by default
                        # (active yields immediately); override to
                        # include Nested.
                        signals = frozenset({self.thread_signal[Nested]})
                    # else: active without arm -> empty signals (yield);
                    # nesting -> empty signals (yield).
                    signals = frozenset(signals)

                self.signals = signals
                return signals


            def signal(self, signals):
                ts = self.thread_signal

                # IDLE has limited listening; handle separately.
                if self.state is self.idle:
                    if ts[Terminated] in signals:
                        return self.to(self.terminated)
                    assert self.thread in signals
                    assert self.tx is None
                    self.cache_tx()
                    self.base = self.tx
                    return self.to(self.active)

                assert self.state in self.active_states
                assert self.tx

                # Nested is a child-push event, distinct from tx-state
                # signals.
                if ts[Nested] in signals:
                    # A child tx has been pushed on top of self.tx
                    # while the Driver was active.  Save the current
                    # pursue context (tx, state, target, base) so it
                    # can be restored when the child eventually pops.
                    self.stack.append(
                        (self.tx, self.state, self.target, self.base))
                    parent = self.tx
                    self.cache_tx()
                    # If the caller armed nested-aware mode, transition
                    # to NESTING (empty signals -> yields to caller).
                    # self.tx is now the child; self.base follows tx
                    # into the child so the active-state invariant
                    # (base == tx) holds and pursue from NESTING acts
                    # on the child.  The original (parent) base is
                    # preserved on the stack frame above for restore
                    # at tx-end.  One-shot: arm cleared here so the
                    # next Nested fire follows default (skipping/
                    # delegate) behavior unless re-armed.
                    if self.nested_armed:
                        self.nested_armed = False
                        self.base = self.tx
                        return self.to(self.nesting)
                    # Default behavior: switch focus to the child and
                    # go to skipping so the child runs through to its
                    # own terminal.  Exception: if we were parking
                    # toward a tx state AND the parent says the child
                    # is a delegate, rotate self.base to the child and
                    # re-enter parking -- the target state now applies
                    # to the child (whose state signals replace the
                    # parent's via cache_tx).
                    if (self.state is self.parking
                            and parent.is_delegate(self.tx)):
                        self.base = self.tx
                        return self.to(self.parking)
                    return self.to(self.skipping)

                # tx-end is the most-progressed possible signal: the
                # tx is done walking states.  Handle specially because
                # the dispatch depends on Driver state and the stack.
                if self.tx in signals:
                    if self.stack:
                        tx, state, target, base = self.stack.pop()
                        assert self.tx.parent is tx
                        # cache_tx() refreshes self.tx and the
                        # thread_signal namespace against the now-
                        # current top of the thread's tx stack.
                        self.cache_tx()
                        if self.tx is not tx:
                            # Parent also terminated while we were
                            # skipping through the child.  Whatever
                            # we were resuming the parent toward, the
                            # parent's terminal moots it.  For
                            # finishing, the terminal IS the target;
                            # for parking, the parent overshot to a
                            # terminal instead of our target state;
                            # for skipping, we just continue (next
                            # pop or idle).
                            if state is self.finishing:
                                return self.to(self.finished)
                            if state is self.parking:
                                raise self.overshot(tx.state.name)
                            assert state is self.skipping
                            if self.stack:
                                # Re-pop: the next ancestor is the
                                # actual resume point.  (Multi-level
                                # nested where intermediates all
                                # closed.)  We rely on cache_tx having
                                # surfaced whichever tx is current
                                # (which may itself be the next pop's
                                # tx, or further up).  Loop until we
                                # find a match or run out of stack.
                                while self.stack:
                                    tx, state, target, base = self.stack.pop()
                                    if self.tx is tx:
                                        self.target = target
                                        self.base = base
                                        return self.to(state)
                                    if state is self.finishing:
                                        return self.to(self.finished)
                                    if state is self.parking:
                                        raise self.overshot(tx.state.name)
                            # Cascade walked off the bottom of the
                            # stack without a match: every saved
                            # ancestor has also closed and the
                            # worker has advanced past them all.
                            # self.tx (from the earlier cache_tx)
                            # holds whatever it surfaced.  Yield
                            # ACTIVE on that tx if it exists; only
                            # IDLE if the thread truly has no
                            # current tx (IDLE asserts self.tx is
                            # None).
                            if self.tx is None:
                                return self.to(self.idle)
                            self.base = self.tx
                            return self.to(self.active)
                        self.target = target
                        self.base = base
                        return self.to(state)
                    if self.state is self.parking:
                        # tx terminated while we were trying to park.
                        # The ultimate overshoot.  Pass the actual
                        # terminal state (RETURNED or RAISED) for
                        # diagnostic clarity.
                        raise self.overshot(self.tx.state.name)
                    if self.state is self.finishing:
                        return self.to(self.finished)
                    if self.state is self.active:
                        # The scheduler is in ACTIVE because the user
                        # paused to inspect the tx, which means the tx
                        # is at BLOCKED waiting for the scheduler to
                        # advance it.  If the tx terminated anyway,
                        # either Loki is interfering or there's a race
                        # -- either way the scheduler isn't actually
                        # in control.  Complain.
                        self.to(self.raised)
                        raise RuntimeError(f"{self}: tx terminated while Driver was ACTIVE (tx now {self.tx.state.name})")

                    assert self.state is self.skipping
                    self.cache_tx()
                    # cache_tx may have surfaced a fresh tx (worker
                    # has already advanced to a subsequent regulated
                    # call after the skipped tx terminated).  IDLE
                    # asserts self.tx is None; transition there only
                    # if cache_tx confirms the thread has no current
                    # tx, otherwise yield ACTIVE on the new tx so the
                    # caller can issue the next imperative.
                    if self.tx is None:
                        return self.to(self.idle)
                    self.base = self.tx
                    return self.to(self.active)

                # Among the non-end tx-state signals, find the most-
                # progressed.  When multiple state signals fire in the
                # same wake (the tx ran past several states quickly),
                # earlier ones are stale and we process only the
                # furthest.  Listed highest-progression first so the
                # first match short-circuits.
                for state in (State.EXITING, State.PAUSED, State.COMMITTED,
                              State.STALLED, State.WAITING, State.COMMIT, State.BLOCKED):
                    if ts[state] in signals:
                        current_state = state
                        break
                else:
                    assert ts[Terminated] in signals
                    return self.to(self.terminated)

                if self.state is self.parking:
                    # self.tx is the pursue base.  It may have a
                    # parent if the base was selected via is_delegate
                    # (e.g. a cond.wait child whose wait_for parent
                    # delegates park-state semantics to it); otherwise
                    # walk-up reached the actual root.

                    if current_state in self.canaries[self.target]:
                        # Canary: we overshot the target state.
                        raise self.overshot(current_state.name)

                    if self.target is current_state:
                        return self.to(self.parked)

                # Transit states: scheduler advances the tx past these.
                # COMMIT, COMMITTED, and EXITING are caught above as
                # canaries (the only paths that put them in listening),
                # and target match has handled the case where one of
                # the four parking states was the user's target in
                # PARKING.  Auto-advance past whichever fired and keep
                # listening: driver state is unchanged, the tx is now
                # in flight again, and the next signal (the next park
                # or target match) will route here again.
                # Including PAUSED: if Loki sets pause on a tx the
                # scheduler is driving, the scheduler wins.
                if current_state is State.BLOCKED:
                    self.tx.unblock()
                elif current_state is State.STALLED:
                    self.tx.unstall()
                else:
                    assert current_state is State.PAUSED
                    # Frog-march past PAUSED: scheduler wins over a
                    # user-set pause.  Clear both flag and counter,
                    # then unpark to EXITING.
                    self.tx.pause = False
                    self.tx.pausing = 0
                    self.tx.unpark(State.EXITING)
                return self.signals

            def overshot(self, state_name):
                """Driver reached state_name unexpectedly (canary
                signaled, or tx terminated while trying to park).
                Transition to RAISED and return the exception, which
                the caller raises."""
                msg = f"{self} reached {state_name}, overshot target {self.target.name}"
                self.to(self.raised)
                return RuntimeError(msg)

            def pursue(self, to_state, verb, *, tx_cls=None, setup=None, target=None, unblock=True):
                if self.lazy_work is not None:
                    raise RuntimeError(f"can't {verb}, haven't run {self.lazy_verb} yet")

                if self.state is None:
                    self.initialize()
                elif self.state in (self.parked, self.finished):
                    self.reactivate()
                # NESTING is treated like ACTIVE here: pursue continues
                # the user's driving of the child without losing the
                # parent context on self.stack.  Reactivate would clear
                # the stack -- and besides, reactivate only works from
                # a terminal state.

                # Refresh self.tx: while the score lock was released
                # in a prior score.wait (e.g. inside settle), the
                # worker may have completed and unregistered our
                # cached tx.  If so, the cached pointer is stale --
                # tx.state has advanced past anything Driver handles.
                # Transition to FINISHED so the dispatch cycle moves
                # us through reactivate -> IDLE -> Terminated cleanly,
                # instead of looping on a stale terminal tx.
                #
                # CONTRACT: pursue is currently the only external
                # entry point on Driver that fires after a window
                # in which the score lock was released.  If you add
                # another such entry point (an imperative that
                # doesn't funnel through pursue), it must perform
                # the same self.cache_tx() + transition-on-None
                # refresh before reading self.tx -- otherwise the
                # cached pointer may be stale and the state machine
                # will misbehave.
                if self.state in (self.active, self.nesting):
                    self.cache_tx()
                    if self.tx is None:
                        return self.to(self.finished)

                tx = self.tx

                if self.state not in (self.active, self.nesting):
                    raise RuntimeError(f"can't {verb}, currently in {self.state}")
                if tx is None:
                    raise RuntimeError(f"can't {verb}, no current tx")  # pragma: no cover -- defensive: active state implies non-None tx
                if tx_cls and not isinstance(tx, tx_cls):
                    raise RuntimeError(
                        f"can't {verb}, tx doesn't park in {target.name} state, tx={tx!r}")

                # Driver is sophisticated enough to take a tx that's
                # already in flight and drive it to a target state.
                # Including the parked state PAUSED: if the caller's
                # target IS the parked state, the cascade yields parked
                # immediately on first signal; if the target is past
                # the parked state, the cascade auto-unparks via the
                # transit handler.  We only refuse if the tx is past
                # any state Driver can handle.
                if tx.state not in (State.BLOCKED, State.COMMIT, State.WAITING, State.STALLED, State.PAUSED):
                    raise RuntimeError(f"can't {verb}, tx currently in {tx.state}")  # pragma: no cover -- defensive: pursue only reachable while tx is in a driver-handleable state

                # Eagerly advance past BLOCKED -- saves one round of
                # wake-up-and-analyze-signals when the cascade kicks
                # in.  Safe because BLOCKED is never a parking
                # target (no imperative sets target=BLOCKED), so the
                # tx never wants to stay there.  We could do the
                # same for STALLED and PAUSED, but those rarely fire
                # (STALLED is post-notify, PAUSED is post-prior-
                # drive; neither dominates the common case the way
                # BLOCKED does), PAUSED's frog-march is more
                # involved (clear flag + zero counter + unpark),
                # and signal()'s elif chain already covers all
                # three uniformly -- so we only optimize the cheap,
                # high-frequency case here.
                unblock = unblock and tx.state is State.BLOCKED

                # Everything an imperative does -- mutate driver
                # state (target, state-via-to), mutate tx state
                # (unblock), run imperative-specific setup (e.g.
                # pause's pausing-counter increment) -- is staged as
                # a closure on self.lazy_work.  The closure fires
                # when the driver actually starts being driven: by
                # Driver.__call__ at entry, or by Dispatch.__next__'s
                # drain of recent.  A driver is uniformly lazy:
                # nothing observable changes on the driver or its tx
                # between the imperative call and the drive.
                def lazy_work():
                    if setup is not None:
                        setup()
                    if target is not None:
                        self.target = target
                    if unblock:
                        tx.unblock()
                    self.to(to_state)

                self.lazy_work = lazy_work
                self.lazy_verb = verb


            def skip(self):
                return self.pursue(self.skipping, 'skip')

            def block(self):
                return self.pursue(self.parked, 'block', unblock=False)

            def commit(self):
                return self.pursue(self.parking, 'commit', target=State.COMMIT, tx_cls=self.score.Core.TimeoutTransaction)

            def wait(self):
                return self.pursue(self.parking, 'wait', target=State.WAITING, tx_cls=self.score.WaitingTransaction)

            def stall(self):
                return self.pursue(self.parking, 'stall', target=State.STALLED, tx_cls=self.score.Core.StallingTransaction)

            def pause(self):
                # User-facing: incref pausing AND set the user pause
                # flag, so the user can later release via the matching
                # api.unpause / tx.api.pause = False decref.  Stage
                # the tx-side mutations in pursue's setup hook so
                # everything fires together when the driver is run.
                # Note we read self.base inside setup() (at fire
                # time), not here at imperative-call time: with lazy
                # Driver init, self.base may not be set until pursue
                # runs initialize.
                def setup():
                    base = self.base
                    base.pause = True
                    base.pausing += 1
                return self.pursue(self.parking, 'pause',
                                   target=State.PAUSED, setup=setup)

            def pausing(self):
                # Blanket-internal: incref pausing without touching
                # the pause flag.  Used by scheduler-side code (Cycle
                # init, Lock.relay) that holds a tx at PAUSED on its
                # own bookkeeping and arranges its own matching
                # decref (via tx.unpausing on hand-off, or frog-march
                # on drive-past).  Not exposed on the Driver api.
                def setup():
                    self.base.pausing += 1
                return self.pursue(self.parking, 'pausing',
                                   target=State.PAUSED, setup=setup)

            def finish(self):
                return self.pursue(self.finishing, 'finish')

            def nested(self):
                """Arm nested-aware mode: the next Nested fire transitions
                the Driver to the NESTING state (which has empty signals,
                so the Driver yields for caller review) instead of the
                default skipping/delegate-parking behavior.

                Independent of pursue: doesn't stage lazy_work, doesn't
                change state to a driving state, doesn't pick a target.
                What it does:
                  - Reactivates / initializes if the Driver is in a
                    terminal-ish state (parked, finished, nesting, or
                    None), so the arm lands on a live active session.
                  - Sets self.nested_armed = True.
                  - Adds Nested to self.signals if not already present
                    so the Driver actually listens for Nested while
                    yielded in active state.

                Composes with the six pursue-based imperatives:
                  - Called alone: Driver waits in active state on
                    Nested; on fire, transitions to nesting and yields.
                  - Called after pursue (e.g. driver.pause(); driver.nested()):
                    pursue's parking state already has Nested in its
                    signal set; the arm changes routing of the fire
                    so handler picks "yield to nesting" over default
                    "skip the child".
                  - Called before pursue (driver.nested(); driver.pause()):
                    nested() adds Nested to active-state signals (a
                    no-op if pursue is about to re-set them), and the
                    arm flag persists across the pursue's state
                    transition because pursue doesn't touch it.

                Like the pursue-based imperatives, nested() refuses
                back-to-back calls without an intervening drive: a
                second call before the first arm has been consumed
                raises RuntimeError, mirroring pursue's lazy_work
                guard ("haven't run {verb} yet").

                One-shot: cleared when consumed by a Nested fire.
                Re-arm by calling .nested() again.  Cleared on
                reactivate/initialize (a fresh active session starts
                unarmed).
                """
                if self.nested_armed:
                    raise RuntimeError(
                        "can't nested, arm is still set; "
                        "drive the Driver to consume it first")
                if self.state in (self.parked, self.finished):
                    self.reactivate()
                elif self.state is None:
                    self.initialize()
                # NESTING is handled like ACTIVE here: we re-arm on
                # top of the existing pursue-context-on-stack so a
                # subsequent Nested fire (e.g. the child spawns its
                # own grandchild) yields cleanly.  No reactivate,
                # which would wipe the stack.
                # If initialize moved us through to a terminal (e.g.
                # tx was already done), nothing to arm against.
                if self.state in self.terminal_states:
                    return
                self.nested_armed = True
                # Make sure Nested is in our listened-for signal set.
                # In a driving state it's already there; in active
                # state it isn't until we add it here.
                nested_signal = self.thread_signal.get(Nested)
                if nested_signal is not None and nested_signal not in self.signals:
                    self.signals = frozenset(self.signals | {nested_signal})

            def __call__(self):
                """Standalone alternative to Dispatch iteration: block
                until this Driver yields (becomes ACTIVE or reaches a
                terminal state).  Loops calling score.wait on the
                Driver's published signals and feeding the fired
                signals back to self.signal; exits when self.signals
                goes empty, which is the same "ready to yield" marker
                Dispatch uses."""
                if self.done:
                    self.reactivate()
                self.drive()
                while self.signals:
                    fired = set(self.score.wait(self.signals))
                    self.signal(fired)


        @base()
        @BoundInnerClass
        class Chain:
            """An ordered chain of Drivers for use with Dispatch.

            A Chain holds a current Driver (or None) and a pending
            list of Drivers waiting their turn.  Pass a Chain to
            Dispatch.add() and Dispatch will drive the current, then
            promote the next pending to current on each yield, until
            the chain is empty.

            Ownership rules: a Driver can be claimed by exactly one
            owner at a time (a Dispatch directly, or a Chain).  A
            Chain itself can be claimed by at most one Dispatch.  The
            owner interface (register / unregister / .owner) mirrors
            Driver's so the two compose uniformly.

            Chain.append() registers each driver with the Chain;
            Chain.remove() unregisters.  When Dispatch promotes
            pending[0] to current, ownership of the promoted driver
            transfers from Chain to Dispatch for the duration of the
            drive.

            A Chain can be appended to / removed from at any time,
            including during iteration of the owning Dispatch (the
            score lock serializes the operations).  An empty Chain
            stays owned by its Dispatch -- appending more Drivers
            after it goes empty re-engages the chain on the next
            iteration.  To detach a Chain call Dispatch.remove(chain):
            the chain's current Driver (if any) is unregistered and
            dropped from the Dispatch, but the pending list is left
            intact so the Chain remains useful (add it to another
            Dispatch and the pending head activates).
            """

            def __init__(self, score, *drivers):
                self.score = score
                self.pending = deque()
                self.owner = None
                for d in drivers:
                    self.append(d)

            def __repr__(self):
                # Pending drivers are owned by this Chain, so their
                # .owner is this Chain.  Calling repr() on them would
                # recurse back through the Chain's repr via the
                # Driver's owner field.  Render them as short labels
                # (thread name + state) to break the cycle.  The
                # owner field of the Chain itself is safe to repr:
                # it's either None or a Dispatch, whose repr doesn't
                # recurse.
                def short(d):
                    state = d.state.name if d.state is not None else 'uninitialized'
                    return f"<Driver {d.thread.name!r} {state}>"
                pending = ', '.join(short(d) for d in self.pending)
                return f"<Chain owner={self.owner!r} pending=[{pending}]>"

            def __len__(self):
                # Number of Drivers waiting in this chain.  Drivers
                # that have been promoted out of the chain (via
                # promote() or iteration) are no longer counted --
                # they've been transferred to the consumer.
                return len(self.pending)

            def __bool__(self):
                # Truthy iff there's at least one pending Driver.
                return bool(self.pending)

            def register(self, owner):
                if self.owner is not None:
                    raise RuntimeError(f"chain is already owned by {self.owner!r}")
                self.owner = owner

            def unregister(self):
                if self.owner is None:
                    raise RuntimeError(f"chain is not owned")
                self.owner = None

            def append(self, driver):
                """Add `driver` to the pending list.  Raises if `driver`
                is already claimed by any owner (this Chain, another
                Chain, or a Dispatch).  If this Chain is owned by a
                Dispatch and was previously empty, notify the owner
                so its next drain promotes us."""
                driver.register(self)
                was_empty = not self.pending
                self.pending.append(driver)
                if was_empty and self.owner is not None:
                    # Chain becomes head-promotable.  Put us in the
                    # owner's recent so the next drain promotes.
                    self.owner.recent.append(self)

            def remove(self, driver):
                """Remove `driver` from the pending list and unregister
                it.  Raises ValueError if `driver` is not in pending.
                (Use Dispatch.remove(chain) to detach an entire chain
                from a Dispatch -- the chain's pending list stays
                intact so the Chain remains useful.)"""
                for i, d in enumerate(self.pending):
                    if d is driver:
                        del self.pending[i]
                        d.unregister()
                        return
                raise ValueError(f"driver {driver!r} not in Chain.pending")

            def __contains__(self, driver):
                return any(d is driver for d in self.pending)

            def promote(self):
                """Pop the pending head and unregister it from this
                Chain.  Returns the Driver (now unowned) or None if
                pending is empty.

                This is the Chain's promotion primitive: both the
                standalone __next__ iteration path and the Dispatch's
                drain_recent use it to remove the head from this
                Chain's management before driving it.  The caller is
                responsible for any subsequent re-registration (e.g.
                Dispatch.register(self) on the returned Driver) and
                for driving it; the Chain itself is now neutral on
                the Driver's fate.

                Stateless w.r.t. iteration: nothing in this Chain
                tracks the just-promoted Driver, so breaking out of
                iteration leaves no half-state behind.  Caller must
                hold score.lock."""
                if not self.pending:
                    return None
                d = self.pending.popleft()
                d.unregister()
                return d

            def __iter__(self):
                return self

            def __next__(self):
                """Yield each Driver in pending order: promote the
                pending head, drive it until it yields, return the
                (now-unowned) Driver to the caller.  After return, the
                Driver is no longer tracked by this Chain; the caller
                can examine state, issue further imperatives, drive
                d() manually, then resume iteration.  Re-adding the
                Driver via chain.append(d) puts it back at the tail.

                Raises if this Chain is owned (by a Dispatch): in
                that case the owner is responsible for driving, and
                iterating directly would race the owner.  Iteration
                also works when pending is currently empty -- the
                Chain becomes a do-nothing iterator that raises
                StopIteration immediately.  Caller must hold
                score.lock (the lock is released during the drive's
                internal score.wait calls).

                Break-safety: after promote() pops the head and
                before drive completes, the Chain no longer tracks
                the popped Driver.  An exception during drive (or
                a break-out-of-iteration after yield) leaves no
                half-state in the Chain.  The popped Driver is
                unowned; the caller must close it explicitly if it
                wants to dispose."""
                if self.owner is not None:
                    raise RuntimeError(
                        f"can't iterate Chain directly, it's owned "
                        f"by {self.owner!r}")
                d = self.promote()
                if d is None:
                    raise StopIteration
                d()  # drive; lock released during score.wait inside
                return d

            def close(self):
                """Close all Drivers in this Chain's pending list and
                empty it.  Drivers that have been promoted out of the
                Chain (via iteration or Dispatch handling) are owned
                by their new consumer and are not touched here --
                close them via the consumer (e.g. Dispatch.close()).
                Idempotent."""
                while self.pending:
                    d = self.pending.popleft()
                    d.close()
                    d.unregister()


        @base()
        @BoundInnerClass
        class Dispatch:
            def __init__(self, score):
                self.score = score
                self.drivers = defaultdict(set)
                self.queue = deque()
                # When a Driver is the `current` of a Chain owned by
                # this Dispatch, an entry maps the Driver to the Chain
                # so __next__ (and Driver-level remove/discard) can
                # advance the Chain when the Driver leaves the active
                # set.
                self.driver_to_chain = {}
                # Items (Drivers or Chains) added since the last
                # drain.  add_* methods are pure bookkeeping
                # (register ownership + append here); the actual
                # state changes -- driver.drive() and chain head-
                # promotion -- happen when __next__ drains recent
                # at the top of each iteration.  This keeps
                # add+remove observationally side-effect-free on
                # the items themselves: a driver.pause() followed
                # by dispatch.add(driver); dispatch.remove(driver)
                # never fires the staged tx.unblock.
                self.recent = deque()

            def __repr__(self):
                return f"<Dispatch {len(self.drivers)} drivers {len(self.queue)} ready>"

            def add(self, o):
                if isinstance(o, self.score.Chain):
                    chain = o
                    chain.register(self)
                    self.recent.append(chain)
                    return

                driver = o
                if (driver in self.drivers
                    or driver in self.queue
                    or any(item is driver for item in self.recent)):
                    return
                driver.register(self)
                self.recent.append(driver)

            def advance_chain_after(self, driver):
                """Called when `driver` (a Driver that was promoted out
                of some Chain owned by this Dispatch) has just left
                the active set.  Drops the driver-to-chain link; if
                the Chain still has pending Drivers, puts the chain
                back in recent so the next __next__ drain promotes
                the new head."""
                chain = self.driver_to_chain.pop(driver)
                if chain.pending:
                    self.recent.append(chain)

            def update(self, items):
                for item in items:
                    self.add(item)

            def discard_driver(self, driver):
                """Discard a single Driver.  If the Driver is the
                current of a Chain owned by this Dispatch, the Chain
                also advances (pending head activates if any).
                Pending Drivers aren't claimed by Dispatch -- they
                aren't reachable here; the not-in-active-set path
                falls through to a no-op."""
                in_drivers = driver in self.drivers
                in_queue = driver in self.queue
                in_recent = any(item is driver for item in self.recent)
                assert not (in_drivers and in_queue)
                if not (in_drivers or in_queue or in_recent):
                    return
                if in_recent:
                    self.recent = deque(
                        item for item in self.recent if item is not driver)
                if in_drivers:
                    del self.drivers[driver]
                elif in_queue:
                    self.queue.remove(driver)
                driver.unregister()
                if driver in self.driver_to_chain:
                    self.advance_chain_after(driver)

            def discard_chain(self, chain):
                """Detach a Chain from this Dispatch.  Any Driver
                this Dispatch promoted out of the Chain (tracked via
                driver_to_chain) is removed from the active set and
                unregistered, freeing it to be re-claimed elsewhere.
                The Chain's pending list is left untouched: those
                Drivers stay owned by the Chain, so a follow-up
                Dispatch.add(chain) activates the pending head
                normally."""
                if chain.owner is not self:
                    return
                if any(item is chain for item in self.recent):
                    self.recent = deque(
                        item for item in self.recent if item is not chain)
                # Find any Driver this Dispatch promoted from this
                # Chain (driver_to_chain reverse-maps).  At most one
                # such Driver exists at a time (Dispatch promotes one
                # head per chain, drives it, and advances on yield),
                # but the loop tolerates none or one without special
                # casing.
                promoted = [d for d, c in self.driver_to_chain.items() if c is chain]
                for d in promoted:
                    del self.driver_to_chain[d]
                    if d in self.drivers:
                        del self.drivers[d]
                        d.unregister()
                    elif d in self.queue:
                        self.queue.remove(d)
                        d.unregister()
                chain.unregister()

            def discard(self, obj):
                if isinstance(obj, self.score.Chain):
                    self.discard_chain(obj)
                else:
                    self.discard_driver(obj)

            def remove(self, obj):
                if isinstance(obj, self.score.Chain):
                    if obj.owner is not self:
                        raise ValueError(f"unknown Chain {obj!r}")
                    self.discard_chain(obj)
                    return
                # Driver case.  Pending-in-a-Chain Drivers aren't in
                # any Dispatch structure, so they're "unknown" here
                # and raise.  Current-of-a-Chain Drivers go through
                # the normal discard path which advances the Chain.
                in_drivers = obj in self.drivers
                in_queue = obj in self.queue
                in_recent = any(item is obj for item in self.recent)
                if not (in_drivers or in_queue or in_recent):
                    raise ValueError(f"unknown driver {obj!r}")
                self.discard_driver(obj)

            def __contains__(self, obj):
                if isinstance(obj, self.score.Chain):
                    return obj.owner is self
                if obj in self.drivers or obj in self.queue:
                    return True
                return any(item is obj for item in self.recent)

            def __iter__(self):
                return self

            def drain_recent(self):
                """Apply state changes deferred from add() calls and
                from chain-advance bookkeeping.  Drivers get drive()-
                fired and placed in drivers / queue per their
                signals; chains promote their head (if any) -- the
                promoted driver is itself enqueued in recent for
                same-drain processing.  Internal: called at the top
                of __next__.  External callers drive a dispatch by
                iterating it."""
                while self.recent:
                    item = self.recent.popleft()
                    if isinstance(item, self.score.Chain):
                        d = item.promote()
                        if d is not None:
                            d.register(self)  # now owned by this Dispatch
                            self.driver_to_chain[d] = item
                            self.recent.append(d)
                        continue
                    # Driver.
                    if item.done:
                        item.reactivate()
                    item.drive()
                    signals = item.signals
                    if signals:
                        self.drivers[item] = signals
                    else:
                        self.queue.append(item)

            def __next__(self):
                while True:
                    self.drain_recent()
                    if self.queue:
                        driver = self.queue.popleft()
                        driver.unregister()
                        # If the yielded driver was the current of a
                        # Chain, advance the chain: clear its current
                        # and, if pending isn't empty, put the chain
                        # back in recent so the next iteration's drain
                        # promotes the new head.  Chain stays owned
                        # by this Dispatch even when it ends up empty.
                        if driver in self.driver_to_chain:
                            self.advance_chain_after(driver)
                        return driver

                    if not self.drivers:
                        raise StopIteration

                    all_signals = set()
                    signal_to_driver = {}
                    for driver, signals in self.drivers.items():
                        assert not (all_signals & signals)
                        all_signals |= signals
                        for signal in signals:
                            signal_to_driver[signal] = driver

                    signaled = list(self.score.wait(all_signals))

                    ready = defaultdict(set)
                    for signal in signaled:
                        driver = signal_to_driver[signal]
                        ready[driver].add(signal)

                    for driver, signals in ready.items():
                        if driver not in self.drivers:
                            # driver was removed while we were asleep
                            continue

                        signals = driver.signal(signals)
                        if signals:
                            self.drivers[driver] = signals
                        else:
                            del self.drivers[driver]
                            self.queue.append(driver)

            def close(self):
                """Close all objects (Chains and Drivers) inside
                this Dispatch and clear the Dispatch.  Idempotent."""
                # Snapshot first; collections get cleared below.
                drivers = list(self.drivers) + list(self.queue)
                chains = []
                for item in self.recent:
                    if isinstance(item, self.score.Chain):
                        chains.append(item)
                    else:
                        drivers.append(item)
                # driver_to_chain values are chains; some may not
                # be in `chains` (their entries got registered
                # before drain_recent processed them out of
                # recent).  Collect those too.
                for chain in self.driver_to_chain.values():
                    if chain not in chains:
                        chains.append(chain)

                # Clear our state now so nothing reentered finds
                # half-cleaned bookkeeping.
                self.drivers.clear()
                self.queue.clear()
                self.recent.clear()
                self.driver_to_chain.clear()

                for d in drivers:
                    d.close()
                    if d.owner is self:
                        d.unregister()
                for c in chains:
                    if c.owner is self:
                        c.unregister()
                    c.close()


        def parse_park_skip_args(self, args, caller):
            """Parse park/skip args into a list of (thread, [method+]) tuples."""
            if not args:
                raise ValueError(f"{caller}: no thread specified")

            is_park = caller == 'park'
            plan = []
            seen = set()
            current = None
            thread = None
            last_was_thread = False

            for arg in args:
                if isinstance(arg, threading.Thread):
                    # thread
                    if (not current) and (current is not None):
                        raise ValueError(f"{caller}: thread {thread.name!r} has no method specified")
                    thread = arg
                    last_was_thread = True
                    if is_park:
                        if thread in seen:
                            raise ValueError(f"{caller}: thread {thread.name!r} specified more than once")
                        seen.add(thread)
                    current = []
                    plan.append((thread, current))
                    continue

                # method
                if not thread:
                    raise ValueError(f"{caller}: first argument must be a thread")
                if not isinstance(arg, MethodType):
                    raise ValueError(f"{caller}: expected thread or method, got {arg!r}")
                method = arg
                last_was_thread = False
                primitive = method.__self__
                valid = isinstance(primitive, self.api.Primitive)
                if valid:
                    core = primitive._core
                    valid = primitive is core.primitive
                if not valid:
                    raise ValueError(f"{caller}: {method.__name__!r} isn't a regulated method call (raw or foreign primitive?)")

                if is_park and len(current):
                    raise ValueError(f"{caller}: thread {thread.name!r} must be followed by exactly one method")
                current.append(method)

            if last_was_thread:
                raise ValueError(f"{caller}: thread {thread.name!r} has no method specified")

            return plan

        def park(self, *args, wait=False):
            """Park each thread at its specified method.

            Called with score.lock held.  Returns a dict mapping each
            thread to the parked transaction.  See Scenario.park for
            full docs.

            Per thread, wait for the thread to push the named method's
            tx, validate it's at BLOCKED, and stop there.  With
            wait=True, also unblock and drive each tx to terminal
            concurrently at the end.  With wait=False, the tx is left
            at BLOCKED for the caller to inspect or unblock manually.
            """
            caller = 'park'
            plan = self.parse_park_skip_args(args, caller)
            result = {}

            drivers = []
            try:
                for thread, methods in plan:
                    first = methods[0]   # park: exactly one method per thread

                    d = self.Driver(thread)
                    drivers.append(d)

                    # Wait for the thread to be on a tx; d() drives
                    # idle or active through to a yielding state.
                    # After d() the Driver is in active (with a
                    # current tx) or terminated (thread ended before
                    # a tx appeared).
                    d()
                    if d.state is d.terminated:
                        raise RuntimeError(
                            f"{caller}: thread {thread.name!r} terminated "
                            f"before reaching {first.__name__!r}")
                    assert d.state is d.active
                    if d.tx.method != first:
                        raise RuntimeError(
                            f"{caller}: thread {thread.name!r} pushed "
                            f"{d.tx.method.__name__!r}, expected "
                            f"{first.__name__!r}")
                    if d.tx.state != State.BLOCKED:
                        raise RuntimeError(
                            f"{caller}: thread {thread.name!r} has active tx "
                            f"on {first.__name__!r} but it is in "
                            f"{d.tx.state.name} state, not BLOCKED state")

                    result[thread] = d.tx

                if wait:
                    # Drive every parked tx to terminal concurrently.
                    # Sequential per-thread drive would deadlock when
                    # one thread's target depends on another being
                    # unblocked first.
                    for d in drivers:
                        d.finish()
                    dispatch = self.Dispatch()
                    for d in drivers:
                        dispatch.add(d)
                    for yielded in dispatch:
                        assert yielded.state is yielded.finished
                else:
                    # Detach every Driver via d.block(): parks Driver
                    # at BLOCKED (terminal, slot released) while
                    # leaving the tx itself at BLOCKED for the
                    # caller to inspect / unblock / unpark.
                    for d in drivers:
                        d.block()
            finally:
                # On success every Driver is already terminal (close()
                # ran on entering parked / finished) and this loop is
                # a no-op.  On the exception path it closes whichever
                # Drivers are still holding slots so a follow-up
                # skip / park / cycle doesn't collide.
                for d in drivers:
                    if not d.done:
                        d.close()

            return result

        def skip(self, *args, wait=False):
            """Drive each thread through a sequence of methods.

            Called with score.lock held.  See Scenario.skip for docs.
            Returns a dict mapping each thread to its last tx.

            Per thread, drive through the listed methods in order:
            wait for each method's tx to appear, validate it's the
            expected method, drive it past its terminal, then proceed
            to the next.  The "final" method for each thread (the last
            method of the last segment naming that thread) is treated
            differently:
              - wait=True: defer the drive-to-terminal and drain all
                threads' final drivers concurrently via dispatch.
              - wait=False: park the Driver at BLOCKED (terminal, slot
                released) and unblock the tx by hand so the worker
                proceeds.  The caller waits via r[thread] if desired.
            """
            caller = 'skip'
            plan = self.parse_park_skip_args(args, caller)
            result = {}
            last_drivers = []

            # Pre-compute which (segment_idx, method_idx) is the final
            # operation for each thread.  Repeated thread segments
            # are allowed; only the last method of the last segment
            # is "final" for that thread.  Intermediate methods
            # (including the last method of a non-final segment) are
            # driven to terminal inline.
            final_per_thread = {}
            for i, (thread, methods) in enumerate(plan):
                final_per_thread[thread] = (i, len(methods) - 1)

            drivers = []
            try:
                for i, (thread, methods) in enumerate(plan):
                    d = self.Driver(thread)
                    drivers.append(d)

                    for j, method in enumerate(methods):
                        # Bring the Driver to active on the next tx.
                        # After a prior d.finish() + d() in this
                        # segment the Driver is in finished and needs
                        # reactivate; on the first iteration it's
                        # active/idle/terminated and reactivate
                        # doesn't apply.  d() drives idle or
                        # finishing-after-reactivate through to
                        # whatever's next.
                        if d.state is d.finished:
                            d.reactivate()
                        d()
                        if d.state is d.terminated:
                            raise RuntimeError(
                                f"{caller}: thread {thread.name!r} terminated "
                                f"before reaching {method.__name__!r}")
                        assert d.state is d.active
                        if d.tx.method != method:
                            raise RuntimeError(
                                f"{caller}: thread {thread.name!r} pushed "
                                f"{d.tx.method.__name__!r}, expected "
                                f"{method.__name__!r}")

                        if final_per_thread[thread] == (i, j):
                            result[thread] = d.tx
                            if wait:
                                # Defer drive-to-terminal so all
                                # threads' final txs can run
                                # concurrently in the drain below.
                                d.finish()
                                last_drivers.append(d)
                            else:
                                # Detach the Driver from the tx: park
                                # at BLOCKED (terminal, close()
                                # releases the slot), then unblock by
                                # hand so the worker proceeds.  Caller
                                # can wait on r[thread] if desired.
                                d.block()
                                d.tx.unblock()
                        else:
                            # Non-final method: drive to terminal
                            # inline.  The current tx must end before
                            # the worker pushes the next method's tx,
                            # so the next iteration's d() can find it.
                            d.finish()
                            d()

                if wait:
                    # Drive all final txs to terminal concurrently.
                    dispatch = self.Dispatch()
                    for d in last_drivers:
                        dispatch.add(d)
                    for yielded in dispatch:
                        assert yielded.state is yielded.finished
            finally:
                # On success every Driver is already terminal (closed
                # itself via Driver.to() on entering parked / finished
                # / etc.) and this loop is a no-op.  On the exception
                # path it closes whichever Drivers are still holding
                # slots so a follow-up skip / park / cycle doesn't
                # collide with leftover slot claims.
                for d in drivers:
                    if not d.done:
                        d.close()

            return result

        def finish(self, *threads):
            """Best-effort drive of each thread to terminated state.
            See Scenario.finish docs.  Called with score.lock held.
            """
            drivers = []
            dispatch = self.Dispatch()
            try:
                for t in threads:
                    d = self.Driver(t)
                    drivers.append(d)
                    # If there's a current tx, push finish so the
                    # dispatch can drive it.  Idle drivers (no current
                    # tx) just join the dispatch; when the thread
                    # eventually pushes a tx the thread signal fires
                    # and the driver advances to active, at which
                    # point the yield-handling loop below pushes
                    # finish on it.
                    if d.state is d.active:
                        d.finish()
                    dispatch.add(d)
                for d in dispatch:
                    if d.state is d.terminated:
                        continue
                    # Thread isn't done -- might be at a new tx
                    # (active), or just finished a tx and waiting to
                    # push the next (parked/finished/raised/idle).
                    # Try to push finish; pursue auto-reactivates
                    # from parked/finished but not raised/idle.
                    try:
                        if d.state is d.raised:
                            d.reactivate()
                        d.finish()
                    except RuntimeError:
                        # Can't drive (no current tx, or in a state
                        # finish doesn't handle).  Re-add to dispatch
                        # so it can wake on the next thread signal /
                        # push of a tx.
                        pass
                    dispatch.add(d)
            finally:
                # Close any Driver still holding a slot.
                for d in drivers:
                    if not d.done:
                        d.close()

        @BoundInnerClass
        class LockedDictProxy:
            """Read-only dict proxy that automatically locks on every access."""

            def __init__(self, score, d=None):
                self._lock = score.lock
                if d is None:
                    d = {}
                self._dict = d

            def __repr__(self):
                with self._lock:
                    return repr(self._dict)

            def get(self, key, default=None):
                with self._lock:
                    return self._dict.get(key, default)

            def __getitem__(self, key):
                with self._lock:
                    return self._dict[key]

            def __contains__(self, key):
                with self._lock:
                    return key in self._dict

            def __iter__(self):
                with self._lock:
                    return iter(self._dict)

            def keys(self):
                with self._lock:
                    return self._dict.keys()

            def values(self):
                with self._lock:
                    return self._dict.values()

            def items(self):
                with self._lock:
                    return self._dict.items()

        @BoundInnerClass
        class CoreAttrMapProxy:
            """Read-only mapping of primitive to objects stored as weakrefs in a score attribute.

            Used to expose scenario.apis (attr='api') and
            scenario.raws (attr='raw') as live views without holding
            primitives strongly: the only registry is score.cores
            (a WeakSet), so primitives are GC'd naturally when the
            user releases them.

            Keys are the live primitives; lookups resolve via
            primitive._core.<attr>.  Membership and iteration go
            through score.cores so dropped primitives disappear
            without bookkeeping.
            """

            def __init__(self, score, attr):
                self._lock = score.lock
                self._score = score
                self._attr = attr

            def _live_cores(self):
                # Snapshot under the lock to avoid weak-set churn
                # during iteration.
                return list(self._score.cores)

            def get(self, primitive, default=None):
                with self._lock:
                    core = getattr(primitive, '_core', None)
                    if core is None or core not in self._score.cores:
                        return default
                    return getattr(core, self._attr)

            def __getitem__(self, primitive):
                with self._lock:
                    core = getattr(primitive, '_core', None)
                    if core is None or core not in self._score.cores:
                        raise KeyError(primitive)
                    return getattr(core, self._attr)

            def __contains__(self, primitive):
                with self._lock:
                    core = getattr(primitive, '_core', None)
                    return core is not None and core in self._score.cores

            def __iter__(self):
                with self._lock:
                    cores = self._live_cores()
                return iter(c.primitive for c in cores)

            def __len__(self):
                return len(self._score.cores)

            def keys(self):
                with self._lock:
                    return [c.primitive for c in self._live_cores()]

            def values(self):
                with self._lock:
                    return [getattr(c, self._attr) for c in self._live_cores()]

            def items(self):
                with self._lock:
                    return [(c.primitive, getattr(c, self._attr))
                            for c in self._live_cores()]

            def __repr__(self):
                with self._lock:
                    return repr({c.primitive: getattr(c, self._attr)
                                 for c in self._live_cores()})

        @BoundInnerClass
        class ReadOnlyListProxy:
            """Read-only list proxy."""

            def __init__(self, score, list_obj):
                self._list = list_obj

            def __repr__(self):
                return f"ReadOnlyListProxy({list(self._list)!r})"

            def __contains__(self, item):
                return item in self._list

            def __getitem__(self, key):
                return self._list[key]

            def __len__(self):
                return len(self._list)

            def __iter__(self):
                return iter(self._list)

            def __reversed__(self):
                return reversed(self._list)

            def __eq__(self, other):
                if isinstance(other, self.__class__):
                    return list(self._list) == list(other._list)
                return list(self._list) == other

            def __ne__(self, other):
                return not self.__eq__(other)

            def __bool__(self):
                return bool(self._list)

            def copy(self):
                return list(self._list)

            def count(self, value):
                return list(self._list).count(value)

            def index(self, value, start=0, stop=None):
                if stop is None:
                    stop = len(self._list)
                return list(self._list).index(value, start, stop)

        @BoundInnerClass
        class LogProxy(ReadOnlyListProxy):
            """Read-only list proxy for the transaction log, with clear() support."""

            def __repr__(self):
                return f"LogProxy({list(self._list)!r})"

            def clear(self):
                self._list.clear()

        @BoundInnerClass
        class LockedSetProxy:
            """A read-write set-like proxy to a dict that locks on all operations."""

            def __init__(self, score, dict=None, change=None):
                self._lock = score.lock
                if dict is None:
                    dict = {}
                self._dict = dict
                self.__change__ = change

            def _notify(self, added=(), removed=()):
                if self.__change__ and (added or removed):
                    self.__change__(added, removed)

            def add(self, object):
                with self._lock:
                    if object not in self._dict:
                        self._dict[object] = None
                        self._notify(added={object})

            def remove(self, object):
                with self._lock:
                    del self._dict[object]
                    self._notify(removed={object})

            def discard(self, object):
                with self._lock:
                    if object in self._dict:
                        del self._dict[object]
                        self._notify(removed={object})

            def pop(self):
                with self._lock:
                    object, _ = self._dict.popitem()
                    self._notify(removed={object})
                    return object

            def clear(self):
                with self._lock:
                    removed = set(self._dict)
                    self._dict.clear()
                    self._notify(removed=removed)

            def update(self, *others):
                with self._lock:
                    before = set(self._dict)
                    for other in others:
                        for object in other:
                            self._dict[object] = None
                    self._notify(added=set(self._dict) - before)

            def difference_update(self, *others):
                with self._lock:
                    before = set(self._dict)
                    for other in others:
                        for object in other:
                            self._dict.pop(object, None)
                    self._notify(removed=before - set(self._dict))

            def intersection_update(self, *others):
                with self._lock:
                    before = set(self._dict)
                    keep = before.intersection(*others)
                    for object in before - keep:
                        del self._dict[object]
                    self._notify(removed=before - set(self._dict))

            def symmetric_difference_update(self, other):
                with self._lock:
                    before = set(self._dict)
                    for object in other:
                        if object in self._dict:
                            del self._dict[object]
                        else:
                            self._dict[object] = None
                    after = set(self._dict)
                    self._notify(added=after - before, removed=before - after)

            def __ior__(self, other):
                self.update(other)
                return self

            def __iand__(self, other):
                self.intersection_update(other)
                return self

            def __isub__(self, other):
                self.difference_update(other)
                return self

            def __ixor__(self, other):
                self.symmetric_difference_update(other)
                return self

            def __contains__(self, object):
                with self._lock:
                    return object in self._dict

            def __len__(self):
                with self._lock:
                    return len(self._dict)

            def __bool__(self):
                with self._lock:
                    return bool(self._dict)

            def __iter__(self):
                with self._lock:
                    return iter(self._dict)

            def __repr__(self):
                with self._lock:
                    return f"LockedSetProxy({set(self._dict)!r})"

            def __str__(self):
                with self._lock:
                    return str(set(self._dict))

            def copy(self):
                with self._lock:
                    return set(self._dict)

            def issubset(self, other):
                with self._lock:
                    return set(self._dict).issubset(other)

            def issuperset(self, other):
                with self._lock:
                    return set(self._dict).issuperset(other)

            def isdisjoint(self, other):
                with self._lock:
                    return set(self._dict).isdisjoint(other)

            def __le__(self, other):
                with self._lock:
                    return set(self._dict) <= other

            def __lt__(self, other):
                with self._lock:
                    return set(self._dict) < other

            def __ge__(self, other):
                with self._lock:
                    return set(self._dict) >= other

            def __gt__(self, other):
                with self._lock:
                    return set(self._dict) > other

            def __eq__(self, other):
                with self._lock:
                    return set(self._dict) == other

            def __ne__(self, other):
                with self._lock:
                    return set(self._dict) != other

            def union(self, *others):
                with self._lock:
                    return set(self._dict).union(*others)

            def intersection(self, *others):
                with self._lock:
                    return set(self._dict).intersection(*others)

            def difference(self, *others):
                with self._lock:
                    return set(self._dict).difference(*others)

            def symmetric_difference(self, other):
                with self._lock:
                    return set(self._dict).symmetric_difference(other)

            def __or__(self, other):
                with self._lock:
                    return set(self._dict) | other

            def __and__(self, other):
                with self._lock:
                    return set(self._dict) & other

            def __sub__(self, other):
                with self._lock:
                    return set(self._dict) - other

            def __xor__(self, other):
                with self._lock:
                    return set(self._dict) ^ other

        @BoundInnerClass
        class ReadOnlyDictProxy:
            def __init__(self, score, transactions_dict):
                self._lock = score.lock
                self._dict = transactions_dict
                self._score = score

            def __getitem__(self, thread):
                with self._lock:
                    return self._dict[thread]

            def get(self, thread, default=None):
                with self._lock:
                    return self._dict.get(thread, default)

            def __contains__(self, thread):
                with self._lock:
                    return bool(self._dict.get(thread))

            def __len__(self):
                with self._lock:
                    return len(self._dict)

            def __bool__(self):
                with self._lock:
                    return bool(self._dict)

            def __iter__(self):
                with self._lock:
                    return iter(self._dict)

            def keys(self):
                with self._lock:
                    return self._dict.keys()

            def values(self):
                with self._lock:
                    return self._dict.values()

            def items(self):
                with self._lock:
                    return self._dict.items()

        @staticmethod
        def blocker_factory():
            blocker = threading.Lock()
            blocker.acquire()
            return blocker

        # This is implemented in the core--and yet it's exposed directly
        # to the user.  Just a more convenient implementation choice;
        # the ContextManager kinda straddles both worlds.
        @BoundInnerClass
        class ContextManager:
            def __init__(self, score):
                self.lock = score.lock
                self.score = score

            def __repr__(self):
                entered = "entered" if self.score.entered else "not entered"
                return f"<Scenario.ContextManager {entered}>"

            def __enter__(self):
                score = self.score
                with self.lock:
                    if score.log:
                        score.log.clear()
                    score.entered = True
                    for thread in score.managed:
                        try:
                            thread.start()
                        except RuntimeError:
                            pass
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                score = self.score
                with self.lock:
                    # 1: close any still-active Driver,
                    # so it releases its slot in score.drivers.
                    for d in list(score.drivers.values()):
                        if not d.done:
                            d.close()

                    # 2: deregulate.  New method calls on
                    # scenario primitives from this point on
                    # produce unregulated txs.
                    score.entered = False

                    # 3: unpark all scheduler-parked txs,
                    # so the program returns to normal operation.
                    for tx in tuple(score.transactions.values()):
                        tx.scenario_exit()

                    # 4a: clear the managed thread list.
                    threads = tuple(score.managed)
                    score.managed.clear()

                # 4b: join all the managed threads.
                for thread in threads:
                    thread.join()

                return False

        def thread(self, target, *args, **kwargs):
            """Create and register a managed thread."""
            thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
            self.managed[thread] = None
            if self.entered:
                thread.start()
            return thread

        def signal(self, item):
            """Wake any waiters parked on item.

            Every wait-item is Signaling: item.sample(scenario) is its
            source of truth, queried on demand.  signal() is purely a
            wakeup notification -- it confirms the item is currently
            high (waking on a low signal would be a bug) and releases
            each parked wtx.  No accumulation, no signaling set.

            item must already be in normalized form (the same form
            WaitTransaction installs as a waiters key).
            """
            assert isinstance(item, Signaling), (
                f"signal({item!r}): expected a Signaling instance; "
                "non-Signaling wait items are no longer supported")
            assert item.sample(self.api), (
                f"signal({item!r}): item is not high; "
                "Signaling items must be high when signaled")
            for wtx in self.waiters.pop(item, ()):
                if wtx.blocker is not None:
                    wtx.blocker()
                    wtx.blocker = None
                wtx.signaled.add(item)

        def signals(self, items):
            "Wake waiters for each item.  See signal()."
            for item in items:
                self.signal(item)

        def incref_usage(self, *items):
            """Increment the usage refcount for each item (a cooked
            primitive or a cooked bound method).  When an item's count
            goes 0->1 it just became used by some thread, so we signal
            its aggregate signal (Primitive(item) or BoundMethod(item))
            to wake any waiters parked on it.

            Items must already be cooked (normalized); callers pass the
            tx's cooked family, never raw handles."""
            for item in items:
                first = item not in self.usage_counter
                self.usage_counter[item] += 1
                if first:
                    self.signal(self._aggregate_signal(item))

        def decref_usage(self, *items):
            """Decrement the usage refcount for each item.  When an
            item's count reaches 0 it's no longer used by any thread,
            so we drop the key and signal Not(aggregate) to wake any
            waiters parked on "nobody is using this"."""
            for item in items:
                count = self.usage_counter[item] - 1
                if count:
                    self.usage_counter[item] = count
                else:
                    del self.usage_counter[item]
                    self.signal(Not(self._aggregate_signal(item)))

        def _aggregate_signal(self, item):
            "The aggregate Signaling for a cooked primitive or method."
            if isinstance(item, MethodType):
                return self.BoundMethod(item)
            return Primitive(item)

        class Thread(Signaling, ImmutableThreadSignalToken):
            """A private level signal: high while a thread has any
            active regulated transaction in the scenario.

            Nested in _ScenarioCore and not part of the public API.
            scenario.wait() boxes a bare top-level thread into
            Thread(thread) to use as a waiters key, and unboxes back
            to the user's thread on return.  Bare threads inside other
            signals (Not, Terminated, Call, Use) are NOT boxed -- those
            signals store and interpret bare threads directly.
            """
            __slots__ = ()

            def __new__(cls, thread):
                if not isinstance(thread, threading.Thread):
                    raise TypeError(
                        f"thread must be threading.Thread, not {thread!r}")
                return tuple.__new__(cls, (thread,))

            def sample(self, scenario):
                return self.thread in scenario._core.transactions

            def __repr__(self):
                return f"Thread({self[0].name!r})"

        class BoundMethod(Signaling, ImmutableSignalToken):
            """A private aggregate level signal: high while ANY thread
            is calling the wrapped bound method (normalized cooked, and
            conditions collapsed to their lock).

            Nested in _ScenarioCore and not part of the public API.
            scenario.wait() boxes a bare top-level bound method into
            BoundMethod(method), normalizing it, and unboxes back to
            the user's original method on return.
            """
            __slots__ = ()

            def __new__(cls, method):
                primitive = getattr(method, '__self__', None)
                if primitive is None or not hasattr(primitive, '_core'):
                    raise TypeError(
                        f"BoundMethod expected a bound method on a regulated "
                        f"primitive, got {method!r}")
                return tuple.__new__(cls, (method,))

            @property
            def method(self):
                return self[0]

            def sample(self, scenario):
                method = self.method
                score = scenario._core
                for tx in score.transactions.values():
                    cur = tx
                    while cur is not None:
                        if method in cur.call_methods():
                            return True
                        cur = cur.parent
                return False

            def normalized(self):
                cooked = _normalize_method(self.method)
                if cooked is self.method:
                    return self
                return type(self)(cooked)

            def __repr__(self):
                name = getattr(self.method, '__name__', repr(self.method))
                return f"BoundMethod({name})"

        def box_signal(self, o):
            """Box a bare top-level thread or bound method into its
            Signaling form.  Already-Signaling objects pass through.
            Signals are inert, so no scenario is stored."""
            if isinstance(o, Signaling):
                return o
            if isinstance(o, threading.Thread):
                return self.Thread(o)
            if isinstance(o, MethodType):
                return self.BoundMethod(o)
            raise TypeError(f"{o!r} is not signalable")

        def wait(self, items, timeout=None):
            """Wait for any item in items to signal.

            Returns the set of items that signaled.
            The set can be empty if wait times out.

            You must be holding the score lock when
            you call wait.  If wait blocks, it will
            temporarily release the score lock.
            """
            wtx = self.WaitTransaction(items)
            return wtx.wait(timeout=timeout)

        def reset(self):
            "Clear accumulated scenario state."
            self.log.clear()

        def transaction(self, thread):
            return self.transactions.get(thread)

        primitive_names = (
            'Lock',
            'RLock',
            'Condition',
            'Semaphore',
            'BoundedSemaphore',
            'Event',
            'Barrier',
            )

        @BoundInnerClass
        class Injection:
            """Monkey-patches a module.  Used by scenario.inject.

            Safely overwrites module attributes containing references to
            either the threading module or a synchronization primitve
            with equivalents that use blanket synchronization primitives
            from this scenario.  On close (or exit), safely restores
            the original values.
            """

            def __init__(self, score, module, impersonator):
                self.score = score
                self.module = module
                self.impersonator = impersonator

                # name -> (old, new); used by close() to verify and restore.
                self.replacements = {}
                self.closed = False

                # Pattern 1: names bound directly to threading primitive classes.
                # We iterate (threading_cls, name) pairs and identity-check against
                # each module attribute's value.  Identity (rather than `value in
                # {...}`) is both what we want semantically (a user-defined class
                # called Lock should not be touched) and tolerant of unhashable
                # attribute values like __builtins__.
                threading_classes = [(getattr(threading, n), n)
                                     for n in score.primitive_names]
                for attr_name, value in list(vars(module).items()):
                    for tcls, tname in threading_classes:
                        if value is tcls:
                            new_value = getattr(score.api, tname)
                            self.replacements[attr_name] = (value, new_value)
                            setattr(module, attr_name, new_value)
                            break

                # Pattern 2: a name bound to the threading module itself.
                for attr_name, value in list(vars(module).items()):
                    if attr_name in self.replacements:
                        continue  # already handled above
                    if value is threading:
                        self.replacements[attr_name] = (value, impersonator)
                        setattr(module, attr_name, impersonator)

                if not self.replacements:
                    # Diagnostic: if any primitive-named attribute is itself
                    # a class defined in blanket.primitives, or any attribute
                    # is a ThreadingImpersonator, this module has likely already
                    # been injected by some scenario.  Give a more useful
                    # error than "nothing to patch."
                    already_injected = False
                    for _, tname in threading_classes:
                        candidate = getattr(module, tname, None)
                        if (isinstance(candidate, type)
                                and getattr(candidate, '__module__', None)
                                    == 'blanket.primitives'):
                            already_injected = True
                            break
                    if not already_injected:
                        for value in vars(module).values():
                            if isinstance(value, Scenario.ThreadingImpersonator):
                                already_injected = True
                                break
                    if already_injected:
                        raise ValueError(
                            f"inject: {module.__name__!r} appears to already "
                            f"have a blanket inject active (threading-primitive "
                            f"references are already replaced); close that "
                            f"injection before starting a new one")
                    raise ValueError(
                        f"inject: no threading primitives or threading-module "
                        f"references found in {module.__name__!r}; nothing to patch")

            def __repr__(self):
                status = "closed" if self.closed else f"{len(self.replacements)} replacements"
                return f"<Injection {self.module.__name__!r} {status}>"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()
                return False

            def close(self):
                """Restore the pre-injection values.

                Verifies first that every value inject installed is
                still there, raises RuntimeError otherwise.
                """
                if self.closed:
                    return
                self.closed = True

                for attr_name, (old, new) in self.replacements.items():
                    current = getattr(self.module, attr_name, None)
                    if current is not new:
                        raise RuntimeError(
                            f"{self.module.__name__}.{attr_name} "
                            f"no longer matches the installed value, can't restore")

                for attr_name, (old, new) in self.replacements.items():
                    setattr(self.module, attr_name, old)


        ###############################################################
        ###############################################################
        ##
        ##
        ##       ___ ___  _ __ ___
        ##      / __/ _ \| '__/ _ \
        ##     | (_| (_) | | |  __/
        ##      \___\___/|_|  \___|
        ##
        ##
        ##
        ###############################################################
        ###############################################################

        @base()
        @BoundInnerClass
        class Core:
            """Base class for primitive cores."""

            def __init__(self, score, primitive, api_cls, raw, methods):
                self.lock = score.lock
                self.score = score
                self.primitive = primitive
                self.transactions = {}
                self.transactions_proxy = score.LockedDictProxy(self.transactions)
                self.methods = methods

                cls_name = self.__class__.__name__[:-4]
                self.serial_number = score.serial_number
                score.serial_number += 1
                self.default_name = f"{cls_name} {self.serial_number} {hex(id(self)).upper()}"
                self._name = self.default_name
                self.use_fancy_repr = False

                # Aggregate signals (BoundMethod/Primitive) and Call
                # matching key on normalized forms
                # (raw->cooked, condition->lock), so no per-core alias
                # set is maintained here anymore.

                self.raw = raw
                self.api = api_cls(raw)
                score.cores.add(self)

            @property
            def name(self):
                return self._name

            @name.setter
            def name(self, value):
                if not isinstance(value, str):
                    raise TypeError(f"name must be a string, not {type(value).__name__}")
                self._name = value
                self.use_fancy_repr = True

            def __call__(self, method, regulated, *, entry=None, **kwargs):
                start_time = _current_time()
                cls = self.methods[method]
                kwargs['regulated'] = regulated
                tx = cls(method, start_time, **kwargs)
                # entry_primitive: the object the user actually invoked
                # the method on.  Normally that's this core's own
                # primitive, but a Condition's acquire/release/locked
                # delegate to the underlying lock's core while passing
                # entry=condition, so use_primitives can credit the
                # condition (cond -> ul containment).
                tx.entry_primitive = self.primitive if entry is None else entry
                return tx()

            def unblock(self, method, threads, *, pause=False):
                """Validate and unblock txs for the given threads.
                Returns the list of unblocked threads.  Score lock must
                be held by the caller.

                If pause=True, set the user pause flag (one user incref
                of flag + counter per tx) and unpark; the tx will park
                at PAUSED after commit, where the user can grab control
                again.  Returns immediately without waiting.

                If pause=False (the default), unpark and then block
                until each tx settles -- reached terminal, or re-parked
                in any state listed in its tx-class parking_states.
                For Transaction that's {BLOCKED, PAUSED}; subclasses
                widen the set (TimeoutTransaction adds COMMIT, etc.)
                so an unblock of an OS-blocking commit returns as soon
                as the worker hits its blocking call."""
                threads, txs = self.threads_to_txs(threads, caller='unblock')
                for tx in txs:
                    tx.validate(method=method, state=State.BLOCKED, caller='unblock')
                for tx in txs:
                    if pause and not tx.pause:
                        tx.pause = True
                        tx.pausing += 1
                    tx.unblock()
                if not pause:
                    self.settle(txs)
                return threads

            def settle(self, txs):
                """Block until each tx in txs has reached a settled
                state.  Sequential per-tx wait via Transaction.settle;
                each tx settles independently.  Score lock must be
                held by the caller."""
                for tx in txs:
                    tx.settle()

            def unpause(self, method, threads):
                """Validate and unpause txs for the given threads.
                Returns the list of unpaused threads.  Score lock must
                be held by the caller.  Performs the user decref:
                clear pause flag, decrement pausing, unpark if no
                holders remain.

                Always blocks until each tx settles (terminal or
                re-parked at a state in its parking_states).  If
                pausing > 0 after the user decref the tx stays at
                PAUSED -- which is in parking_states for every tx
                class, so the wait returns immediately."""
                threads, txs = self.threads_to_txs(threads, caller='unpause')
                for tx in txs:
                    tx.validate(method=method, state=State.PAUSED, caller='unpause')
                for tx in txs:
                    if tx.pause:
                        tx.unpause()
                self.settle(txs)
                return threads

            def expire(self, method, threads):
                """Expire the BLOCKED tx of each named thread.  Used by
                the API as a sanity check that the user knows what
                method they're targeting.  Score lock must be held by
                the caller.  Returns the tuple of threads acted on."""
                threads, txs = self.threads_to_txs(threads, caller='expire')
                for tx in txs:
                    tx.validate(method=method, state=State.BLOCKED, caller='expire')
                for tx in txs:
                    tx.expire()
                return threads

            def disregard(self, method, threads):
                """Disregard timeout on the BLOCKED tx of each named
                thread.  Score lock must be held by the caller.
                Returns the tuple of threads acted on."""
                threads, txs = self.threads_to_txs(threads, caller='disregard')
                for tx in txs:
                    tx.validate(method=method, state=State.BLOCKED, caller='disregard')
                for tx in txs:
                    tx.disregard()
                return threads

            def revert(self, method, threads):
                """Revert any prior expire/disregard on the BLOCKED tx
                of each named thread, restoring the user's original
                timeout.  Score lock must be held by the caller.
                Returns the tuple of threads acted on."""
                threads, txs = self.threads_to_txs(threads, caller='revert')
                for tx in txs:
                    tx.validate(method=method, state=State.BLOCKED, caller='revert')
                for tx in txs:
                    tx.revert()
                return threads


            def threads_to_txs(self, threads, *, caller=None):
                """Translate user-facing thread handles into top tx cores.

                Called by API methods while score.lock is already held.
                This helper only boxes threads to the current top active tx;
                validation of method/state happens separately.
                """
                try:
                    threads = tuple(threads)
                except TypeError:
                    raise TypeError(
                        f"expected an iterable of threads, got {threads!r}") from None
                if not threads:
                    return (threads, [])

                prefix = f"{caller}: " if caller else ""
                txs = []
                append = txs.append
                score = self.score
                current_thread = threading.current_thread()

                for thread in threads:
                    if not isinstance(thread, threading.Thread):
                        raise TypeError(
                            f"{prefix}expected a thread, got {thread!r}")
                    if thread is current_thread:
                        raise ValueError(
                            f"{prefix}thread {thread.name!r} is the calling thread; "
                            f"cannot wait on self")

                    terminated = Terminated(thread)
                    signaled = score.wait({thread, terminated})
                    if terminated in signaled:
                        raise ValueError(f"{prefix}thread {thread.name!r} has exited")
                    append(score.transactions[thread])

                return (threads, txs)

            def thread_to_tx(self, thread, *, caller=None):
                """Translate one user-facing thread handle into its top tx core."""
                threads, txs = self.threads_to_txs((thread,), caller=caller)
                return txs[0]


            @base()
            @BoundInnerClass
            class Transaction(Signaling):
                """Base class for all transactions.

                Self-reporting (Signaling): tx.signal returns tx.done.
                Goes high once tx.state has reached a terminal_state
                (RETURNED, RAISED, etc.); stays high forever (tx state
                is monotonic).  Both the tx core and tx.api are
                Signaling; either form works as a wait item.
                """

                # States in which the tx is considered "settled" -- the
                # worker thread has handed off and the scheduler can
                # safely take action.  Used by settle().
                # The base set is the two parked states reachable by
                # any tx: BLOCKED (initial park) and PAUSED (user
                # pause).  Subclasses extend this for the additional
                # states they visit -- TimeoutTransaction adds COMMIT
                # (where commit() may OS-block in actual.X with score
                # lock released), WaitingTransaction adds WAITING, etc.
                parking_states = (State.BLOCKED, State.PAUSED)

                def __init__(self, core, method, start_time, regulated):
                    score = core.score
                    thread = threading.current_thread()

                    self.blocker = None

                    # A tx is regulated iff:
                    #   - the caller used the regulated reference
                    #     (vs. the .raw one), AND
                    #   - the scenario is currently active (between
                    #     __enter__ and __exit__).
                    # And if a tx is regulated, it will always block.

                    self.blocking = self.regulated = regulated and score.entered
                    self.core = core
                    self.end_time = None
                    self.in_callback = False
                    self.kwargs = kwargs = {}
                    self.kwargs_proxy = score.LockedDictProxy(kwargs)
                    # method is the exact bound method the call arrived
                    # on -- raw or cooked, condition or lock -- so the
                    # user-facing tx.method shows what they actually
                    # called.  normalized_method is the normalized form
                    # used internally for Call/BoundMethod matching:
                    # raw->cooked, and a Condition's acquire/release/
                    # locked collapsed to the underlying lock's method.
                    # method is never invoked (commit runs core.actual),
                    # so the two can differ harmlessly.
                    self.method = method
                    self.normalized_method = _normalize_method(method)
                    self.pause = False
                    self.pausing = 0
                    self.raised = False
                    self.result = None
                    self.score = score
                    self.start_time = start_time
                    self.state = State.BLOCKED
                    # Per-tx history of (time, state) transitions.
                    # Seeded with (start_time, BLOCKED); appended to by
                    # tx.to on every state change.  Useful for after-
                    # the-fact queries like "did this tx pause?" or
                    # "how long was it at WAITING?", which the
                    # monotonic state attribute alone can't answer
                    # (states are passed through; presence-in-state
                    # information is otherwise lost once the tx moves
                    # on).
                    self.log = [(start_time, State.BLOCKED)]
                    # last_reached_state tracks the highest state for
                    # which we've signaled Reached(self, state).  tx
                    # starts at BLOCKED, so Reached(self, BLOCKED) is
                    # considered already-reached at creation (its
                    # level signal is high from birth; tx.to needs no
                    # bookkeeping for it).  tx.to signals every state
                    # strictly between last_reached_state and the new
                    # state, inclusive of the new state.
                    self.last_reached_state = State.BLOCKED
                    self.state_observers = []
                    self.succeeded = None
                    self.thread = thread
                    self.timed_out = False

                    self.parent = parent = score.transactions.get(thread)
                    if parent:
                        self.core_parent = self.find_parent_core(parent, core)
                        self.method_parent = self.find_parent_method(parent, method)
                        self.depth = 0 if self.method_parent is None else self.method_parent.depth + 1
                        parent.child = self
                    else:
                        self.core_parent = None
                        self.method_parent = None
                        self.depth = 0
                    self.child = None
                    self.nested_signal = None

                    # True while a user callback associated with this tx
                    # is running (currently: barrier.wait action,
                    # condition.wait_for predicate).  Action(tx.api)
                    # samples this flag.
                    self.in_action = False

                    p = core.primitive
                    api = core.api
                    cls = getattr(api.__class__, method.__func__.__name__)
                    self.api = cls(api, self)

                    # Pre-built set of items settle() passes to
                    # score.wait: the tx.api signal (sticky-high after
                    # tx.close) plus a TransactionState per parking
                    # state (level-triggered, high while the tx is in
                    # that state).  Computed once at construction since
                    # parking_states is a class attr and self.api is
                    # immutable; settle() can call score.wait without
                    # rebuilding the set.
                    self.parking_signals = frozenset(
                        {TransactionState(self.api, state)
                         for state in self.parking_states}
                        | {self.api})

                @staticmethod
                def find_parent_core(parent, core):
                    while parent is not None:
                        if parent.core is core:
                            return parent
                        parent = parent.parent
                    return None

                @staticmethod
                def find_parent_method(parent, method):
                    key = (method.__self__, method.__func__)
                    while parent is not None:
                        parent_key = (parent.method.__self__, parent.method.__func__)
                        if parent_key == key:
                            return parent
                        parent = parent.parent
                    return None

                def is_delegate(self, child):
                    """Return True if `child` is a nested tx that
                    represents delegated work this tx wants the Driver
                    to drive as if it were a top-level tx.  Default
                    False: nested children are transient work the
                    Driver passes through (skipping).

                    Override on parent tx classes whose children are
                    operationally significant -- the children visit
                    the park states the user cares about, while the
                    parent stays in COMMIT doing controller-y work.
                    Example: cond.wait_for whose body iterates,
                    spawning a child cond.wait per iteration.

                    Driver consults this at two points:
                      * Walk-up in pursue: walk-up stops if the
                        current tx's parent says it's a delegate
                        (and pursue's target is a tx state, not a
                        driver-terminal).
                      * Nested-fire signal handler: when a child is
                        born mid-pursue and the current tx (parent)
                        says it's a delegate, redirect focus to the
                        child without going to skipping.
                    """
                    return False

                @property
                def done(self):
                    return self.state in State.terminal_states

                def sample(self, scenario):
                    # Signaling protocol: tx is high once it has reached
                    # a terminal state (RETURNED, RAISED, ...).  Sticky
                    # because tx.state is monotonic.
                    return self.state in State.terminal_states

                @property
                def failed(self):
                    if self.succeeded is None:
                        return None
                    return not self.succeeded

                @property
                def timeout(self):
                    return None

                def repr_helper(self):
                    return f"{self.thread.name!r} start_time={self.start_time} result={self.result} blocking={self.blocking}"

                def repr(self, cls_name):
                    name = self.core.name or self.core.default_name
                    return f"<{cls_name} {self.state.name} {self.repr_helper()} for {name}>"

                def call_methods(self):
                    """Return the normalized method(s) this tx is calling
                    for Call/BoundMethod matching.  Normalization
                    collapses raw->cooked and condition->lock to a
                    single normalized method, so this is just the
                    normalized method.  (A set for back-compat with
                    callers that iterate / test membership.)"""
                    return {self.normalized_method}

                def call_signals(self, state):
                    thread = self.thread
                    depth = self.depth
                    return {Call(thread, method, state, depth=depth) for method in self.call_methods()}

                def observe(self, state, callback):
                    if state <= self.state:
                        raise ValueError(f"can't register observer for {state!r}, tx is already at state {self.state!r}")
                    self.state_observers.append((state, callback))
                    self.state_observers.sort(key=lambda t: t[0], reverse=True)

                def to(self, state):
                    if self.state == state:
                        return
                    assert self.state < state, "tx state back transitions aren't allowed"

                    previous_state = self.state
                    self.state = state
                    self.log.append((_current_time(), state))

                    # Wake waiters parked on the new state's
                    # Call(t, m, state, depth) signals (going-high
                    # transition) and the TransactionState(api, state)
                    # signal.  The previous state's Calls go low
                    # silently -- Call.signal walks the chain and now
                    # sees the new state, so anyone querying after
                    # this transition gets the truth; waiters parked
                    # on the OLD state's Call missed their window when
                    # the tx left that state and don't need a wake.
                    if self.score.transactions.get(self.thread) is self:
                        score = self.score
                        score.signals(self.call_signals(state))
                        score.signal(TransactionState(self.api, state))
                        # Signal Reached(self.api, X) for every state X
                        # we've just crossed.  See last_reached_state
                        # docstring at __init__.
                        while self.last_reached_state.index < state.index:
                            next_index = self.last_reached_state.index + 1
                            self.last_reached_state = State.by_index[next_index]
                            score.signal(Reached(self.api, self.last_reached_state))

                    while self.state_observers:
                        s, observer = self.state_observers[-1]
                        if s > state:
                            break
                        self.state_observers.pop()
                        observer()

                def commit(self):
                    raise NotImplementedError

                def park(self, state):
                    """Park the tx on the scheduler's blocker until released.

                    Sets self.blocker BEFORE transitioning to the parked
                    state, so that a scheduler racing with the worker
                    thread can never observe the parked state without
                    seeing a non-None blocker.

                    Caller must hold score.lock.
                    """
                    score = self.score
                    self.blocker = score.blockers[self.thread]
                    self.to(state)
                    with unlock(score.lock):
                        self.blocker.acquire()

                def unpark(self, state):
                    """Release a parked tx and transition it to a new state.

                    Transitions BEFORE releasing, so a second caller racing
                    with the first sees the post-transition state and its
                    state guard catches the double-call.  Caller must hold
                    score.lock.

                    Raises RuntimeError if the tx was never parked (no
                    blocker installed); this catches misuse such as
                    calling unblock() on a freshly constructed tx that
                    hasn't yet entered its primitive's open()/park()
                    sequence.
                    """
                    blocker = self.blocker
                    if blocker is None:
                        raise RuntimeError(
                            f"can't unpark, tx has no blocker (not parked)")
                    self.blocker = None
                    blocker.release()
                    self.to(state)

                def settle(self):
                    """Block until this tx has reached a settled
                    state: terminal (close ran, tx.api signal goes
                    sticky-high) or re-parked at one of this tx-class's
                    parking_states (TransactionState signal is
                    level-triggered).  Score lock must be held by the
                    caller; it's held on return.  If already settled
                    when called, the level-triggered signal returns
                    immediately."""
                    self.score.wait(self.parking_signals)

                # Timeout operations.  Only TimeoutTransaction (calls
                # that can block waiting on a timeout, e.g. lock.acquire,
                # cond.wait) supports these; it overrides all three with
                # real implementations.  On a plain Transaction (e.g.
                # lock.release) they don't apply, so the base raises.
                # This keeps the single TransactionAPI uniform: the API
                # always delegates to the core, and the core decides
                # whether the operation is meaningful for this call.
                def expire(self):
                    raise NotImplementedError(
                        f"{self.method.__name__} transactions don't support expire()")

                def disregard(self):
                    raise NotImplementedError(
                        f"{self.method.__name__} transactions don't support disregard()")

                def revert(self):
                    raise NotImplementedError(
                        f"{self.method.__name__} transactions don't support revert()")

                def unblock(self):
                    if self.state != State.BLOCKED:
                        raise RuntimeError(f"can't unblock tx in {self.state.name} state")
                    self.unpark(State.COMMIT)
                    self.settle()

                def unstall(self):
                    """Release a STALLED park.  Used on Condition.wait
                    transactions after notify: the transaction has
                    parked mid-commit waiting for the scheduler's
                    permission to run the internal _acquire_restore
                    (which actually re-acquires the Condition's lock
                    and returns from the wait call).

                    The transaction transitions to RESUMED, then runs
                    _acquire_restore and the rest of commit() in the
                    worker thread before reaching COMMITTED.  Waits
                    until the tx settles after the unpark.
                    """
                    if self.state != State.STALLED:
                        raise RuntimeError(f"can't unstall tx in {self.state.name} state")
                    self.unpark(State.RESUMED)
                    self.settle()

                def unpause(self):
                    """User decref: clear pause flag, decrement pausing,
                    maybe unpark.  Called from api-level tx.unpause() and
                    tx.pause = False.  Idempotent and state-safe: no-op if
                    pause flag is already False, raises if tx has advanced
                    past PAUSED.  If pausing reaches zero and state is
                    PAUSED, unpark to EXITING and wait for settle."""
                    if self.state > State.PAUSED:
                        raise RuntimeError("transaction has already advanced past PAUSED state")
                    if not self.pause:
                        return
                    self.pause = False
                    self.pausing -= 1
                    assert self.pausing >= 0, f"pausing went negative: {self.pausing}"
                    if self.state is State.PAUSED and self.pausing == 0:
                        self.unpark(State.EXITING)
                        self.settle()

                def set_pause(self, value):
                    """User pause incref/decref entry point.  True sets the
                    pause flag (and bumps pausing); False clears it (and
                    decrements pausing via unpause()).  Idempotent and
                    state-safe."""
                    value = bool(value)
                    if self.state > State.PAUSED:
                        raise RuntimeError("transaction has already advanced past PAUSED state")
                    if self.pause == value:
                        return
                    if value:
                        self.pause = True
                        self.pausing += 1
                    else:
                        self.unpause()

                def unpausing(self):
                    """Decrement pausing without touching the pause
                    flag.  Internal helper for blanket code that
                    incremented pausing directly and is now releasing
                    its hold (e.g. cycle.pause releasing its Cycle-
                    init contribution at handoff time).  If pausing
                    reaches zero and state is PAUSED, unpark to
                    EXITING."""
                    self.pausing -= 1
                    assert self.pausing >= 0, f"pausing went negative: {self.pausing}"
                    if self.state is State.PAUSED and self.pausing == 0:
                        self.unpark(State.EXITING)

                @property
                def timeout_state(self):
                    """TimeoutState snapshot.  Plain (non-timeout)
                    Transactions return all-None for value/time; the
                    timed_out slot reflects the tx's timed_out flag.
                    TimeoutTransaction subclasses override."""
                    return TimeoutState(None, None, self.timed_out)

                def scenario_exit(self):
                    """Release this tx if it's parked in a blanket-
                    controlled state, so its worker thread can resume
                    and run the rest of the tx natively.  Called by
                    ContextManager.__exit__ AFTER score.entered has
                    gone False (so post-resume method calls produce
                    unregulated txs that don't park).  No-op if the
                    tx isn't currently parked (blocker is None).
                    Caller holds score.lock.

                    BLOCKED, STALLED, PAUSED are the three blanket-
                    parked states.  WAITING and COMMIT aren't blanket-
                    parked: WAITING is OS-parked inside actual.wait /
                    actual.acquire (resolves naturally once the
                    surrounding primitives are unregulated and other
                    threads run normally); COMMIT is a transit state
                    the worker advances through itself.
                    """
                    if self.blocker is None:
                        return
                    state = self.state
                    if state is State.BLOCKED:
                        self.unpark(State.COMMIT)
                    elif state is State.STALLED:
                        self.unpark(State.RESUMED)
                    elif state is State.PAUSED:
                        # Match the cascade's frog-march: clear flag,
                        # zero counter, unpark.  Nothing else will
                        # consult pause/pausing after this, but
                        # leaving them dirty would be untidy.
                        self.pause = False
                        self.pausing = 0
                        self.unpark(State.EXITING)

                def validate(self, method=None, state=None, method_description=None, caller=None):
                    """Validate a tx's method/state.

                    Called with score.lock held.  Returns tx for convenient
                    call-site chaining.
                    """
                    if method is None:
                        method_matches = True
                        methods = ()
                    else:
                        methods = method if isinstance(method, tuple) else (method,)
                        # Match on normalized form: raw/cooked and
                        # condition/lock all collapse via normalization,
                        # so a tx entered as cond.acquire matches a
                        # candidate of lock.acquire (and vice versa).
                        mine = self.normalized_method
                        method_matches = False
                        for candidate in methods:
                            try:
                                cooked = _normalize_method(candidate)
                            except (AttributeError, TypeError):
                                cooked = candidate
                            if self.method == candidate or mine == cooked:
                                method_matches = True
                                break

                    any_state = state is None
                    state_is_state = isinstance(state, State)
                    state_is_tuple = isinstance(state, tuple) and (not state_is_state)
                    state_matches = (
                        any_state
                        or self.state == state
                        or (state_is_tuple and (self.state in state)))

                    if method_matches and state_matches:
                        return

                    prefix = f"{caller}: " if caller else ""
                    if method_description is None:
                        if method is None:
                            method_description = "a transaction"
                        elif len(methods) == 1:
                            method_name = getattr(methods[0], '__name__', repr(methods[0]))
                            cls_name = self.method.__self__.__class__.__name__
                            method_description = f"{method_name} on this {cls_name}"
                        else:
                            method_description = "one of " + ", ".join(
                                getattr(m, '__name__', repr(m)) for m in methods)

                    if any_state:
                        state_str = "any state"
                    elif state_is_tuple:
                        state_names = [s.name for s in state]
                        state_names[-1] = "or " + state_names[-1]
                        state_str = ", ".join(state_names)
                    else:
                        state_str = state.name

                    raise ValueError(
                        f"{prefix}thread {self.thread.name!r} should be calling "
                        f"{method_description} in {state_str} state, but is "
                        f"calling {self.method} in {self.state.name} state")

                def __call__(self):
                    # Publish: register this tx as the active tx on the
                    # thread.  Invisible (raw) transactions don't
                    # appear in score.transactions and fire no minders;
                    # we still register the thread and  (which
                    # is a no-op for invisible txs).
                    score = self.score
                    core = self.core
                    thread = self.thread

                    score.register_thread(thread)

                    if self.regulated:
                        parent = self.parent

                        score.transactions[thread] = self
                        core.transactions[thread] = self
                        score.transaction_apis[thread] = self.api

                        self.signals_thread = parent is None

                        if self.signals_thread:
                            # Wake waiters parked on this thread's
                            # presence-of-tx signal.  Thread(t) is
                            # self-reporting (reads score.transactions).
                            score.signal(score.Thread(thread))

                        # Aggregate method-usage: bump the usage counter
                        # for the normalized method this tx is calling.
                        # incref_usage signals BoundMethod(m) on 0->1.
                        # Stored on the tx so close() decrefs the same.
                        self.cooked_methods = {self.normalized_method}
                        score.incref_usage(*self.cooked_methods)

                        if parent is not None:
                            self.nested_signal = Nested(parent.api)
                            score.signal(self.nested_signal)

                        state_signal = TransactionState(self.api, self.state)

                        call_none_signals = self.call_signals(None)
                        call_state_signals = self.call_signals(self.state)
                        score.signals(call_none_signals | call_state_signals | {state_signal})

                        # use_primitives: the objects this tx uses.  We
                        # derive usage from entry_primitive -- the object
                        # the method was actually invoked on (a
                        # Condition's acquire records entry=condition even
                        # though the tx runs on the lock's core) -- plus,
                        # if that's a Condition, its underlying lock.
                        # Using a condition uses its lock (cond -> ul),
                        # never the reverse and never a sibling.
                        # Primitive(cond) and Primitive(ul) stay distinct
                        # signals; both fire from membership here.
                        caller = _normalize_primitive(self.entry_primitive)
                        use = {self.core.primitive, caller}
                        underlying = getattr(caller._core, 'underlying', None)
                        if underlying is not None:
                            use.add(underlying.primitive)
                        self.use_primitives = use
                        # Per-(thread, primitive) Use is self-reporting;
                        # strobe each cooked form to wake its waiters.
                        for p in self.use_primitives:
                            score.signal(Use(thread, p))
                        # Aggregate primitive-usage via the counter.
                        score.incref_usage(*self.use_primitives)

                    # Now navigate the state machine.
                    if self.blocking and (self.state <= State.BLOCKED):
                        self.park(State.BLOCKED)

                    if self.state <= State.COMMIT:
                        self.to(State.COMMIT)
                        try:
                            self.result = self.commit()
                            self.raised = False
                        except BaseException as e:
                            self.result = e
                            self.raised = True
                        # commit() may have driven state forward (e.g.
                        # cond.wait passes through WAITING / STALLED
                        # via its sunder shims).
                    terminal_state = self.committed()

                    if self.pausing > 0 and (self.state <= State.PAUSED):
                        self.park(State.PAUSED)

                    self.exit(terminal_state)

                    if self.state == State.RAISED:
                        raise self.result
                    return self.result

                def committed(self):
                    """Compute succeeded based on raised/timed_out and
                    transition to COMMITTED.  Returns the terminal state
                    (RAISED if raised, RETURNED otherwise) without
                    advancing to it -- caller is responsible for any
                    further parks (PAUSED) and the final
                    transition + close() via exit().  Caller must hold
                    score.lock."""
                    self.succeeded = (not self.raised) and (not self.timed_out)
                    if self.state < State.COMMITTED:
                        self.to(State.COMMITTED)
                    return State.RAISED if self.raised else State.RETURNED

                def exit(self, terminal_state):
                    """Drive the tx through EXITING into terminal_state
                    and close.  Caller must hold score.lock."""
                    if self.state < State.EXITING:
                        self.to(State.EXITING)
                    self.to(terminal_state)
                    self.close()

                def aborted(self):
                    """Called by the thread monitor when this tx's thread
                    terminates abruptly with the tx still active.  Synthesizes a
                    RAISED termination so observers fire and close() runs
                    naturally; recurses up the parent chain so the entire stack
                    of stuck txs is unwound.  Caller must hold score.lock.
                    """
                    if not self.raised:
                        self.raised = True
                        self.result = RuntimeError(
                            f"thread {self.thread.name!r} terminated with active tx")
                    terminal_state = self.committed()
                    self.exit(terminal_state)
                    parent = self.parent
                    if parent is not None:
                        parent.aborted()

                def close(self):
                    self.end_time = _current_time()

                    if not self.done:
                        raise RuntimeError("transaction not done")

                    score = self.score
                    core = self.core
                    thread = self.thread
                    method = self.method

                    score.log.append(self.api)

                    # tx and tx.api both signal when the tx terminates
                    # and stay signaled forever after.  Cleared by
                    # scenario.reset() (auto-called on context exit).
                    score.signal(self)
                    score.signal(self.api)

                    assert not self.state_observers

                    # Clear parent.child unconditionally.  __init__ sets
                    # parent.child = self for any tx with a parent
                    # (regulated or not); without symmetric clearing,
                    # non-regulated raw calls (raw_lock.acquire inside
                    # barrier.wait, etc.) leak a permanent child
                    # reference on their parent.  Under self-reporting
                    # Nested, that leak becomes a permanently-stuck
                    # Nested signal -- worth fixing properly rather
                    # than papering over.
                    parent_obj = self.parent
                    if parent_obj is not None and parent_obj.child is self:
                        parent_obj.child = None

                    if self.regulated:
                        assert score.transactions.get(thread) is self, \
                            f"close: active tx on {thread.name!r} is not self; active={score.transactions.get(thread)!r}; self={self!r}"

                        # Update score.transactions BEFORE signaling
                        # Not(Thread(t)): Not(Thread(t)).signal reads
                        # `t not in score.transactions`, so the
                        # transaction-removal must precede the signal
                        # so the assertion in score.signal (item.signal
                        # is True) holds.  Likewise the chain change
                        # silently takes Method(m), Use(t, p),
                        # Primitive(p), and Nested(parent) low; those
                        # don't need decrement wakeups.
                        parent = self.parent
                        if parent is None:
                            del score.transactions[thread]
                            del score.transaction_apis[thread]
                        else:
                            score.transactions[thread] = parent
                            score.transaction_apis[thread] = parent.api
                            # parent.child was already cleared
                            # unconditionally above the regulated block.

                            if self.core_parent is None:
                                del core.transactions[thread]
                            else:
                                core.transactions[thread] = self.core_parent

                        # Drop aggregate usage counts.  decref_usage
                        # signals Not(BoundMethod(m)) / Not(Primitive(p))
                        # for any item whose count just reached 0.  We
                        # decref exactly what open increfed (stored on
                        # the tx) so a mid-tx family growth can't
                        # unbalance the counter.
                        score.decref_usage(*self.cooked_methods)
                        score.decref_usage(*self.use_primitives)

                        if self.signals_thread:
                            # Not(Thread(t)) high transition needs a
                            # wakeup; the chain change above already took
                            # Thread(t) low so the assertion holds.
                            score.signal(Not(score.Thread(thread)))

            @base()
            @BoundInnerClass
            class TimeoutTransaction(Transaction):
                # Commits of TimeoutTransaction subclasses typically do
                # actual.X() with score.lock released, where the worker
                # may OS-block indefinitely.  Treat COMMIT as a parking
                # state for settle-wait purposes: the worker handed off,
                # blanket has nothing more to drive until either the OS
                # call returns or the wait is told to give up.
                parking_states = (State.BLOCKED, State.COMMIT, State.PAUSED)

                # Defaults for the cond-based-protocol attributes.
                # WaitingTransaction and StallingTransaction override these.
                wait_on_release_save = False
                stall_on_acquire_restore = False

                # The sentinel value that the underlying threading
                # primitive uses to mean "no timeout."  None for
                # everything except Lock.acquire and RLock.acquire,
                # which use -1 to match threading.Lock.acquire.
                no_timeout = None

                def __init__(self, core, method, start_time, regulated, timeout):
                    super().__init__(method, start_time, regulated)
                    self.kwargs['timeout'] = timeout
                    # original_timeout is the literal user-supplied value
                    # (or the synthetic value the subclass derived from
                    # combining blocking+timeout).  _timeout is the
                    # current effective duration from start_time; the
                    # timeout property reads it and returns the time
                    # remaining against that deadline.
                    self.original_timeout = timeout
                    self._timeout = timeout

                @property
                def timeout(self):
                    """Current remaining time before the timeout fires,
                    or None for no timeout.  _timeout equal to the
                    per-class no_timeout sentinel means "no timeout";
                    otherwise compute live remaining against start_time."""
                    if self._timeout is None or self._timeout == self.no_timeout:
                        return None
                    return max(0, self._timeout - (_current_time() - self.start_time))

                @timeout.setter
                def timeout(self, value):
                    if self.state != State.BLOCKED:
                        raise RuntimeError(
                            f"can't modify timeout, can only be done in BLOCKED state, "
                            f"already in {self.state.name} state")
                    self._timeout = value

                @property
                def timeout_time(self):
                    """Absolute time at which the timeout will fire,
                    or None for no timeout."""
                    if self._timeout is None or self._timeout == self.no_timeout:
                        return None
                    return self.start_time + self._timeout

                @property
                def timeout_state(self):
                    """TimeoutState snapshot: (original_timeout, timeout_time,
                    timed_out)."""
                    return TimeoutState(self.original_timeout, self.timeout_time, self.timed_out)

                def expire(self):
                    """Force the timeout to fire on the next commit.
                    Settings-only: doesn't move the tx out of BLOCKED.
                    Caller drives via assign/relay/unblock/finish/etc."""
                    self.timeout = 0

                def disregard(self):
                    """Drop the timeout: next commit blocks indefinitely.
                    Settings-only; symmetric with expire.  Writes the
                    per-class no_timeout sentinel (None for everything
                    except Lock/RLock.acquire, which use -1)."""
                    self.timeout = self.no_timeout

                def revert(self):
                    """Undo any prior expire/disregard: restore the
                    user's original timeout (against the original
                    start_time deadline)."""
                    self.timeout = self.original_timeout

            @base()
            @BoundInnerClass
            class WaitingTransaction(TimeoutTransaction):
                """TimeoutTransaction that visits the WAITING state via the
                release-save phase of the cond-based blocking protocol.
                Subclasses: ConditionCore.wait, EventCore.wait,
                BarrierCore.wait, SemaphoreCore.acquire."""
                # WAITING is the OS-park state these txs visit via the
                # release-save shim during commit; add it to parking.
                parking_states = (State.BLOCKED, State.COMMIT, State.WAITING, State.PAUSED)
                wait_on_release_save = True

            @base()
            @BoundInnerClass
            class StallingTransaction(WaitingTransaction):
                """WaitingTransaction that also visits the STALLED state via
                the acquire-restore phase.  Currently only ConditionCore.wait."""
                # STALLED is the post-WAITING park before the scheduler
                # permits the acquire-restore phase; add it to parking.
                parking_states = (State.BLOCKED, State.COMMIT, State.WAITING, State.STALLED, State.PAUSED)
                stall_on_acquire_restore = True

            @base()
            @BoundInnerClass
            class API:
                """Scheduler interface to a primitive."""

                def __init__(self, core, raw):
                    score = core.score
                    self._lock = score.lock
                    self._core = core
                    self.raw = raw

                @property
                def name(self):
                    with self._lock:
                        return self._core.name

                @name.setter
                def name(self, value):
                    with self._lock:
                        self._core.name = value

                @property
                def transactions(self):
                    return self._core.transactions_proxy

                def transaction(self, thread):
                    with self._lock:
                        tx = self._core.transactions.get(thread)
                        if tx is None:
                            return None
                        return tx.api

                def unblock(self, method, *threads, pause=False):
                    with self._lock:
                        return self._core.unblock(method, threads, pause=pause)

                def unpause(self, method, *threads):
                    with self._lock:
                        return self._core.unpause(method, threads)

                @base()
                @BoundInnerClass
                @base('UnboundTransactionAPI')
                class TransactionAPI(Signaling):
                    """User-facing tx wrapper.

                    Self-reporting (Signaling): api.signal returns
                    self._core.done.  Mirrors the tx core's Signaling
                    contract -- either form (api or core) can be passed
                    to scenario.wait.
                    """

                    def __init__(self, api, transaction):
                        self._lock = api._core.lock
                        self._core = transaction


                    @property
                    def thread(self):
                        with self._lock:
                            return self._core.thread

                    @property
                    def method(self):
                        with self._lock:
                            return self._core.method

                    @property
                    def done(self):
                        with self._lock:
                            return self._core.done

                    def sample(self, scenario):
                        # tx.state is monotonic and the done-transition
                        # is one-way; lock-free read.
                        return self._core.done

                    @property
                    def start_time(self):
                        with self._lock:
                            return self._core.start_time

                    @property
                    def state(self):
                        with self._lock:
                            return self._core.state

                    @property
                    def kwargs(self):
                        with self._lock:
                            return self._core.kwargs_proxy

                    @property
                    def pause(self):
                        with self._lock:
                            return self._core.pause

                    @pause.setter
                    def pause(self, value):
                        with self._lock:
                            self._core.set_pause(value)

                    @property
                    def pausing(self):
                        """Read-only view: is anything currently holding
                        this tx at PAUSED?  True iff the internal counter
                        is non-zero."""
                        with self._lock:
                            return bool(self._core.pausing)

                    @property
                    def log(self):
                        """The tx's state-transition history as a tuple
                        of (time, state) entries.  Seeded with
                        (start_time, START) at creation; every state
                        transition appends an entry.  Useful for
                        retrospective queries like "did this tx visit
                        PAUSED?" or "how long was it at WAITING?",
                        which the monotonic `state` attribute alone
                        can't answer once the tx has moved past."""
                        with self._lock:
                            return tuple(self._core.log)

                    @property
                    def parent(self):
                        with self._lock:
                            parent = self._core.parent
                            return None if parent is None else parent.api

                    @property
                    def depth(self):
                        with self._lock:
                            return self._core.depth

                    @property
                    def end_time(self):
                        with self._lock:
                            return self._core.end_time

                    @property
                    def result(self):
                        with self._lock:
                            return self._core.result

                    @property
                    def succeeded(self):
                        """True if the tx ran to a successful terminal
                        state (RETURNED); False if it terminated by
                        raising or timing out; None while the tx
                        hasn't yet reached a terminal state."""
                        with self._lock:
                            return self._core.succeeded

                    @property
                    def failed(self):
                        """True if the tx terminated unsuccessfully
                        (raised or timed out); False if it succeeded;
                        None while the tx hasn't yet reached a terminal
                        state."""
                        with self._lock:
                            return self._core.failed

                    @property
                    def timeout(self):
                        with self._lock:
                            return self._core.timeout_state

                    def wait(self, state=None):
                        with self._lock:
                            return self._core.wait(state)

                    def unblock(self):
                        with self._lock:
                            return self._core.unblock()

                    def unpause(self):
                        with self._lock:
                            return self._core.unpause()

                    def unstall(self):
                        with self._lock:
                            return self._core.unstall()

                    # Timeout operations.  Delegated straight to the
                    # core: TimeoutTransaction cores implement them;
                    # plain Transaction cores raise NotImplementedError
                    # naming the call.  One uniform Transaction API.
                    def expire(self):
                        with self._lock:
                            return self._core.expire()

                    def disregard(self):
                        with self._lock:
                            return self._core.disregard()

                    def revert(self):
                        with self._lock:
                            return self._core.revert()


        ###############################################################
        ###############################################################
        ##
        ##
        ##    _            _
        ##   | | ___   ___| | __
        ##   | |/ _ \ / __| |/ /
        ##   | | (_) | (__|   <
        ##   |_|\___/ \___|_|\_\
        ##
        ##
        ##
        ###############################################################
        ###############################################################

        @base()
        @BoundInnerClass
        class LockBaseCore(Core):
            """Base class for Lock and RLock primitive cores."""

            def __init__(self, score, primitive, api_cls, raw_cls, actual):
                self.score = score
                self.lock = score.lock
                self.actual = actual
                self.family = None

                p = primitive
                raw = raw_cls(self)

                scenario = score.api
                method_names = ('acquire', 'release', 'locked')
                if isinstance(p, scenario.RLock) and (not _rlock_provides_locked): method_names = method_names[:-1]

                methods = {}
                for method_name in method_names:
                    impl = getattr(self, method_name)
                    methods[getattr(p, method_name)] = impl
                    methods[getattr(raw, method_name)] = impl

                super().__init__(primitive, api_cls, raw, methods)

            @BoundInnerClass
            class acquire(base.TimeoutTransaction):
                no_timeout = -1

                def __init__(self, core, method, start_time, regulated, blocking=True, timeout=-1):
                    if (not blocking) and (timeout != -1):
                        raise ValueError("can't specify a timeout for a non-blocking call")
                    super().__init__(method, start_time, regulated, timeout)
                    self.kwargs['blocking'] = blocking

                def __repr__(self):
                    return self.repr("Lock.acquire")

                def commit(self):
                    core = self.core
                    actual = core.actual
                    blocking = self.kwargs['blocking']
                    timeout = self.timeout

                    if not blocking:
                        with unlock(self.score.lock):
                            acquired = actual.acquire(False)
                    elif timeout is None:
                        with unlock(self.score.lock):
                            acquired = actual.acquire(True)
                    else:
                        with unlock(self.score.lock):
                            acquired = actual.acquire(True, timeout)

                    # If acquire returned False for *any reason*, it's a timeout.
                    # (I put it to you: blocked=False is exactly equivalent to (and
                    # therefore redundant with) timeout=0.)
                    if not acquired:
                        self.timed_out = True

                    return acquired


            @BoundInnerClass
            class release(base.Transaction):
                def __repr__(self):
                    return self.repr("Lock.release")

                def commit(self):
                    with unlock(self.score.lock):
                        self.core.actual.release()
                    return None

            @BoundInnerClass
            class locked(base.Transaction):
                def __repr__(self):
                    return self.repr("Lock.locked")

                def commit(self):
                    return self.core.actual.locked()

            # _is_owned, _release_save, and _acquire_restore are
            # undocumented interfaces used by threading.Condition.
            # blanket treats them as shims, not visible transactions.
            # The shims implement the actual functionality, and may
            # call back in to the current tx to advance its state.

            class LockPrivateMethod:
                """A private lock method shim."""

                state_name = None

                def __init__(self, core, *args):
                    score = core.score

                    self.lock = lock = core.score.lock
                    self.actual = core.actual
                    self.args = args

                    with lock:
                        tx = score.transactions.get(threading.current_thread())
                        if tx and tx.in_callback:
                            tx = None
                    self.tx = tx

                def __call__(self):
                    raise NotImplementedError

            class _release_save(LockPrivateMethod):
                def __call__(self):
                    state = self.release_save()
                    tx = self.tx
                    if tx and tx.wait_on_release_save:
                        with self.lock:
                            tx.to(State.WAITING)
                    return state

            class _acquire_restore(LockPrivateMethod):
                def __call__(self):
                    tx = self.tx
                    if tx and tx.stall_on_acquire_restore:
                        with self.lock:
                            tx.park(State.STALLED)
                    return self.acquire_restore(*self.args)

            class _is_owned(LockPrivateMethod):
                pass

            def assign(self, thread, acquirer, pause):
                """Orchestration body for LockBaseAPI.assign.  Must be called
                with score.lock held.  Arg shape matches the API signature:
                if `acquirer` is None, `thread` is the lone acquirer (no
                releaser); otherwise `thread` is the releaser and
                `acquirer` is the acquirer.
                """
                if acquirer is not None:
                    releaser = thread
                else:
                    acquirer = thread
                    releaser = None
                score = self.score

                if releaser:
                    if not self.actual_held():
                        raise RuntimeError(f"expected {self.api} to be locked, not unlocked")
                else:
                    if self.actual_held():
                        raise RuntimeError(f"expected {self.api} to be unlocked, not locked")

                dispatch = score.Dispatch()

                releaser_driver = None
                if releaser:
                    releaser_driver = score.Driver(releaser)
                    dispatch.add(releaser_driver)
                acquirer_driver = score.Driver(acquirer)
                dispatch.add(acquirer_driver)

                # Phase 1: validate each driver as it yields ACTIVE.
                # Both are in dispatch from the start; yield order
                # doesn't matter -- each is validated against its
                # expected role, and either failing raises before any
                # tx makes progress.
                for d in dispatch:
                    if d is releaser_driver:
                        thread = releaser
                        method = self.primitive.release
                        role = 'release'
                    else:
                        assert d is acquirer_driver
                        thread = acquirer
                        method = self.primitive.acquire
                        role = 'acquire'
                    assert d.state is d.active, f"expected {role} driver to be in active state, not {d.state!r}"
                    if d.tx.method != method:
                        raise RuntimeError(f"expected {thread.name!r} to call {method}, not {d.tx.method}")
                    if d.tx.state is not State.BLOCKED:
                        raise RuntimeError(
                            f"expected {thread.name!r} {role} tx to be BLOCKED, "
                            f"not {d.tx.state.name}")

                # Phase 2: drive the releaser (if any) to finished.
                if releaser_driver:
                    releaser_driver.finish()
                    releaser_driver()
                    assert releaser_driver.state is d.finished, f"expected releaser driver to be finished, not {releaser_driver.state!r}"
                    # release can raise (lock not held, RLock not
                    # owned by us, etc.).  The worker thread saw the
                    # real exception; we surface a chained RuntimeError
                    # at the scheduler.
                    if releaser_driver.tx.state == State.RAISED:
                        raise RuntimeError(
                            f"assign: thread {releaser.name!r} {self.primitive.release} "
                            f"raised {releaser_driver.tx.result!r}"
                        ) from releaser_driver.tx.result

                # Phase 3: drive the acquirer to its final state.
                if pause:
                    acquirer_driver.pause()
                    final_state = acquirer_driver.parked
                else:
                    acquirer_driver.finish()
                    final_state = acquirer_driver.finished
                acquirer_driver()
                assert acquirer_driver.state is final_state, f"expected acquirer driver to be {final_state.name}, not {acquirer_driver.state!r}"
                # acquire returning False means the actual.X call
                # timed out (either originally or via tx.expire()).
                # Detect and raise at the scheduler.
                if (acquirer_driver.tx.method == self.primitive.acquire
                        and acquirer_driver.tx.state == State.RETURNED
                        and acquirer_driver.tx.result is False):
                    raise RuntimeError(
                        f"assign: thread {acquirer.name!r} {self.primitive.acquire} timed out")

                return [acquirer]


            def relay(self, initial, acquirers, pause):
                """Generator for relay.

                Builds a Driver per participant thread (initial +
                acquirers), validates inputs, and returns a generator
                that yields each acquirer thread as it successfully
                takes the lock.

                `initial` is the lead and may play either role:

                  - release/BLOCKED: hot start.  The lock is currently
                    held by initial's thread; initial is driven to
                    terminal so the first entry of `acquirers` can
                    take the lock.

                  - acquire/BLOCKED: cold start.  The lock is presumed
                    unheld.  `initial` becomes the first acquirer; it
                    takes the lock, then each `acquirers` entry takes
                    it in order.

                Per non-cold-start iter: validate releaser at release/
                BLOCKED, drive it to terminal, validate acquirer at
                acquire/BLOCKED, drive it through PAUSED to its final
                state (PAUSED if pause, terminal otherwise), and yield
                its thread.  The just-driven acquirer becomes the next
                iteration's releaser.

                For iter > 0, the releaser is the prior iter's
                acquirer in `finished` state; dispatch.add reactivates
                it (refreshing cache_tx).  If the worker hasn't yet
                called release(), the Driver lands in `idle` and
                dispatch waits on its thread signal until the new
                release tx is created.

                Caller (API method) holds score.lock through setup.
                The returned generator manages score.lock itself
                across iterations: re-acquiring inside each transfer,
                releasing between yields.
                """
                if not acquirers:
                    raise ValueError("relay requires at least one acquirer")
                threads = [initial] + list(acquirers)
                drivers = [self.score.Driver(t) for t in threads]
                return self._relay_generator(drivers, pause)

            def _relay_generator(self, drivers, pause):
                locked = False
                lock = self.score.lock
                score = self.score
                primitive = self.primitive
                try:
                    if not locked:
                        lock.acquire()
                        locked = True

                    # Drive initial validate its role.  ACTIVE Driver:
                    # drain_recent yields it immediately, no drive.
                    initial = drivers[0]
                    initial()
                    initial.tx.validate(
                        method=(primitive.release, primitive.acquire),
                        state=State.BLOCKED,
                        caller='relay')

                    if initial.tx.method == primitive.acquire:
                        # initial wants to acquire and release.
                        releaser = None
                        chain = drivers
                        drive_and_validate_acquirer = False
                        drive_and_validate_releaser = True
                    else:
                        # initial only wants to release.
                        releaser = initial
                        chain = list(drivers)
                        chain.pop(0)
                        drive_and_validate_acquirer = True
                        drive_and_validate_releaser = False

                    for acquirer in chain:
                        if not locked:
                            lock.acquire()
                            locked = True

                        if releaser is not None:
                            if drive_and_validate_releaser:
                                releaser()
                                releaser.tx.validate(
                                    method=primitive.release,
                                    state=State.BLOCKED,
                                    caller='relay')
                            else:
                                drive_and_validate_releaser = True
                            releaser.finish()
                            releaser()
                            # release can raise (lock not held by
                            # caller, etc.).  The worker thread saw
                            # the real exception; surface a chained
                            # RuntimeError at the scheduler.
                            if releaser.tx.state == State.RAISED:
                                raise RuntimeError(
                                    f"relay: thread {releaser.thread.name!r} {primitive.release} "
                                    f"raised {releaser.tx.result!r}"
                                ) from releaser.tx.result

                        if drive_and_validate_acquirer:
                            acquirer()
                            acquirer.tx.validate(
                                method=primitive.acquire,
                                state=State.BLOCKED,
                                caller='relay')
                        else:
                            drive_and_validate_acquirer = True

                        # Drive acquirer to PAUSED.  Disregard any
                        # timeout on the acquire tx so a pending
                        # timeout doesn't preempt the relay.
                        # d.pausing() increments tx.pausing (scheduler-
                        # side, no flag) and unblocks atomically inside
                        # pursue, so the acquirer parks at PAUSED when
                        # actual.acquire returns.
                        acquirer.tx.disregard()
                        acquirer.pausing()
                        acquirer()

                        # Drive past PAUSED to the final state.
                        if pause:
                            acquirer.pause()
                        else:
                            acquirer.finish()
                        acquirer()
                        # Relay disregard()s the acquire's timeout above,
                        # so commit always blocks until it actually
                        # acquires.  acquire returning False would mean
                        # the disregard didn't take effect -- a blanket
                        # invariant violation, not a user error.
                        assert not (acquirer.tx.method == primitive.acquire
                                    and acquirer.tx.result is False), (
                            f"relay: thread {acquirer.thread.name!r} "
                            f"{primitive.acquire} returned False after disregard")

                        if locked:
                            lock.release()
                            locked = False
                        yield acquirer.thread

                        # This iter's acquirer becomes next iter's
                        # releaser.  Same Driver, same thread; the
                        # next loop top reactivates it onto the new
                        # release tx.
                        releaser = acquirer
                finally:
                    # Driver close needs score.lock; acquire if not
                    # currently held.  Happy path: every Driver is
                    # parked/finished (terminal) and close() already
                    # ran from Driver.to(), so the loop below is a
                    # no-op.  Exception or early generator close
                    # path: non-terminal Drivers get their slots
                    # released so subsequent relay / cycle / skip /
                    # park can use the threads.
                    if not locked:
                        lock.acquire()
                    try:
                        for d in drivers:
                            if not d.done:
                                d.close()
                    finally:
                        lock.release()


            @base()
            @BoundInnerClass
            class LockBaseAPI(base.API):

                @BoundInnerClass
                class acquire(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Lock.acquire")

                @BoundInnerClass
                class release(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Lock.release")

                @BoundInnerClass
                class locked(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Lock.locked")

                def assign(self, thread, acquirer=None, *, pause=False):
                    with self._lock:
                        return self._core.assign(thread, acquirer, pause)

                def relay(self, initial, *acquirers, pause=False):
                    """Hand off the lock through a chain of threads.

                    Setup happens with score.lock held: a Driver is built
                    for each participant thread (which claims the score's
                    one-Driver-per-thread slot).  Validation that each is
                    at the expected method/state happens inside the
                    returned iterator's first per-step dispatch; score.lock
                    is released before this method returns.

                    The returned iterator yields each acquirer thread
                    as it successfully takes the lock.  The underlying
                    generator manages score.lock itself across steps --
                    re-acquiring inside each transfer, releasing between
                    yields -- so the caller can run code between hops
                    without holding the score's lock.

                    initial:   the lead thread.  May be either:
                               - the current lock holder, parked at
                                 release/BLOCKED (hot start: drive it
                                 past release, then `acquirers` take
                                 the lock in order); or
                               - an acquirer parked at acquire/BLOCKED
                                 (cold start: the lock is presumed
                                 unheld; `initial` takes it, then
                                 `acquirers` take it in order).
                    acquirers: the threads, in order, to receive the
                               lock after `initial`.
                    pause:     park each acquirer at PAUSED after it
                               takes the lock, instead of running to
                               completion.
                    """
                    with self._lock:
                        return self._core.relay(initial, acquirers, pause)

                def expire(self, method, *threads):
                    with self._lock:
                        return self._core.expire(method, threads)

                def disregard(self, method, *threads):
                    with self._lock:
                        return self._core.disregard(method, threads)

                def revert(self, method, *threads):
                    with self._lock:
                        return self._core.revert(method, threads)

        @BoundInnerClass
        class LockCore(LockBaseCore):
            def __init__(self, score, primitive):
                super().__init__(primitive, self.LockAPI, score.api.RawLock, threading.Lock())

            def actual_held(self):
                """True iff the underlying threading.Lock is currently
                held (by any thread).  Used by assign() to validate
                lock state."""
                return self.actual.locked()

            class _is_owned(base.LockBaseCore._is_owned):
                def __call__(self):
                    # Fallback: try a non-blocking acquire.  If it
                    # succeeded, we didn't own it; release and return
                    # False.  If it failed, someone (we) owned it; return
                    # True.  Matches threading.Condition's _is_owned
                    # fallback exactly.
                    got = self.actual.acquire(False)
                    if got:
                        self.actual.release()
                    return not got

            class _release_save(base.LockBaseCore._release_save):
                def release_save(self):
                    return self.actual.release()

            class _acquire_restore(base.LockBaseCore._acquire_restore):
                def acquire_restore(self, state):
                    return self.actual.acquire()

            def repr_helper(self, interjection=''):
                state = "locked" if self.actual.locked() else "unlocked"
                if interjection:
                    interjection = interjection + " "
                return f"{state} {interjection}"

            def fancy_repr(self, cls_name):
                name = f"{self.name} " if self.name else ""
                helper = self.repr_helper(f"{cls_name} object")
                return f"<{name}{helper}at {hex(id(self)).upper()}>"

            def compatibility_repr(self):
                return f"<{'locked' if self.actual.locked() else 'unlocked'} _thread.lock object at {hex(id(self.primitive)).upper()}>"

            def __repr__(self):
                return self.fancy_repr('LockCore')

            @BoundInnerClass
            @base()
            class LockAPI(base.LockBaseAPI):
                def __repr__(self):
                    return self._core.fancy_repr('LockAPI')


        @BoundInnerClass
        class RLockCore(LockBaseCore):
            def __init__(self, score, primitive):
                super().__init__(primitive, self.RLockAPI, score.api.RawRLock, threading.RLock())

            def actual_held(self):
                """True iff the underlying threading.RLock is currently
                held.  threading.RLock didn't gain a .locked() method
                until CPython 3.13, so we probe via ._is_owned().

                CAVEAT: ._is_owned() is thread-relative -- it reports
                whether the *calling* thread owns the lock, not whether
                any thread owns it.  assign() is called from the user's
                scheduler thread (not the worker that actually holds
                the RLock during a release-handoff), so this returns
                False even when a worker holds the RLock.  Adequate
                for catching the "unheld but we wanted held / held but
                we wanted unheld" case where scheduler IS the
                (non-)holder; not adequate for the worker-holds-it
                case.  If that becomes a problem, switch to
                self.actual_recursion_count() > 0 (parses repr; works
                cross-thread)."""
                return self.actual._is_owned()

            class _is_owned(base.LockBaseCore._is_owned):
                def __call__(self):
                    return self.actual._is_owned()

            class _release_save(base.LockBaseCore._release_save):
                def release_save(self):
                    return self.actual._release_save()

            class _acquire_restore(base.LockBaseCore._acquire_restore):
                def acquire_restore(self, state):
                    return self.actual._acquire_restore(state)

            def parse_int_from_repr(self, name):
                prefix = f"{name}="
                actual_repr = repr(self.actual)
                fields = actual_repr.split()
                for field in fields:
                    if field.startswith(prefix):
                        return int(field.partition('=')[2])
                raise RuntimeError(f"couldn't parse {name} from {actual_repr!r}")

            def actual_owner(self):
                return self.parse_int_from_repr('owner')

            def actual_recursion_count(self):
                # RLock._recursion_count() is thread-relative on CPython:
                # from the scheduler thread it reports 0 for an RLock owned
                # by a worker.  The repr exposes the cross-thread count,
                # which is what blanket needs for diagnostics and compatible
                # repr output.
                return self.parse_int_from_repr('count')

            def repr_helper(self, interjection=''):
                state = "locked" if self.actual_recursion_count() else "unlocked"
                if interjection:
                    interjection = interjection + " "
                rlock = f"owner={self.actual_owner()} count={self.actual_recursion_count()} "
                return f"{state} {interjection}{rlock}"

            def fancy_repr(self, cls_name):
                name = f"{self.name} " if self.name else ""
                helper = self.repr_helper(f"{cls_name} object")
                return f"<{name}{helper}at {hex(id(self)).upper()}>"

            def compatibility_repr(self):
                owner = self.actual_owner()
                count = self.actual_recursion_count()
                return f"<{'locked' if count else 'unlocked'} _thread.RLock object owner={owner} count={count} at {hex(id(self.primitive)).upper()}>"

            def __repr__(self):
                return self.fancy_repr('RLockCore')

            @BoundInnerClass
            @base()
            class RLockAPI(base.LockBaseAPI):
                def __repr__(self):
                    return self._core.fancy_repr('RLockAPI')

                @property
                def count(self):
                    """Cross-thread recursion count.  An RLock acquired N
                    times by its owner has count == N; an unlocked RLock
                    has count == 0.  Visible from any thread (unlike
                    threading.RLock._recursion_count(), which is
                    thread-relative)."""
                    with self._lock:
                        return self._core.actual_recursion_count()

        @BoundInnerClass
        class ConditionFamily:
            """Manages shared signaling for Conditions using the same underlying lock."""

            def __init__(self, score, lock):
                self.score = score
                self.lock = lock
                self.conditions = {}
                assert lock.family is None
                lock.family = self

            def add(self, cond_core):
                # The condition->lock relationship lives on
                # cond_core.underlying (set at condition-core init) and
                # is consulted by _normalize_method and the tx's
                # use_primitives computation.  Nothing else needs a
                # cross-alias set anymore, so add() just records the
                # condition as a member of the family.
                self.conditions[cond_core.primitive] = None



        @base()
        @BoundInnerClass
        class ConditionBaseCore(Core):
            """Common base for Condition / Event / Barrier cores.

            Each of these primitives has a transaction that passes
            through WAITING during commit() and STALLED on the way
            out, and a cycle-construction story that drives a managed
            group of waiter Drivers in a controlled order.  Nothing
            specific to share at this layer beyond the inheritance
            point itself.
            """

            def __init__(self, score, primitive, api_cls, raw, methods):
                super().__init__(primitive, api_cls, raw, methods)

            @base()
            @BoundInnerClass
            class CycleBase:
                """Core-side base class for cycle objects.

                A cycle drives a managed group of waiter Drivers
                through their final transitions in a controlled order.
                Subclassed by ConditionCore.cycle, EventCore.cycle, and
                BarrierCore.cycle; user code interacts with the
                CycleAPIBase wrapper, never with a CycleCore directly.

                self.remaining is a dict mapping thread to Driver,
                in spec order (dicts preserve insertion order).  The
                Drivers are in `parked` state (their score-slots
                already released by Driver.close, which runs on
                entering any terminal state including parked).  The
                wake/pause/close methods reactivate each Driver (via
                pursue's auto-reactivate from parked) before driving
                it past PAUSED.

                Subclasses must implement __init__ (which validates
                inputs, fires the wake-causing event, and publishes
                self.remaining) and repr().  The default wake_drivers
                drives each Driver through to PAUSED or terminal,
                sequentially in the order given.
                """

                def __init__(self, core):
                    self.core = core
                    # Set by subclass after the trigger fires: dict
                    # mapping thread to Driver in the caller's
                    # argument order (insertion order is preserved by
                    # dict).  Each Driver is parked at the cycle's
                    # post-trigger park (PAUSED for Event/Barrier,
                    # STALLED for Condition).
                    self.remaining = {}
                    # Default 0; subclass overwrites if extras exist
                    # (Condition with notify_all, Event).  See the
                    # CycleAPIBase.extra_waiters property docstring for
                    # per-primitive semantics.
                    self.extra_waiters = 0
                    self.closed = False

                def wake_drivers(self, drivers, *, pause=False):
                    """Drive each Driver past its PAUSED park,
                    in the order given via a Drivers chain.  Ordered
                    drive (not ready-order) is the user-visible
                    contract: threads reach their final state in
                    spec order.

                    Drives the full named set even if some txs raise
                    along the way: only after every driver has been
                    driven past PAUSED do we re-raise the first
                    tx.result (in spec order) of any tx that ended
                    RAISED.  Bailing mid-drain would leave the cycle
                    in a half-closed limbo with some threads still
                    parked at PAUSED; callers expect the assign-like
                    "if it returned normally, it succeeded" guarantee.

                    First-in-spec-order, not BaseExceptionGroup (not
                    available on supported Python versions); the
                    common case is one exception shared by object
                    identity across multiple raising txs (e.g.
                    BrokenBarrierError), and first-in-spec covers it.

                    Score lock must be held by the caller.

                    If pause=True, each Driver parks at PAUSED
                    instead of running to terminal.

                    Returns the list of Drivers (in spec order),
                    after dropping them from self.remaining and
                    closing the cycle if it's now empty.
                    """
                    score = self.core.score
                    if pause:
                        # Hand-off: each Driver is parked at PAUSED
                        # via Cycle init's d.pausing().  Cycle owns
                        # one pausing increment per driver.  Hand it
                        # off to the user as a flag-set pause:
                        #   * tx.pause = True       -- user takes a pause (flag)
                        #   * tx.pausing += 1       -- user's incref balances
                        #   * tx.unpausing()        -- cycle releases its incref
                        # Net counter unchanged, flag now True.  The
                        # Driver stays in its parked terminal state
                        # (close already ran via to(parked) when it
                        # reached PAUSED during cycle init -- no need
                        # to call it again).
                        for d in drivers:
                            tx = d.tx
                            tx.pause = True
                            tx.pausing += 1
                            tx.unpausing()
                        for d in drivers:
                            del self.remaining[d.thread]
                        if not self.remaining:
                            self.closed = True
                        return drivers

                    # Drive past PAUSED.  d.finish() takes each Driver
                    # through commit (frog-march at PAUSED zeroes
                    # pause/pausing and unparks), via a Chain to
                    # preserve spec order.
                    for d in drivers:
                        d.finish()
                    chain = score.Chain(*drivers)
                    dispatch = score.Dispatch()
                    dispatch.add(chain)

                    first_raised = None
                    for d in dispatch:
                        if d.tx.state == State.RAISED and first_raised is None:
                            first_raised = d.tx.result

                    for d in drivers:
                        del self.remaining[d.thread]
                    if not self.remaining:
                        self.closed = True
                    if first_raised is not None:
                        raise first_raised
                    return drivers

                def resolve_drivers(self, threads):
                    """Resolve a tuple of threads to their Drivers in
                    self.remaining.  Raises if any thread isn't a
                    remaining waiter or appears more than once."""
                    seen = set()
                    drivers = []
                    for thread in threads:
                        if not isinstance(thread, threading.Thread):
                            raise TypeError(f"cycle expected a thread, got {thread!r}")
                        if thread in seen:
                            raise ValueError(f"cycle: thread {thread.name!r} specified more than once")
                        seen.add(thread)
                        d = self.remaining.get(thread)
                        if d is None:
                            raise ValueError(f"cycle: thread {thread.name!r} is not a remaining waiter")
                        drivers.append(d)
                    return drivers

                def wake(self, threads):
                    """Drive past PAUSED in spec order.  With at least
                    one thread named, drives just the named threads and
                    returns a tuple of their threads.  With no threads
                    named, drives the first remaining waiter and returns
                    just that thread (raises ValueError if remaining is
                    empty)."""
                    if self.closed:
                        raise RuntimeError("cycle is closed")
                    if threads:
                        drivers = self.resolve_drivers(threads)
                    else:
                        if not self.remaining:
                            raise ValueError("wake(): no threads given and no remaining waiters")
                        drivers = [next(iter(self.remaining.values()))]
                    woke = self.wake_drivers(drivers, pause=False)
                    if threads:
                        return tuple(d.thread for d in woke)
                    return woke[0].thread

                def pause(self, threads):
                    """Drive past PAUSED (post-trigger park) and set the
                    user pause flag, in spec order.  The threads end up
                    parked at PAUSED again but with tx.pause True; the
                    caller releases each later via tx.api.unpause.  With
                    at least one thread named, pauses just the named
                    threads and returns a tuple of their threads.  With
                    no threads named, pauses the first remaining waiter
                    and returns just that thread (raises ValueError if
                    remaining is empty)."""
                    if self.closed:
                        raise RuntimeError("cycle is closed")
                    if threads:
                        drivers = self.resolve_drivers(threads)
                    else:
                        if not self.remaining:
                            raise ValueError("pause(): no threads given and no remaining waiters")
                        drivers = [next(iter(self.remaining.values()))]
                    paused = self.wake_drivers(drivers, pause=True)
                    if threads:
                        return tuple(d.thread for d in paused)
                    return paused[0].thread

                def next_thread(self):
                    """Wake the first remaining waiter and return its
                    thread, or return None if there's nothing to wake
                    (cycle is empty or closed).  Drives the iterator
                    protocol on CycleAPIBase."""
                    if self.closed or not self.remaining:
                        return None
                    return self.wake(())

                def close(self):
                    """Drive all remaining threads past PAUSED to terminal,
                    in spec order, and mark the cycle closed.  Returns the
                    tuple of threads (empty if already closed)."""
                    if self.closed:
                        return ()
                    drivers = list(self.remaining.values())
                    woke = self.wake_drivers(drivers, pause=False)
                    self.closed = True
                    return tuple(d.thread for d in woke)

                def repr(self):
                    raise NotImplementedError

            @base()
            @BoundInnerClass
            class ConditionBaseAPI(base.API):

                @base()
                @BoundInnerClass
                class CycleAPIBase:
                    """User-facing wrapper around a CycleCore.

                    Constructed via api.cycle(*threads) on a Condition,
                    Event, or Barrier API.  Constructor locks the score,
                    builds the corresponding CycleCore (which validates,
                    admits, drives the wake-causing event, and parks every
                    named thread at PAUSED), and releases the lock as it
                    returns.

                    User-facing surface (all methods take score.lock):
                      wake(*threads)    -- drive the named threads past
                                           PAUSED to terminal.  Returns
                                           the threads woken, as a tuple.
                                           Requires at least one thread.
                      pause(*threads)   -- drive the named threads past
                                           the post-trigger PAUSED park
                                           and set the user pause flag.
                                           Returns the threads paused,
                                           as a tuple.  Requires at
                                           least one thread.
                                           The threads are removed from
                                           remaining; the cycle doesn't
                                           re-wake or otherwise manage them.
                      wait()            -- wake the FIRST remaining thread
                                           (per spec order).  Returns the
                                           thread, or None if the cycle is
                                           empty.
                      iter(*threads)    -- generator yielding each named
                                           thread as it's woken.  With no
                                           threads, drains all remaining in
                                           spec order, yielding each as
                                           woken.
                      close()           -- drive all remaining threads past
                                           PAUSED to terminal and mark the
                                           cycle closed.  Returns the
                                           threads, as a tuple.
                      cycle(*threads)   -- shorthand for wake(*threads)
                                           followed by close().
                      waiters           -- snapshot of remaining threads,
                                           in spec order.
                      closed            -- True once the cycle has been
                                           drained or close()d.
                      extra_waiters     -- threads parked on the underlying
                                           primitive that the cycle was
                                           not asked to manage (e.g. extra
                                           Condition waiters not named in
                                           cycle()).

                    On constructor return, every named thread is parked at
                    PAUSED -- the cycle's post-trigger park.  The user
                    controls the order of post-event release: they can wake
                    or pause threads in any order, and the cycle will drive
                    each one through to its final state in the order asked.

                    Also a context manager: __exit__ closes the cycle on
                    normal exit.  If the with-body raised, cycle cleanup
                    errors are suppressed so the original exception
                    isn't masked.

                    Also an iterator: __iter__ returns self, __next__
                    wakes and yields the next remaining waiter (in spec
                    order), raising StopIteration when remaining is
                    empty or the cycle has been closed.  The standard
                    drain idiom:

                        with api.cycle(*waiters) as c:
                            for t in c:
                                do_post_wake_work(t)

                    Equivalent to ``for t in c.iter(): ...``.  Use the
                    args form ``c.iter(*threads)`` for ordered-subset
                    drains where iteration order is user-specified
                    rather than spec-order.
                    """

                    def __init__(self, api, *threads, **kwargs):
                        # BIC passes the outer API instance as 'api' automatically.
                        # kwargs (e.g. scheduler= for Barrier) pass through to the
                        # underlying core Cycle constructor.
                        self._lock = api._lock
                        with self._lock:
                            self._core = api._core.Cycle(threads, **kwargs)

                    def __repr__(self):
                        with self._lock:
                            return self._core.repr()

                    @property
                    def waiters(self):
                        with self._lock:
                            return tuple(self._core.remaining)

                    @property
                    def closed(self):
                        with self._lock:
                            return self._core.closed

                    @property
                    def extra_waiters(self):
                        """Number of waiters parked on the underlying primitive
                        that this cycle was not asked to manage.

                        Set at cycle construction; not updated as the cycle
                        drains.  Per primitive:

                          Condition with notify(n): always 0 (cycle
                               construction raises if actual waiters !=
                               managed waiters, since a finite notify cannot
                               be cleanly aimed at a subset).
                          Condition with notify_all: actual cond._waiters
                               count minus the cycle's managed count.
                               The cycle drives the notify_all, which wakes
                               all waiters; the extras are no longer parked
                               on the Condition but are now under separate
                               management (e.g. user-pended at PAUSED) and
                               are the caller's responsibility.
                          Event: actual cond._waiters minus managed count.
                               Event.set wakes all waiters; same caveat.
                          Barrier: always 0 (cycle construction requires
                               all parties to be specified, with no extras
                               already waiting on the underlying barrier).
                        """
                        with self._lock:
                            return self._core.extra_waiters

                    def wake(self, *threads):
                        """Drive past PAUSED in spec order.  With at
                        least one thread named, drives the named
                        threads and returns a tuple of their threads.
                        With no threads named, drives the first
                        remaining waiter and returns just its thread
                        (raises ValueError if remaining is empty)."""
                        with self._lock:
                            return self._core.wake(threads)

                    def pause(self, *threads):
                        """Drive past the post-trigger PAUSED park and
                        set the user pause flag, in spec order.  With
                        at least one thread named, pauses the named
                        threads and returns a tuple of their threads.
                        With no threads named, pauses the first
                        remaining waiter and returns just its thread
                        (raises ValueError if remaining is empty)."""
                        with self._lock:
                            return self._core.pause(threads)

                    def iter(self, *threads):
                        if threads:
                            for thread in threads:
                                for item in self.wake(thread):
                                    yield item
                        else:
                            while True:
                                try:
                                    yield self.wake()
                                except ValueError:
                                    break

                    def __iter__(self):
                        # The cycle is a single-pass stateful sequence of
                        # wakings; iterator protocol matches that exactly.
                        # Equivalent to .iter() with no args.  See the
                        # class docstring for the standard with-for idiom.
                        return self

                    def __next__(self):
                        with self._lock:
                            thread = self._core.next_thread()
                        if thread is None:
                            raise StopIteration
                        return thread

                    def close(self):
                        with self._lock:
                            return self._core.close()

                    def __call__(self, *threads):
                        if threads:
                            woke = self.wake(*threads)
                        else:
                            woke = ()
                        closed = self.close()
                        return woke + closed

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        # If the user's body already raised, suppress any error
                        # from cycle cleanup so we don't mask the real failure.
                        # The original exception still surfaces; it just doesn't
                        # get its __context__ replaced with cycle bookkeeping noise.
                        if exc_type is not None:
                            try:
                                self.close()
                            except Exception:
                                pass
                        else:
                            self.close()
                        return False

        @BoundInnerClass
        class ConditionCore(base.ConditionBaseCore):
            """Core for Condition primitive."""

            def __init__(self, score, primitive, lock):
                self.lock = score.lock
                p = primitive
                raw = score.api.RawCondition(self)

                if lock is None:
                    lock = score.api.RLock()
                else:
                    if not isinstance(lock, (score.api.Lock, score.api.RLock)):
                        raise TypeError("Condition only accepts a blanket Lock or RLock from this Scenario")
                    lock_core = getattr(lock, '_core', None)
                    if lock_core is None or lock_core.score is not score:
                        raise TypeError("Condition only accepts a blanket Lock or RLock from this Scenario")

                self.underlying = lock._core
                self.actual = self.RegulatedWaitCondition()

                methods = {
                    p.wait:                 self.wait,
                    raw.wait:       self.wait,

                    p.wait_for:             self.wait_for,
                    raw.wait_for:   self.wait_for,

                    p.notify:               self.notify,
                    raw.notify:     self.notify,

                    p.notify_all:           self.notify_all,
                    raw.notify_all: self.notify_all,
                }

                super().__init__(primitive, self.ConditionAPI, raw, methods)

                self.family = family = self.underlying.family or score.ConditionFamily(self.underlying)
                family.add(self)

                self.wait_methods = (
                    self.primitive.wait,
                    self.raw.wait,
                    )

                self.wait_entry_methods = (
                    self.primitive.wait,
                    self.primitive.wait_for,
                    self.raw.wait,
                    self.raw.wait_for,
                    )

                self.notify_methods = (
                    self.primitive.notify,
                    self.primitive.notify_all,
                    self.raw.notify,
                    self.raw.notify_all,
                    )



            def fancy_repr(self, cls_name):
                lock_repr = repr(self.underlying.primitive)
                name = f"{self.name} " if self.name else ""
                return f"<{name}{cls_name}({lock_repr}, {len(self.actual._waiters)})>"

            def compatibility_repr(self):
                lock_repr = repr(self.underlying.primitive)
                return f"<Condition({lock_repr}, {len(self.actual._waiters)})>"

            def __repr__(self):
                return self.fancy_repr('ConditionCore')

            @BoundInnerClass
            class Cycle(base.CycleBase):
                """Fired Condition cycle.

                Constructing this drives the trigger (notify or
                notify_all).  The user invokes api.cycle(*waiters,
                waker) which routes here via the thin wrapper.  The
                constructed cycle is the iterable the user drains via
                wake / pause / wait / iter / close.

                On constructor return, every named waiter is parked at
                STALLED -- the cond.wait post-notify park point.  The
                waker has been driven all the way to its terminal
                state and does not appear in self.remaining.
                """

                def __init__(self, core, threads):
                    super().__init__()
                    score = core.score
                    caller = 'cycle'

                    # Phase 0: cheap validation.  No resources yet.
                    # Driver construction (Phase 1) catches per-thread
                    # bad inputs: a non-Thread argument fails inside
                    # initialize, and a duplicate thread raises
                    # CompetingDriversError on the second Driver(t)
                    # call.  We only enforce the count constraint here
                    # since it's not deducible from Driver semantics.
                    if len(threads) < 2:
                        raise ValueError(
                            "cycle requires at least one waiter and one waker")

                    # Phase 1: claim Driver slots.  The try/finally
                    # below closes any Driver that didn't reach a
                    # terminal state on its own (Driver.close runs
                    # from to() on every terminal transition; the
                    # finally only catches stragglers from an
                    # exception path).  The score's drivers registry
                    # is a WeakValueDictionary, so in principle losing
                    # references would release the slot -- but
                    # tracebacks captured by `assertRaises` and
                    # similar tooling keep the stack frame's locals
                    # alive, which keeps the Drivers alive, which
                    # keeps the slots claimed.  The finally guarantees
                    # cleanup regardless.
                    dispatch = score.Dispatch()
                    drivers = []
                    try:
                        for t in threads:
                            d = score.Driver(t)
                            drivers.append(d)
                            dispatch.add(d)
                        waiters = drivers[:-1]
                        waker = drivers[-1]

                        ul_acquire = core.underlying.primitive.acquire
                        wait_for_methods = (core.primitive.wait_for,
                                            core.raw.wait_for)

                        # Drain: each yields ACTIVE.  Validate each
                        # against its role.
                        for d in dispatch:
                            assert d.state is d.active
                            if d is waker:
                                d.tx.validate(
                                    method=core.notify_methods + (ul_acquire,),
                                    state=State.BLOCKED,
                                    method_description=(
                                        'cond.notify, cond.notify_all, '
                                        'or UL.acquire'),
                                    caller=caller)
                            else:
                                d.tx.validate(
                                    method=(core.wait_entry_methods
                                            + (ul_acquire,)),
                                    state=(State.BLOCKED, State.COMMIT,
                                           State.WAITING),
                                    method_description=(
                                        'cond.wait, cond.wait_for, '
                                        'or UL.acquire'),
                                    caller=caller)

                        # Phase 2: waker-already-past-acquire is only
                        # safe if every waiter has also already passed
                        # cond.wait into WAITING.  Otherwise driving a
                        # waiter from BLOCKED would require it to
                        # acquire the underlying lock the waker is now
                        # holding -- a deadlock.
                        if waker.tx.method in core.notify_methods:
                            for d in waiters:
                                if not (d.tx.method in core.wait_methods
                                        and d.tx.state == State.WAITING):
                                    raise ValueError(
                                        "cycle: when the waker is already "
                                        "at notify, all waiters must "
                                        "already be in WAITING")

                        # Phase 3: drive each waiter (in spec order)
                        # through to parked@WAITING.  Sequential because
                        # the underlying lock chain serializes them:
                        # each waiter must reach WAITING (which releases
                        # the underlying lock) before the next can
                        # acquire it.  Each step drives a single Driver,
                        # so d() suffices.
                        for d in waiters:
                            if d.tx.method == ul_acquire:
                                # Step 3a: drive UL.acquire to terminal.
                                d.finish()
                                d()
                                assert d.state is d.finished
                                # Step 3b: reactivate; the new tx is
                                # cond.wait or cond.wait_for.
                                d.reactivate()
                                d()
                                assert d.state is d.active
                                d.tx.validate(
                                    method=core.wait_entry_methods,
                                    state=(State.BLOCKED, State.COMMIT,
                                           State.WAITING),
                                    method_description=(
                                        'cond.wait or cond.wait_for'),
                                    caller=caller)

                            if d.tx.method in wait_for_methods:
                                # wait_for at BLOCKED or COMMIT.  In
                                # practice we only land here in the
                                # predicate-true edge case: the
                                # predicate returns truthy on first
                                # check and wait_for terminates without
                                # spawning a child cond.wait.  (For
                                # predicate-false, the caller is
                                # expected to have already driven the
                                # child cond.wait to WAITING, in which
                                # case the active tx for this thread is
                                # the child cond.wait, not wait_for
                                # itself.)  d.finish() drives wait_for
                                # to its natural terminal; we then
                                # surface 'predicate succeeded'.
                                d.finish()
                                d()
                                # d.finish targets d.finished; the only
                                # other exits are d.raised (always paired
                                # with an exception that's already
                                # propagated) and d.terminated (only
                                # reachable from idle).  By the time the
                                # drive returns, the driver is in d.finished.
                                assert d.state is d.finished
                                if d.tx.state == State.RAISED:
                                    raise d.tx.result
                                raise RuntimeError(
                                    f"{caller}: waiter "
                                    f"{d.thread.name!r}'s wait_for "
                                    f"predicate succeeded before it "
                                    f"reached WAITING")

                            # Step 3c: cond.wait at BLOCKED, COMMIT, or
                            # WAITING -- drive to parked@WAITING.  In
                            # nested cond.wait-inside-cond.wait_for, the
                            # active tx is the child cond.wait; its
                            # parent (wait_for) declares it a delegate
                            # via is_delegate, so the park lands on the
                            # child's WAITING rather than the parent's
                            # never-reached WAITING.
                            d.wait()
                            d()
                            assert d.state is d.parked
                            assert d.tx.state is State.WAITING

                        # Phase 4: drive waker past UL.acquire (if it
                        # entered there) into notify[_all]/BLOCKED.
                        if waker.tx.method == ul_acquire:
                            waker.finish()
                            waker()
                            assert waker.state is waker.finished
                            waker.reactivate()
                            waker()
                            assert waker.state is waker.active
                            waker.tx.validate(
                                method=core.notify_methods,
                                state=State.BLOCKED,
                                method_description=(
                                    'cond.notify or cond.notify_all'),
                                caller=caller)

                        # Phase 5: validate waiter count BEFORE notify
                        # (actual._waiters reflects pre-notify state).
                        actual_waiters = len(core.actual._waiters)
                        cycle_waiters = len(waiters)
                        n = waker.tx.n
                        if n is not math.inf:
                            if n != cycle_waiters:
                                raise ValueError(
                                    f"{caller}: notify({n}) requires "
                                    f"exactly {n} cycle waiters, got "
                                    f"{cycle_waiters}")
                            if actual_waiters != cycle_waiters:
                                raise ValueError(
                                    f"{caller}: finite notify would affect "
                                    f"extra waiters ({actual_waiters} actual, "
                                    f"{cycle_waiters} managed)")
                            extra_waiters = 0
                        else:
                            if actual_waiters < cycle_waiters:
                                raise RuntimeError(
                                    f"{caller}: actual Condition waiter count "
                                    f"{actual_waiters} is less than managed "
                                    f"count {cycle_waiters}")
                            extra_waiters = actual_waiters - cycle_waiters

                        # Phase 6: fire notify by driving the waker to
                        # its terminal.
                        waker.finish()
                        waker()
                        assert waker.state is waker.finished
                        if waker.tx.state == State.RAISED:
                            raise waker.tx.result

                        # Phase 7: drive each waiter from parked@WAITING
                        # forward to parked@STALLED.  Reactivate via
                        # pursue's auto-reactivate (from parked); the
                        # cond.wait tx is now transiting WAITING -->
                        # STALLED in the worker (or already at STALLED),
                        # and parking@STALLED's listening set picks up
                        # the STALLED signal whether it fires now or
                        # during this drain.  Multi-driver: use dispatch.
                        for d in waiters:
                            d.stall()
                            dispatch.add(d)
                        for yielded in dispatch:
                            yielded.tx.validate(
                                method=core.wait_methods,
                                state=State.STALLED,
                                method_description='cond.wait',
                                caller=caller)
                            assert yielded.state is yielded.parked

                        # Phase 8: publish.  Drivers in remaining are
                        # parked@STALLED with their score-slots already
                        # released (Driver.close runs on entering any
                        # terminal state, including parked).  The
                        # default wake_drivers reactivates each via
                        # pursue's auto-reactivate and drives past
                        # STALLED to terminal/PAUSED.
                        self.remaining = {w.thread: w for w in waiters}
                        self.extra_waiters = extra_waiters
                    finally:
                        # Close any Driver still holding a slot.  On
                        # the success path every Driver has already
                        # reached a terminal state (parked at STALLED
                        # for the waiters, finished for the waker)
                        # and d.close() ran from inside Driver.to();
                        # this loop is a no-op then.  On the exception
                        # path it closes whichever Drivers were
                        # constructed before the failure -- WeakValue
                        # GC isn't enough because tracebacks held by
                        # `assertRaises` and friends keep stack frames
                        # (and thus the Drivers) alive.
                        for d in drivers:
                            if not d.done:
                                d.close()

                def repr(self):
                    status = 'closed' if self.closed else f'{len(self.remaining)} waiters'
                    return f"<Condition.cycle {status}>"

            @BoundInnerClass
            class notify(base.Transaction):
                def __init__(self, core, method, start_time, regulated, n=1):
                    super().__init__(method, start_time, regulated)
                    self.n = n
                    self.kwargs['n'] = n

                def __repr__(self):
                    return self.repr("Condition.notify")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.notify(self.n)

            @BoundInnerClass
            class notify_all(notify):
                def __init__(self, core, method, start_time, regulated):
                    super().__init__(method, start_time, regulated, n=math.inf)

                def __repr__(self):
                    return self.repr("Condition.notify_all")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.notify_all()

            @BoundInnerClass
            class wait(base.StallingTransaction):
                def __init__(self, core, method, start_time, regulated, timeout=None):
                    super().__init__(method, start_time, regulated, timeout)

                def __repr__(self):
                    return self.repr("Condition.wait")

                def commit(self):
                    timeout = self.timeout
                    with unlock(self.score.lock):
                        result = self.core.actual.wait(timeout)
                    if (result is False) and (timeout is not None):
                        self.timed_out = True
                    return result

            @BoundInnerClass
            class RegulatedWaitCondition(threading.Condition):
                """A threading.Condition subclass whose internal wait() goes through
                blanket's regulated tx machinery.

                threading.Condition.wait_for loops by calling self.wait().  By
                default that bypasses blanket's tx machinery, so the wait_for
                tx's lock-shim transitions are the only ones the scheduler
                sees.  We subclass and override wait so that each iteration's
                inner wait dispatches through the blanket primitive's wait
                method, producing a regulated child wait tx parented to the
                wait_for tx.

                The override has to distinguish two callers:
                  1. threading.Condition.wait_for's loop calls self.wait, where
                     self is this BlanketCondition.  Here we dispatch through
                     the blanket primitive to create a child wait tx.
                  2. The wait tx's own commit() calls self.core.actual.wait,
                     which lands here too.  Here we must fall through to the
                     underlying threading.Condition.wait to actually park.

                We discriminate by looking at the active tx for this thread:
                if it's a wait tx on this core, we're inside that tx's commit
                and must fall through; otherwise we dispatch through blanket.
                """

                @property
                def __class__(self):
                    # Masquerade as threading.Condition so user-facing
                    # introspection sees the real class.
                    return threading.Condition

                def __dir__(self):
                    # masquerade as a real Condition object
                    return [n for n in super().__dir__() if n != '_core']

                def __init__(self, core):
                    super().__init__(core.underlying.primitive)
                    self._core = core

                def wait(self, timeout=None):
                    core = self._core
                    score = core.score
                    thread = threading.current_thread()
                    wait = core.primitive.wait

                    with score.lock:
                        entered = score.entered
                        tx = score.transactions.get(thread)
                        in_wait_commit = (
                            tx is not None
                            and tx.core is core
                            and tx.method == wait
                        )
                    if (not entered) or in_wait_commit:
                        # Falling through to threading.Condition.wait:
                        # - in_wait_commit: this is the wait tx's own commit doing
                        #   the real lock-release/cond-park/lock-reacquire dance.
                        # - not entered: scenario has exited; any new tx would be
                        #   non-regulated and non-blocking, leading to infinite
                        #   recursion through this method.  Just delegate.
                        return threading.Condition.wait(self, timeout)

                    # Otherwise we're being called from inside actual.wait_for
                    # (or some other unregulated entry that lands here while a
                    # different tx is active).  Dispatch through the blanket
                    # primitive to create a regulated child wait tx.
                    return wait(timeout)

            @BoundInnerClass
            class wait_for(base.TimeoutTransaction):
                def __init__(self, core, method, start_time, regulated, predicate, timeout=None):
                    super().__init__(method, start_time, regulated, timeout)
                    self.user_predicate = predicate
                    self.iterations = 0
                    self.kwargs['predicate'] = predicate

                def __repr__(self):
                    return self.repr("Condition.wait_for")

                def repr_helper(self):
                    return f"iterations={self.iterations} {super().repr_helper()}"

                def call_predicate(self):
                    """Wrapper around the user predicate, called by
                    threading.Condition.wait_for once before each
                    inner self.wait() and once after each wakeup.

                    in_callback suppresses private-lock shims caused
                    by user code inside the predicate.
                    """
                    self.in_callback = True
                    try:
                        result = self.user_predicate()
                    finally:
                        self.in_callback = False
                    self.iterations += 1
                    return result

                def commit(self):
                    timeout = self.timeout
                    with unlock(self.score.lock):
                        result = self.core.actual.wait_for(
                            self.call_predicate, timeout)
                    if (not result) and (timeout is not None):
                        self.timed_out = True
                    return result

                def is_delegate(self, child):
                    # Each iteration of actual.wait_for spawns a
                    # child cond.wait tx; that child is where the
                    # cycle-relevant park states (WAITING, STALLED)
                    # actually occur.  Driver should drive the child
                    # as if it were top-level, not skip through it.
                    return (isinstance(child, self.score.WaitingTransaction)
                            and child.core is self.core)

            @BoundInnerClass
            @base()
            class ConditionAPI(base.ConditionBaseAPI):
                def __init__(self, core, raw):
                    # BIC adds 'core' (the outer weakref) automatically, so
                    # the super call just forwards 'raw'.
                    super().__init__(raw)

                def __repr__(self):
                    return self._core.fancy_repr('ConditionAPI')

                @property
                def waiters(self):
                    with self._lock:
                        return len(self._core.actual._waiters)

                @BoundInnerClass
                class notify(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Condition.notify")

                    @property
                    def n(self):
                        return self._core.n

                @BoundInnerClass
                class notify_all(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Condition.notify_all")

                    @property
                    def n(self):
                        return self._core.n

                @BoundInnerClass
                class wait(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Condition.wait")

                @BoundInnerClass
                class wait_for(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Condition.wait_for")

                    @property
                    def iterations(self):
                        return self._core.iterations

                    @property
                    def predicate(self):
                        return self._core.user_predicate

                @BoundInnerClass
                class cycle(base.CycleAPIBase):
                    """Start a Condition wake cycle.

                    The final thread is the waker, which must be blocked
                    either on notify / notify_all, or on the Condition's
                    lock acquire immediately before notify / notify_all.
                    All preceding threads are waiters.  Instantiating
                    returns a cycle object controlling the awakened
                    waiters.

                    Each cycle is one round of notify.  A waiter that
                    is in cond.wait_for surfaces in cycle as its
                    current child cond.wait tx -- the standard library's
                    wait_for loop spawns a fresh child each iteration.
                    If the predicate fails after wakeup, threading
                    spawns another child wait tx (visible via
                    Nested(wait_for_tx)); use a subsequent cycle to
                    drive that next round.
                    """

                def assign(self, thread, acquirer=None, *, pause=False):
                    with self._lock:
                        lock = self._core.underlying
                        lock_api = lock.api
                    return lock_api.assign(thread, acquirer, pause=pause)

                def unstall(self, method, *threads):
                    """Release STALLED parks on the cond.wait
                    transactions of the given threads.  Each named
                    thread must have an active cond.wait transaction
                    parked at STALLED, matching method.  After this
                    call, the transaction re-acquires the Condition's
                    lock and proceeds to COMMITTED.  (For threads in
                    cond.wait_for, the active tx at STALLED is the
                    current child cond.wait tx, not the wait_for tx
                    itself.)"""
                    with self._lock:
                        core = self._core
                        threads, txs = core.threads_to_txs(threads, caller='unstall')
                        for tx in txs:
                            tx.validate(method=method, state=State.STALLED, caller='unstall')
                        for tx in txs:
                            tx.unstall()
                        return tuple(tx.thread for tx in txs)

                def expire(self, method, *threads):
                    with self._lock:
                        return self._core.expire(method, threads)

                def disregard(self, method, *threads):
                    with self._lock:
                        return self._core.disregard(method, threads)

                def revert(self, method, *threads):
                    with self._lock:
                        return self._core.revert(method, threads)


        ###############################################################
        ###############################################################
        ##
        ##
        ##                                   _
        ##    ___  ___ _ __ ___   __ _ _ __ | |__   ___  _ __ ___
        ##   / __|/ _ \ '_ ` _ \ / _` | '_ \| '_ \ / _ \| '__/ _ \
        ##   \__ \  __/ | | | | | (_| | |_) | | | | (_) | | |  __/
        ##   |___/\___|_| |_| |_|\__,_| .__/|_| |_|\___/|_|  \___|
        ##                            |_|
        ##
        ##
        ##
        ##
        ###############################################################
        ###############################################################

        @base()
        @BoundInnerClass
        class SemaphoreCoreBase(Core):
            """Core for Semaphore and BoundedSemaphore primitives."""

            def __init__(self, score, primitive, value, bounded, actual_cls, raw_cls, api_cls):
                self.lock = score.lock
                p = primitive
                raw = raw_cls(self)
                methods = {
                    p.acquire:           self.acquire,
                    raw.acquire:         self.acquire,

                    p.release:           self.release,
                    raw.release:         self.release,
                }

                self.actual = actual_cls(value)

                super().__init__(primitive, api_cls, raw, methods)

                underlying_lock = score.api.Lock()
                raw_underlying_lock = underlying_lock._core.raw
                self.underlying_lock = underlying_lock
                self.underlying_lock_core = underlying_lock._core
                self.actual._cond = threading.Condition(raw_underlying_lock)

            def actual_waiter_count(self):
                return len(self.actual._cond._waiters)

            @property
            def value(self):
                return self.actual._value

            @property
            def available(self):
                return self.value - self.actual_waiter_count()

            def fancy_repr(self, cls_name):
                addr = hex(id(self.primitive)).upper()
                actual_repr = repr(self.actual)

                name = f"{self.name} " if self.name else ""
                after_colon = actual_repr.partition(':')[2]
                return f"<{name}{cls_name} object at {addr}:{after_colon}>"

            def compatibility_repr(self):
                addr = hex(id(self.primitive)).upper()
                actual_repr = repr(self.actual)

                before, at, after = actual_repr.partition(' at ')
                _, colon, after = after.partition(':')

                return f"{before}{at}{addr}:{after}"

            def __repr__(self):
                return self.fancy_repr()

            def allocate(self, threads, pause=False):
                """Driver-based ordered drive of Semaphore acquires
                and releases.

                Returns an iterator that yields each acquire thread
                in turn.  Setup -- Driver construction, role
                classification, and semaphore-math feasibility check
                -- runs SYNCHRONOUSLY (before the generator returns)
                so that an infeasible batch raises RuntimeError
                before any release in the batch executes.  The
                Driver-yielding loop runs lazily as the iterator is
                consumed.

                Pass 1 (sync): drive each Driver to active, classify
                by method, run pre-flight math check.  Imperatives
                are NOT pushed here -- doing so would unblock all
                txs at once and let workers race.

                Pass 2 (lazy, in the generator): a Chain over the
                drivers in spec order is added to a Dispatch.  As
                Chain promotes each driver, its imperative
                (d.pause() or d.finish()) is pushed -- unblocking
                exactly one tx at a time, so commits happen in spec
                order.  Each acquire thread yields when its driver
                reaches terminal.
                """
                lock = self.score.lock
                score = self.score
                acquire_methods = (self.primitive.acquire, self.raw.acquire)
                release_methods = (self.primitive.release, self.raw.release)

                lock.acquire()
                role = {}
                try:
                    # Build a Driver per thread, in spec order.  Driver
                    # construction raises on duplicates and on the
                    # current thread.
                    for thread in threads:
                        d = score.Driver(thread)
                        role[d] = None

                    # Pass 1a: drive each Driver to active (blocking
                    # until the thread has pushed a tx), classify by
                    # method.  Do NOT push imperatives yet -- math
                    # check below must precede any drive that would
                    # commit a release.
                    for d in role:
                        d()
                        if d.state is d.terminated:
                            raise RuntimeError(
                                f"allocate: thread {d.thread.name!r} "
                                f"terminated before pushing a tx")
                        method = d.tx.method
                        if method in acquire_methods:
                            role[d] = 'acquire'
                        elif method in release_methods:
                            role[d] = 'release'
                        else:
                            raise ValueError(
                                f"allocate: thread {d.thread.name!r} should be "
                                f"calling acquire or release on this Semaphore, "
                                f"but is calling {method}")

                    # Pass 1b: every tx must be at BLOCKED.  Allowing
                    # an acquire past BLOCKED (in COMMIT, or already
                    # parked at WAITING) means the worker is racing
                    # other actual.wait waiters outside the batch
                    # for slots, which breaks any ordering guarantee
                    # allocate could otherwise provide.  Allowing a
                    # release past BLOCKED is the same: the release
                    # has already committed (or is mid-commit) so
                    # allocate's pre-flight math is observing stale
                    # state.  Reject either with a clear error.
                    for d, r in role.items():
                        if d.tx.state is not State.BLOCKED:
                            raise RuntimeError(
                                f"allocate: {r} thread {d.thread.name!r} "
                                f"must be at BLOCKED, got {d.tx.state.name}")

                    # Pass 1c: pre-flight semaphore math.  Must run
                    # before any release commits -- callers expect
                    # that if allocate raises, nothing in the batch
                    # has executed yet.  Every acquire must consume
                    # a unit of slack from the running tally.
                    value = self.available
                    for d, r in role.items():
                        tx = d.tx
                        if r == 'release':
                            value += tx.n
                            continue
                        if value <= 0:
                            raise RuntimeError(
                                f"allocate: acquire thread {tx.thread.name!r} "
                                f"cannot be proven to succeed")
                        value -= 1

                    # Pass 1d: math is proven, push imperatives.
                    # Under lazy driver semantics, calling
                    # d.pause()/d.finish() here stages each driver's
                    # tx.unblock on self.lazy but doesn't fire it.
                    # The Chain in pass 2 promotes one driver at a
                    # time and fires only the promoted driver's
                    # lazy, so workers can't race for score.lock.
                    for d, r in role.items():
                        if r == 'acquire' and pause:
                            d.pause()
                        else:
                            d.finish()
                except BaseException:
                    # Setup failed.  Close any non-terminal Drivers
                    # and release the lock before re-raising.
                    for d in role:
                        if not d.done:
                            d.close()
                    lock.release()
                    raise

                # Setup succeeded; return the generator that does the
                # chain-drive + yields.  The generator owns the lock
                # from here.
                return self._allocate_drive(role, lock)

            def _allocate_drive(self, role, lock):
                """Lazy phase of allocate: build a Chain over the
                drivers in spec order, add it to a Dispatch, and
                yield each acquire thread as its Driver reaches
                terminal.

                The Chain promotes drivers one at a time.  Each
                promotion fires the promoted driver's lazy callable
                (its staged tx.unblock from pass 1d), unblocking
                exactly one tx at a time -- commits happen in spec
                order.
                """
                score = self.score
                try:
                    chain = score.Chain(*role.keys())
                    dispatch = score.Dispatch()
                    dispatch.add(chain)

                    for d in dispatch:
                        tx = d.tx
                        if tx is not None and tx.state == State.RAISED:
                            raise tx.result
                        if tx is not None and tx.result is False:
                            status = "timed out" if tx.timed_out else "failed"
                            raise RuntimeError(
                                f"allocate: {role[d]} thread {d.thread.name!r} {status}")
                        if role[d] != 'acquire':
                            continue
                        lock.release()
                        try:
                            yield d.thread
                        finally:
                            lock.acquire()
                finally:
                    for d in role:
                        if not d.done:
                            d.close()
                    lock.release()

            @BoundInnerClass
            class acquire(base.WaitingTransaction):
                def __init__(self, core, method, start_time, regulated, blocking=True, timeout=None):
                    super().__init__(method, start_time, regulated, timeout)
                    self.kwargs['blocking'] = blocking

                def __repr__(self):
                    return self.repr("Semaphore.acquire")

                def commit(self):
                    blocking = self.kwargs['blocking']
                    timeout = self.timeout
                    with unlock(self.score.lock):
                        result = self.core.actual.acquire(blocking=blocking, timeout=timeout)

                    # If acquire returned False for *any reason*, it's a timeout.
                    # (I put it to you: blocked=False is exactly equivalent to (and
                    # therefore redundant with) timeout=0.)
                    if not result:
                        self.timed_out = True
                    return result

            @BoundInnerClass
            class release(base.Transaction):
                def __init__(self, core, method, start_time, regulated, n=1):
                    super().__init__(method, start_time, regulated)
                    self.n = n
                    self.kwargs['n'] = n

                def __repr__(self):
                    return self.repr("Semaphore.release")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.release(self.n)

            @base()
            @BoundInnerClass
            class SemaphoreAPIBase(base.API):
                def __init__(self, core, raw):
                    super().__init__(raw)

                def __repr__(self):
                    return self._core.fancy_repr('SemaphoreAPI')

                @property
                def waiters(self):
                    with self._lock:
                        return self._core.actual_waiter_count()

                @property
                def value(self):
                    with self._lock:
                        return self._core.value

                @property
                def available(self):
                    with self._lock:
                        return self._core.available

                @BoundInnerClass
                class acquire(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Semaphore.acquire")

                @BoundInnerClass
                class release(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Semaphore.release")

                    @property
                    def n(self):
                        return self._core.n

                def allocate(self, *threads, pause=False):
                    """Drive an ordered sequence of Semaphore acquires and releases.

                    The returned iterator yields each acquirer thread
                    as it succeeds.  The underlying generator manages
                    score.lock itself across steps -- holding it for
                    setup and between steps, releasing it around the
                    yield so the caller can run code in their own
                    with-lock context.

                    threads: a sequence mixing acquire-side threads
                             with release-side threads, all of whose
                             txs must be parked at BLOCKED on
                             Semaphore.acquire / Semaphore.release
                             respectively.  Mixing in a tx that's
                             already past BLOCKED (in COMMIT, parked
                             at WAITING, or further along) raises
                             RuntimeError, because such a tx is
                             racing other waiters outside the batch
                             and allocate can't promise an outcome.
                    pause:   park each acquirer at PAUSED after it
                             succeeds (via d.pause() -- both flag and
                             counter set), instead of running to
                             completion.  Caller releases via the
                             matching tx.api.unpause.
                    """
                    if not threads:
                        raise ValueError("allocate requires at least one thread")
                    for thread in threads:
                        if not isinstance(thread, threading.Thread):
                            raise TypeError(
                                f"allocate: expected a thread, got {thread!r}")
                    # No lock here: the core generator acquires and
                    # manages score.lock itself.  Driver construction
                    # inside the generator catches the calling-thread
                    # and duplicate-thread cases.
                    return self._core.allocate(threads, pause=pause)

                def expire(self, method, *threads):
                    with self._lock:
                        return self._core.expire(method, threads)

                def disregard(self, method, *threads):
                    with self._lock:
                        return self._core.disregard(method, threads)

                def revert(self, method, *threads):
                    with self._lock:
                        return self._core.revert(method, threads)

        @BoundInnerClass
        class SemaphoreCore(base.SemaphoreCoreBase):
            """Core for Semaphore and BoundedSemaphore primitives."""

            def __init__(self, score, primitive, value):
                super().__init__(primitive, value,
                    bounded=False,
                    actual_cls=threading.Semaphore,
                    raw_cls=score.api.RawSemaphore,
                    api_cls=self.SemaphoreAPI,
                    )

            @BoundInnerClass
            @base()
            class SemaphoreAPI(base.SemaphoreAPIBase):
                def __repr__(self):
                    return self._core.fancy_repr('BoundedSemaphoreAPI')

        @BoundInnerClass
        class BoundedSemaphoreCore(base.SemaphoreCoreBase):

            def __init__(self, score, primitive, value):
                super().__init__(primitive, value,
                    bounded=True,
                    actual_cls=threading.BoundedSemaphore,
                    raw_cls=score.api.RawBoundedSemaphore,
                    api_cls=self.BoundedSemaphoreAPI,
                    )

            @BoundInnerClass
            @base()
            class BoundedSemaphoreAPI(base.SemaphoreAPIBase):
                def __repr__(self):
                    return self._core.fancy_repr('BoundedSemaphoreAPI')


        ###############################################################
        ###############################################################
        ##
        ##
        ##                         _
        ##     _____   _____ _ __ | |_
        ##    / _ \ \ / / _ \ '_ \| __|
        ##   |  __/\ V /  __/ | | | |_
        ##    \___| \_/ \___|_| |_|\__|
        ##
        ##
        ##
        ##
        ###############################################################
        ###############################################################

        @BoundInnerClass
        class EventCore(base.ConditionBaseCore):
            """Core for Event primitive.

            Thin wrapper around threading.Event.  blanket replaces the
            event's internal Condition with a real Condition built over a
            raw blanket Lock, so the lock's _release_save/_acquire_restore
            shims expose WAITING/STALLED scheduler points for Event.wait.
            """

            def __init__(self, score, primitive):
                self.lock = score.lock

                p = primitive
                raw = score.api.RawEvent(self)
                methods = {
                    p.is_set:          self.is_set,
                    raw.is_set:        self.is_set,

                    p.isSet:           self.isSet,
                    raw.isSet:         self.isSet,

                    p.set:             self.set,
                    raw.set:           self.set,

                    p.clear:           self.clear,
                    raw.clear:         self.clear,

                    p.wait:            self.wait,
                    raw.wait:          self.wait,
                }

                self.actual = threading.Event()
                super().__init__(primitive, self.EventAPI, raw, methods)

                underlying_lock = score.api.Lock()
                raw_underlying_lock = underlying_lock._core.raw
                self.underlying_lock = underlying_lock
                self.underlying_lock_core = underlying_lock._core
                self.actual._cond = threading.Condition(raw_underlying_lock)

            def event_status(self):
                return 'set' if self.actual.is_set() else 'unset'

            def fancy_repr(self, cls_name):
                addr = hex(id(self.primitive)).upper()
                name = f"{self.name} " if self.name else ""
                return f"<{name}{cls_name} object at {addr}: {self.event_status()}>"

            def compatibility_repr(self):
                return f"<threading.Event at {hex(id(self.primitive)).upper()}: {self.event_status()}>"

            def __repr__(self):
                return self.fancy_repr('EventCore')

            def actual_waiter_count(self):
                return len(self.actual._cond._waiters)

            @BoundInnerClass
            class is_set(base.Transaction):
                def __repr__(self):
                    return self.repr("Event.is_set")

                def commit(self):
                    return self.core.actual.is_set()

            @BoundInnerClass
            class isSet(is_set):
                def __repr__(self):
                    return self.repr("Event.isSet")

            @BoundInnerClass
            class set(base.Transaction):
                def __repr__(self):
                    return self.repr("Event.set")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.set()

            @BoundInnerClass
            class clear(base.Transaction):
                def __repr__(self):
                    return self.repr("Event.clear")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.clear()

            @BoundInnerClass
            class wait(base.WaitingTransaction):
                def __init__(self, core, method, start_time, regulated, timeout=None):
                    super().__init__(method, start_time, regulated, timeout)

                def __repr__(self):
                    return self.repr("Event.wait")

                def commit(self):
                    timeout = self.timeout
                    with unlock(self.score.lock):
                        result = self.core.actual.wait(timeout)
                    if (result is False) and (timeout is not None):
                        self.timed_out = True
                    return result

            @BoundInnerClass
            class Cycle(base.CycleBase):
                """Fired Event cycle.

                Constructing this drives Event.set.  The user invokes
                api.cycle(*waiters, setter) which routes here via the
                thin wrapper.

                On constructor return, every named thread (waiters
                AND setter) is parked at PAUSED (the cycle's post-
                trigger park).  The user drives them past PAUSED
                via wake / pause / wait / iter / close on the
                returned cycle object.
                """

                def __init__(self, core, threads):
                    super().__init__()
                    score = core.score
                    caller = 'cycle'

                    threads = list(threads)
                    if len(threads) < 2:
                        raise ValueError("cycle requires at least one waiter and one setter")

                    if core.actual.is_set():
                        raise RuntimeError("cycle: Event is already set")

                    dispatch = score.Dispatch()
                    drivers = []
                    try:
                        for thread in threads:
                            driver = score.Driver(thread)
                            dispatch.add(driver)
                            drivers.append(driver)

                        waiters = drivers.copy()
                        setter = waiters.pop()

                        # Stage 1: iterate dispatch, validate each yielded
                        # driver against its expected role.  Yield-detaches
                        # the drivers.
                        for d in dispatch:
                            if d is setter:
                                method = (core.primitive.set, core.raw.set)
                                method_description = "event.set"
                            else:
                                method = (core.primitive.wait, core.raw.wait)
                                method_description = "event.wait"
                            d.tx.validate(
                                method=method,
                                state=State.BLOCKED,
                                method_description=method_description,
                                caller=caller)

                        for w in waiters:
                            w.wait()
                            dispatch.add(w)

                        for d in dispatch:
                            assert d.state is d.parked, f"expected parked, got {d.state}"
                            assert d.tx.state is State.WAITING, f"expected tx WAITING, got {d.tx.state}"
                        for d in waiters:
                            d.pausing()
                            dispatch.add(d)

                        actual_waiters = core.actual_waiter_count()
                        extra_waiters = actual_waiters - len(waiters)

                        if core.actual.is_set():
                            raise RuntimeError("cycle: Event was set unexpectedly")

                        # Stage 3 first half: setter.pausing() drives
                        # the setter through commit (actual.set),
                        # releasing waiters from actual.wait.  Iterate:
                        # setter yields PARKED at PAUSED, waiters yield
                        # PARKED at PAUSED.  pausing (not pause): the
                        # Cycle owns these incref's on its bookkeeping
                        # and releases them at wake/pause time, no user
                        # pause flag involved here.
                        setter.pausing()
                        dispatch.add(setter)

                        for d in dispatch:
                            assert d.state is d.parked, f"expected parked, got {d.state}"
                            assert d.tx.state is State.PAUSED, f"expected tx PAUSED, got {d.tx.state}"
                            assert d.tx.succeeded, f"tx didn't succeed (state {d.tx.state.name})"

                        # remaining = thread -> Driver mapping in
                        # specified order (waiters first, setter last);
                        # dicts preserve insertion order.
                        self.remaining = {d.thread: d for d in drivers}
                        self.extra_waiters = extra_waiters
                    finally:
                        # Close any Driver still holding a slot.  Happy
                        # path: every Driver is parked@PAUSED (terminal),
                        # close() already ran from Driver.to(), no-op.
                        for d in drivers:
                            if not d.done:
                                d.close()

                def repr(self):
                    status = 'closed' if self.closed else f'{len(self.remaining)} remaining'
                    return f"<Event.cycle {status}>"



            @BoundInnerClass
            @base()
            class EventAPI(base.ConditionBaseAPI):
                def __init__(self, core, raw):
                    super().__init__(raw)

                def __repr__(self):
                    return self._core.fancy_repr('EventAPI')

                @property
                def waiters(self):
                    with self._lock:
                        return len(self._core.actual._cond._waiters)

                @BoundInnerClass
                class is_set(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Event.is_set")

                @BoundInnerClass
                class isSet(is_set):
                    def __repr__(self):
                        return self._core.repr("Event.isSet")

                @BoundInnerClass
                class set(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Event.set")

                @BoundInnerClass
                class clear(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Event.clear")

                @BoundInnerClass
                class wait(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Event.wait")

                @BoundInnerClass
                class cycle(base.CycleAPIBase):
                    """Start an Event wake cycle.

                    The final thread is the setter (blocked on set()).
                    All preceding threads are waiters blocked on wait().
                    Instantiating returns a cycle object controlling
                    the awakened waiters.
                    """

                def expire(self, method, *threads):
                    with self._lock:
                        return self._core.expire(method, threads)

                def disregard(self, method, *threads):
                    with self._lock:
                        return self._core.disregard(method, threads)

                def revert(self, method, *threads):
                    with self._lock:
                        return self._core.revert(method, threads)

        ###############################################################
        ###############################################################
        ##
        ##
        ##    _                     _
        ##   | |__   __ _ _ __ _ __(_) ___ _ __
        ##   | '_ \ / _` | '__| '__| |/ _ \ '__|
        ##   | |_) | (_| | |  | |  | |  __/ |
        ##   |_.__/ \__,_|_|  |_|  |_|\___|_|
        ##
        ##
        ##
        ##
        ###############################################################
        ###############################################################

        @BoundInnerClass
        class BarrierCore(base.ConditionBaseCore):
            """Core for Barrier primitive.

            Thin wrapper around threading.Barrier.  blanket replaces the
            barrier's internal Condition with a real Condition built over a
            raw blanket Lock, so the lock's _release_save/_acquire_restore
            shims can expose WAITING/STALLED scheduler points.
            """

            def __init__(self, score, primitive, parties, action=None, timeout=None):
                if parties < 1:
                    raise ValueError("parties must be >= 1")

                self.lock = score.lock
                self.default_timeout = timeout
                self.user_action = action

                p = primitive
                raw = score.api.RawBarrier(self)
                methods = {
                    p.wait:            self.wait,
                    raw.wait:  self.wait,

                    p.reset:           self.reset,
                    raw.reset: self.reset,

                    p.abort:           self.abort,
                    raw.abort: self.abort,
                }

                self.actual = threading.Barrier(parties, action=self.run_action, timeout=None)
                super().__init__(primitive, self.BarrierAPI, raw, methods)

                # The underlying raw Lock used only by the real Barrier's
                # internal Condition.
                underlying_lock = score.api.Lock()
                underlying_core = underlying_lock._core
                raw_underlying_lock = underlying_lock._core.raw
                self.underlying_lock = underlying_lock
                self.underlying_lock_core = underlying_core
                # Keep the helper primitive registered; raw calls still need
                # their transaction API classes even though users should not
                # normally reach this object.
                self.actual._cond = threading.Condition(raw_underlying_lock)

            def run_action(self):
                """Worker-side entry point for the barrier's action
                callback.  Marks the tx (so observers know action
                ran), signals Action(tx.api) high, runs the action
                with the score lock released and the parent tx's
                API passed in as the sole argument, then unsignals
                Action.

                If barrier.wait was called outside an active scenario
                (no regulated tx exists), the action still runs but
                receives None as its argument; Action is not signaled
                in that case because there is no tx to signal against.

                Action is bracketed by try/finally so an exception in
                the action still unsignals cleanly.
                """
                thread = threading.current_thread()
                tx = None
                with self.score.lock:
                    candidate = self.score.transactions.get(thread)
                    if candidate is not None and candidate.core is self:
                        candidate.ran_action = True
                        if self.user_action is not None:
                            tx = candidate
                            tx.in_action = True
                            self.score.signal(Action(tx.api))
                try:
                    if self.user_action is not None:
                        return self.user_action(tx.api if tx is not None else None)
                    return None
                finally:
                    if tx is not None:
                        with self.score.lock:
                            tx.in_action = False
                            # Action(tx.api).signal reads
                            # tx._core.in_action; setting in_action
                            # False above takes it low.  Signaling
                            # high->low is silent, so no wakeup.

            @property
            def parties(self):
                return self.actual.parties

            @property
            def n_waiting(self):
                return self.actual.n_waiting

            @property
            def broken(self):
                return self.actual.broken

            def status(self):
                if self.broken:
                    return "broken"
                return f"waiters={self.n_waiting}/{self.parties}"

            def fancy_repr(self, cls_name):
                addr = hex(id(self.primitive)).upper()
                name = f"{self.name} " if self.name else ""
                return f"<{name}{cls_name} object at {addr}: {self.status()}>"

            def compatibility_repr(self):
                return f"<threading.Barrier at {hex(id(self.primitive)).upper()}: {self.status()}>"

            def __repr__(self):
                return self.fancy_repr('BarrierCore')

            @BoundInnerClass
            class wait(base.WaitingTransaction):
                def __init__(self, core, method, start_time, regulated, timeout=None):
                    if timeout is None:
                        timeout = core.default_timeout
                    super().__init__(method, start_time, regulated, timeout)
                    self.index = None
                    self.ran_action = False

                def __repr__(self):
                    return self.repr("Barrier.wait")

                def commit(self):
                    timeout = self.timeout
                    with unlock(self.score.lock):
                        index = self.core.actual.wait(timeout)
                    self.index = index
                    return index

            @BoundInnerClass
            class reset(base.Transaction):
                def __repr__(self):
                    return self.repr("Barrier.reset")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.reset()

            @BoundInnerClass
            class abort(base.Transaction):
                def __repr__(self):
                    return self.repr("Barrier.abort")

                def commit(self):
                    with unlock(self.score.lock):
                        return self.core.actual.abort()

            @BoundInnerClass
            class Cycle(base.CycleBase):
                """Fired Barrier cycle.

                Constructing this drives the opener through to fill
                the barrier.  The user invokes
                api.cycle(*non_final_waiters, opener) which routes
                here via the thin wrapper.

                On constructor return, every named thread is parked
                at PAUSED (the cycle's post-trigger park).  The user
                drives them past PAUSED via wake / pause / wait /
                iter / close on the returned cycle object.
                """

                def __init__(self, core, threads, *, scheduler=_do_nothing):
                    super().__init__()

                    score = core.score
                    caller = 'cycle'

                    # Scheduler without an action is meaningless: the
                    # scheduler is the cycle's handle for driving any
                    # regulated children the action spawns; if the
                    # barrier has no action, no children get spawned,
                    # so a scheduler would never be invoked for the
                    # nested-tx case AND its action-complete late
                    # call would have nothing to scheduler-ish around.
                    if scheduler is not _do_nothing and core.user_action is None:
                        raise ValueError(
                            "cycle: scheduler= requires the Barrier "
                            "to have an action")

                    # Validate inputs before doing anything observable.
                    # TypeError for non-thread args, ThreadOrderingError
                    # for duplicates -- both checked before the count
                    # check so the user gets the most specific diagnostic.
                    seen = set()
                    for thread in threads:
                        if not isinstance(thread, threading.Thread):
                            raise TypeError(
                                f"cycle expected a thread, got {thread!r}")
                        if thread in seen:
                            raise ThreadOrderingError(
                                f"cycle: thread {thread.name!r} specified more than once")
                        seen.add(thread)

                    if len(threads) != core.parties:
                        raise ValueError(
                            f"cycle requires exactly {core.parties} waiter threads, got {len(threads)}")

                    # Create drivers for each thread.  From here on
                    # we hold scoreboard resources (Driver slots); any
                    # exception path must free them by closing the
                    # not-yet-terminal Drivers, or the next attempt
                    # at api.cycle(...) sees the slots still claimed
                    # and raises CompetingDriversError.
                    dispatch = score.Dispatch()
                    drivers = []
                    try:
                        for thread in threads:
                            driver = score.Driver(thread)
                            dispatch.add(driver)
                            drivers.append(driver)
                        waiters = drivers.copy()
                        opener = waiters.pop()

                        primitives = (core.primitive.wait, core.raw.wait)

                        # Validate each tx is at BLOCKED or WAITING on
                        # barrier.wait.
                        waiting_count = 0
                        for d in dispatch:
                            d.tx.validate(
                                method=primitives,
                                state=(State.BLOCKED, State.WAITING),
                                caller=caller,
                            )
                            if d.tx.state is State.WAITING:
                                waiting_count += 1

                        if opener.tx.state != State.BLOCKED:
                            raise ThreadOrderingError(
                                "cycle: last thread (opener) must be in BLOCKED state")

                        # All waiters must have been passed in.  No extra waiters.
                        already_waiting = len(core.actual._cond._waiters)
                        extra = already_waiting - waiting_count
                        if extra:
                            raise RuntimeError(
                                f"cycle: {extra} extra threads calling barrier.wait not passed in to cycle")

                        # Drive each BLOCKED waiter to parked@WAITING,
                        # sequentially.  Arrival order at the actual
                        # barrier determines the index each waiter
                        # receives, and tests rely on that order
                        # matching the user's cycle() argument order.
                        # Parallel drive (unblock all, then drain)
                        # lets OS scheduling interleave the actual.wait
                        # calls and randomizes index assignment.
                        # If the barrier breaks mid-drive, the
                        # offending wait tx goes RAISED and pursue's
                        # parking handler surfaces a RuntimeError
                        # (overshoot); convert it to the underlying
                        # BrokenBarrierError for caller clarity.
                        try:
                            for d in waiters:
                                if d.tx.state is State.BLOCKED:
                                    d.wait()
                                    d()
                                d.tx.validate(
                                    method=primitives,
                                    state=State.WAITING,
                                    caller=caller,
                                )
                        except RuntimeError:
                            for w in waiters:
                                if w.tx.state == State.RAISED:
                                    raise w.tx.result
                            raise

                        for d in waiters:
                            d.pausing()

                        opener.pausing()

                        # Fire each driver's staged lazy_work without
                        # waiting on its yield-condition: dispatch.add
                        # is bookkeeping, iteration would block in
                        # score.wait, and the scheduler callback below
                        # is the actor that advances state the wait
                        # would block on (e.g. driving a child tx
                        # spawned by the opener's action).  Drain
                        # fires the work; the post-scheduler iteration
                        # consumes the now-ready drivers.
                        dispatch.update(drivers)
                        dispatch.drain_recent()

                        if scheduler is not _do_nothing:
                            with unlock(score.lock):
                                scheduler()

                        for d in dispatch:
                            d.tx.validate(
                                method=primitives,
                                state=State.PAUSED,
                                caller=caller,
                            )

                        # remaining = thread -> Driver mapping in spec
                        # order (waiters first, opener last); dicts
                        # preserve insertion order.
                        self.remaining = {d.thread: d for d in drivers}
                        self.extra_waiters = 0
                    finally:
                        for d in drivers:
                            if not d.done:
                                d.close()

                def repr(self):
                    status = 'closed' if self.closed else f'{len(self.remaining)} remaining'
                    return f"<Barrier.cycle {status}>"


            @BoundInnerClass
            @base()
            class BarrierAPI(base.ConditionBaseAPI):
                def __init__(self, core, raw):
                    super().__init__(raw)

                def __repr__(self):
                    return self._core.fancy_repr('BarrierAPI')

                @property
                def waiters(self):
                    with self._lock:
                        return len(self._core.actual._cond._waiters)

                @BoundInnerClass
                class wait(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Barrier.wait")

                    @property
                    def index(self):
                        with self._lock:
                            return self._core.index

                    @property
                    def ran_action(self):
                        with self._lock:
                            return self._core.ran_action

                @BoundInnerClass
                class reset(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Barrier.reset")

                @BoundInnerClass
                class abort(base.TransactionAPI):
                    def __repr__(self):
                        return self._core.repr("Barrier.abort")

                @property
                def parties(self):
                    with self._lock:
                        return self._core.parties

                @property
                def n_waiting(self):
                    with self._lock:
                        return self._core.n_waiting

                @property
                def broken(self):
                    with self._lock:
                        return self._core.broken

                @BoundInnerClass
                class cycle(base.CycleAPIBase):
                    """Start a Barrier wake cycle.

                    The final thread is the opener (the final arrival
                    that opens the barrier and runs any action).  All
                    preceding threads are waiters.  Instantiating
                    returns a cycle object whose .waiters property
                    yields all named threads in spec order (waiters
                    first, opener last).
                    """

                def expire(self, method, *threads):
                    with self._lock:
                        return self._core.expire(method, threads)

                def disregard(self, method, *threads):
                    with self._lock:
                        return self._core.disregard(method, threads)

                def revert(self, method, *threads):
                    with self._lock:
                        return self._core.revert(method, threads)

    ###############################################################
    ###############################################################
    ##
    ##
    ##                    Driver / Chain / Dispatch
    ##
    ##
    ###############################################################
    ###############################################################

    @BoundInnerClass
    class Driver:
        """User-facing wrapper around the Driver state machine.

        A Driver attaches to a worker thread and drives it through
        named transaction-state transitions: SKIPPING (let the tx
        finish), PARKING (stop at a parking state and stay there),
        FINISHING (stop after the tx exits), and so on.  Imperatives
        on the Driver (skip, finish, block, commit, wait, stall,
        pausing, pause) request a state transition; the Driver
        advances the underlying tx and validates the result.

        Construct with a worker thread.  The Driver immediately
        registers itself in the score's drivers registry; only one
        Driver per thread is allowed.
        """

        # State constants, accessible at instance OR class level.
        idle       = base.Driver.idle
        active     = base.Driver.active
        skipping   = base.Driver.skipping
        parking    = base.Driver.parking
        finishing  = base.Driver.finishing
        parked     = base.Driver.parked
        finished   = base.Driver.finished
        raised     = base.Driver.raised
        terminated = base.Driver.terminated
        nesting    = base.Driver.nesting  # NESTING state (see .nested() imperative)

        driving_states  = base.Driver.driving_states
        active_states   = base.Driver.active_states
        terminal_states = base.Driver.terminal_states

        def __init__(self, scenario, thread):
            score = scenario._core
            self._lock = score.lock
            with self._lock:
                self._core = score.Driver(thread)
                score.driver_apis[self._core] = self

        def __repr__(self):
            with self._lock:
                return repr(self._core).replace("<Driver", "<Scenario.Driver")

        @property
        def thread(self):
            return self._core.thread

        @property
        def state(self):
            with self._lock:
                return self._core.state

        @property
        def tx(self):
            with self._lock:
                t = self._core.tx
                return t.api if t is not None else None

        @property
        def txs(self):
            with self._lock:
                return tuple(t.api for t in self._core.txs)

        @property
        def done(self):
            with self._lock:
                return self._core.done

        def skip(self):
            with self._lock:
                self._core.skip()

        def finish(self):
            with self._lock:
                self._core.finish()

        def block(self):
            with self._lock:
                self._core.block()

        def commit(self):
            with self._lock:
                self._core.commit()

        def wait(self):
            with self._lock:
                self._core.wait()

        def stall(self):
            with self._lock:
                self._core.stall()

        def pause(self):
            with self._lock:
                self._core.pause()

        def nested(self):
            with self._lock:
                self._core.nested()

        def __call__(self):
            with self._lock:
                self._core()

        def close(self):
            """Release this Driver's slot in the scenario's per-
            thread driver registry, freeing the worker thread for a
            new Driver (e.g. scenario.finish(thread)).  Idempotent;
            safe from any state.

            close() does not drive the worker -- it only releases
            the score's bookkeeping.  If the worker is still mid-
            transaction, call scenario.finish(thread) afterward to
            drive it to a terminal state.
            """
            with self._lock:
                self._core.close()


    @BoundInnerClass
    class Chain:
        """User-facing wrapper around the Chain.

        A Chain is an ordered sequence of Drivers.  When a Chain is
        added to a Dispatch, the head of pending is promoted to
        current and driven until it reaches a terminal state; then
        the next pending head is promoted, and so on, until the
        chain is empty.

        Construct with zero or more Drivers; append() and remove()
        adjust the pending list at any time, including during
        iteration of the owning Dispatch.  Detach a Chain from a
        Dispatch with Dispatch.remove(chain): the chain's current
        Driver (if any) is unregistered and dropped from the
        Dispatch, but the pending list stays intact so the Chain
        remains useful (add it to another Dispatch and the pending
        head activates).
        """

        def __init__(self, scenario, *drivers):
            score = scenario._core
            self._lock = score.lock
            self._score = score
            with self._lock:
                self._core = score.Chain(*[d._core for d in drivers])

        def __repr__(self):
            with self._lock:
                return repr(self._core).replace("<Chain", "<Scenario.Chain")

        def __len__(self):
            with self._lock:
                return len(self._core)

        def __bool__(self):
            with self._lock:
                return bool(self._core)

        @property
        def pending(self):
            with self._lock:
                return tuple(self._score.driver_apis[d]
                             for d in self._core.pending)

        def append(self, driver):
            with self._lock:
                self._core.append(driver._core)

        def remove(self, driver):
            with self._lock:
                self._core.remove(driver._core)

        def promote(self):
            """Pop and return the pending-head Driver without driving it.
            Returns None if pending is empty.  The returned Driver is
            unowned -- the caller must register or close it.  Useful
            for custom iteration patterns: e.g.
                while (d := chain.promote()) is not None:
                    ... # caller chooses whether to drive d
            """
            with self._lock:
                core_d = self._core.promote()
                if core_d is None:
                    return None
                return self._score.driver_apis[core_d]

        def __contains__(self, driver):
            return driver._core in self._core

        def __iter__(self):
            return self

        def __next__(self):
            with self._lock:
                core_d = next(self._core)
                return self._score.driver_apis[core_d]

        def close(self):
            """Close all Drivers in this Chain's pending list and
            empty it.  Drivers that have been promoted out of the
            Chain (via iteration, promote(), or Dispatch handling)
            are owned by their new consumer and aren't touched here.
            Idempotent."""
            with self._lock:
                self._core.close()


    @BoundInnerClass
    class Dispatch:
        """User-facing wrapper around the Dispatch driver-yielding
        iterator.

        Add Drivers (or Chains of Drivers) via add().  Iterating
        yields Drivers as each becomes active (after some signal
        that requires user attention) or done (terminal state);
        the user typically issues an imperative on the yielded
        Driver to advance it, then resumes iteration.

        A Dispatch wakes via a single score.wait on the union of all
        member Drivers' listening sets, then routes each woken signal
        to the right Driver's signal() method.  For sequencing within
        a subset of Drivers, wrap them in a Chain and add the chain --
        Dispatch will activate the chain's pending Drivers one at a
        time as each prior yields.
        """

        def __init__(self, scenario):
            score = scenario._core
            self._lock = score.lock
            self._score = score
            with self._lock:
                self._core = score.Dispatch()

        def __repr__(self):
            with self._lock:
                return repr(self._core).replace("<Dispatch", "<Scenario.Dispatch")

        def add(self, driver):
            with self._lock:
                self._core.add(driver._core)

        def update(self, drivers):
            with self._lock:
                for driver in drivers:
                    self._core.add(driver._core)

        def remove(self, driver):
            with self._lock:
                self._core.remove(driver._core)

        def discard(self, driver):
            with self._lock:
                self._core.discard(driver._core)

        def __contains__(self, driver):
            return driver._core in self._core

        def __iter__(self):
            return self

        def __next__(self):
            with self._lock:
                core = next(self._core)
                return self._score.driver_apis[core]

        def close(self):
            """Close all objects (Chains and Drivers) inside this
            Dispatch and clear the Dispatch.  Idempotent."""
            with self._lock:
                self._core.close()

    ###############################################################
    ###############################################################
    ##
    ##
    ##               _           _ _   _
    ##    _ __  _ __(_)_ __ ___ (_) |_(_)_   _____
    ##   | '_ \| '__| | '_ ` _ \| | __| \ \ / / _ \
    ##   | |_) | |  | | | | | | | | |_| |\ V /  __/
    ##   | .__/|_|  |_|_| |_| |_|_|\__|_| \_/ \___|
    ##   |_|
    ##
    ##
    ##
    ##
    ###############################################################
    ###############################################################

    @BoundInnerClass
    class ThreadingImpersonator:
        """A drop-in for the threading module that returns scenario
        primitives for the seven names inject patches, and falls through
        to the real threading module for everything else.

        Used when a target module has `import threading; threading.Lock()`-
        style references; the target's `threading` attribute is replaced
        with an instance of this class so attribute lookups for primitive
        names yield the scenario versions while other lookups (like Thread)
        keep working.
        """

        def __init__(self, scenario):
            # Stash the scenario API instance for the repr.  Use a
            # dunder-mangled name so it doesn't show up in attribute
            # lookups (which fall through to threading via __getattr__
            # for everything else).
            self.__scenario = scenario
            for name in scenario._core.primitive_names:
                setattr(self, name, getattr(scenario, name))

        def __getattr__(self, name):
            return getattr(threading, name)

        def __repr__(self):
            return f"<ThreadingImpersonator {self.__scenario!r}>"

    @BoundInnerClass
    class inject:
        """Monkey-patch threading-primitive references in a module
        so calls construct blanket primitives bound to this scenario.

        Two reference patterns are intercepted:

        1.  Names bound directly to a threading primitive class:
                from threading import Lock     # target.Lock is threading.Lock
                Mutex = threading.Lock         # target.Mutex is threading.Lock
            Each such name is rebound to the corresponding scenario
            primitive class.  Identity-checked: a user-defined class
            that happens to share the name 'Lock' is left alone.

        2.  A module attribute whose value is the threading module
            itself:
                import threading               # target.threading is threading
            That attribute is replaced with a small stand-in object
            whose .Lock / .RLock / .Condition / .Semaphore /
            .BoundedSemaphore / .Event / .Barrier are this scenario's
            primitives, and whose other attribute lookups fall
            through to the real threading module.  Calls like
            target_module.threading.Lock() then construct a blanket
            Lock.  Only triggered when target.threading is the actual
            threading module (skipped if user has reassigned the
            name to something else).

        Returns an Injection handle.  The handle is a context manager;
        its __exit__ calls self.close().  close() restores the
        pre-inject values, after first verifying that what inject
        installed is still in place (to refuse to clobber a later
        inject's overrides).

        Raises ValueError if no patchable references are found in
        the target module -- that's almost certainly a user mistake
        (wrong module, target imports threading lazily inside a
        function, primitive references already swapped by an earlier
        inject, etc.).
        """

        def __init__(self, scenario, module):
            # don't bother to lock the scenario lock for this.
            impersonator = scenario.ThreadingImpersonator()
            self._injection = scenario._core.Injection(module, impersonator)

        def __repr__(self):
            status = "closed" if self._injection.closed else f"{len(self._injection.replacements)} replacements"
            return f"<inject {self._injection.module.__name__!r} {status}>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            return False

        def close(self):
            return self._injection.close()


    @base()
    @BoundInnerClass
    class Primitive:
        """Base class for primitives."""

        def __init__(self, scenario, core):
            self._lock = core.lock
            self._core = core

        @property
        def __class__(self):
            return self._core.actual.__class__

        def __dir__(self):
            return dir(self._core.actual)

        @property
        def name(self):
            with self._lock:
                return self._core.name

        @name.setter
        def name(self, value):
            with self._lock:
                self._core.name = value

    @base()
    @BoundInnerClass
    class LockPrimitive(base.Primitive):
        """Base class for Lock with method implementations."""

        def _is_owned(self):
            return self._core._is_owned(self._core)()

        def _release_save(self):
            return self._core._release_save(self._core)()

        def _acquire_restore(self, state):
            return self._core._acquire_restore(self._core, state)()

        def _at_fork_reinit(self):
            with self._lock:
                actual = self._core.actual
                if hasattr(actual, '_at_fork_reinit'):
                    actual._at_fork_reinit()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            return False

    @base()
    @BoundInnerClass
    class RLockPrimitive(base.LockPrimitive):
        # threading.RLock._recursion_count() was added in Python 3.12.
        # Mirror the underlying type: expose the method only when the
        # underlying has it, so hasattr() returns the same answer as
        # for a raw threading.RLock.
        if hasattr(threading.RLock(), '_recursion_count'):
            def _recursion_count(self):
                with self._lock:
                    return self._core.actual._recursion_count()

    @base()
    @BoundInnerClass
    class ConditionPrimitive(base.Primitive):
        """Base class for Condition with method implementations."""

        def acquire(self, blocking=True, timeout=-1):
            ul = self._core.underlying.primitive
            with ul._lock:
                return ul._core(ul.acquire, True, blocking=blocking, timeout=timeout, entry=self)

        def release(self):
            ul = self._core.underlying.primitive
            with ul._lock:
                return ul._core(ul.release, True, entry=self)

        def locked(self):
            ul = self._core.underlying.primitive
            with ul._lock:
                return ul._core(ul.locked, True, entry=self)

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            return False



    @base()
    @BoundInnerClass
    class SemaphorePrimitive(base.Primitive):
        """Base class for Semaphore with method implementations."""

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            return False


    @base()
    @BoundInnerClass
    class BarrierPrimitive(base.Primitive):
        """Base class for Barrier with method implementations."""

        @property
        def parties(self):
            with self._lock:
                return self._core.parties

        @property
        def n_waiting(self):
            with self._lock:
                return self._core.n_waiting

        @property
        def broken(self):
            with self._lock:
                return self._core.broken

    @BoundInnerClass
    class RawLock(base.LockPrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def acquire(self, blocking=True, timeout=-1):
            with self._lock:
                return self._core(self.acquire, False, blocking=blocking, timeout=timeout)

        def release(self):
            with self._lock:
                return self._core(self.release, False)

        def locked(self):
            with self._lock:
                return self._core(self.locked, False)

        def __repr__(self):
            return self._core.fancy_repr('Lock.raw')

    @BoundInnerClass
    class RawRLock(base.RLockPrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def acquire(self, blocking=True, timeout=-1):
            with self._lock:
                return self._core(self.acquire, False, blocking=blocking, timeout=timeout)

        def release(self):
            with self._lock:
                return self._core(self.release, False)

        if _rlock_provides_locked:  # pragma: no cover
            def locked(self):
                with self._lock:
                    return self._core(self.locked, False)

        def __repr__(self):
            return self._core.fancy_repr('RLock.raw')

    @BoundInnerClass
    class RawCondition(base.ConditionPrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, False, timeout=timeout)

        def wait_for(self, predicate, timeout=None):
            with self._lock:
                return self._core(self.wait_for, False, predicate=predicate, timeout=timeout)

        def notify(self, n=1):
            with self._lock:
                return self._core(self.notify, False, n=n)

        def notify_all(self):
            with self._lock:
                return self._core(self.notify_all, False)

        notifyAll = notify_all
        def __repr__(self):
            return self._core.fancy_repr('Condition.raw')


    @BoundInnerClass
    class RawSemaphore(base.SemaphorePrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def acquire(self, blocking=True, timeout=None):
            with self._lock:
                return self._core(self.acquire, False, blocking=blocking, timeout=timeout)

        def release(self, n=1):
            with self._lock:
                return self._core(self.release, False, n=n)

        def __repr__(self):
            return self._core.fancy_repr('Semaphore.raw')

    @BoundInnerClass
    class RawBoundedSemaphore(base.SemaphorePrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def acquire(self, blocking=True, timeout=None):
            with self._lock:
                return self._core(self.acquire, False, blocking=blocking, timeout=timeout)

        def release(self, n=1):
            with self._lock:
                return self._core(self.release, False, n=n)

        def __repr__(self):
            return self._core.fancy_repr('BoundedSemaphore.raw')


    @BoundInnerClass
    class RawEvent(base.Primitive):

        def __init__(self, score, core):
            super().__init__(core)

        def is_set(self):
            with self._lock:
                return self._core(self.is_set, False)

        def isSet(self):
            with self._lock:
                return self._core(self.isSet, False)

        def set(self):
            with self._lock:
                return self._core(self.set, False)

        def clear(self):
            with self._lock:
                return self._core(self.clear, False)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, False, timeout=timeout)

        def __repr__(self):
            return self._core.fancy_repr('Event.raw')

    @BoundInnerClass
    class RawBarrier(base.BarrierPrimitive):

        def __init__(self, score, core):
            super().__init__(core)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, False, timeout=timeout)

        def reset(self):
            with self._lock:
                return self._core(self.reset, False)

        def abort(self):
            with self._lock:
                return self._core(self.abort, False)

        def __repr__(self):
            return self._core.fancy_repr('Barrier.raw')


    @BoundInnerClass
    class Lock(base.LockPrimitive):
        def __init__(self, scenario):
            score = scenario._core
            core = score.LockCore(self)
            super().__init__(core)

        def acquire(self, blocking=True, timeout=-1):
            with self._lock:
                return self._core(self.acquire, True, blocking=blocking, timeout=timeout)

        def release(self):
            with self._lock:
                return self._core(self.release, True)

        def locked(self):
            with self._lock:
                return self._core(self.locked, True)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('Lock')
            return self._core.compatibility_repr()

    @BoundInnerClass
    class RLock(base.RLockPrimitive):
        def __init__(self, scenario):
            score = scenario._core
            core = score.RLockCore(self)
            super().__init__(core)

        def acquire(self, blocking=True, timeout=-1):
            with self._lock:
                return self._core(self.acquire, True, blocking=blocking, timeout=timeout)

        def release(self):
            with self._lock:
                return self._core(self.release, True)

        if _rlock_provides_locked:  # pragma: no cover
            def locked(self):
                with self._lock:
                    return self._core(self.locked, True)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('RLock')
            return self._core.compatibility_repr()

    @BoundInnerClass
    class Condition(base.ConditionPrimitive):
        def __init__(self, scenario, lock=None):
            score = scenario._core
            core = score.ConditionCore(self, lock)
            super().__init__(core)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, True, timeout=timeout)

        def wait_for(self, predicate, timeout=None):
            with self._lock:
                return self._core(self.wait_for, True, predicate=predicate, timeout=timeout)

        def notify(self, n=1):
            with self._lock:
                return self._core(self.notify, True, n=n)

        def notify_all(self):
            with self._lock:
                return self._core(self.notify_all, True)

        notifyAll = notify_all

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('Condition')
            return self._core.compatibility_repr()


    @BoundInnerClass
    class Semaphore(base.SemaphorePrimitive):
        def __init__(self, scenario, value=1):
            score = scenario._core
            core = score.SemaphoreCore(self, value)
            super().__init__(core)

        def acquire(self, blocking=True, timeout=None):
            with self._lock:
                return self._core(self.acquire, True, blocking=blocking, timeout=timeout)

        def release(self, n=1):
            with self._lock:
                return self._core(self.release, True, n=n)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('Semaphore')
            return self._core.compatibility_repr()

    @BoundInnerClass
    class BoundedSemaphore(base.SemaphorePrimitive):
        def __init__(self, scenario, value=1):
            score = scenario._core
            core = score.BoundedSemaphoreCore(self, value)
            super().__init__(core)

        def acquire(self, blocking=True, timeout=None):
            with self._lock:
                return self._core(self.acquire, True, blocking=blocking, timeout=timeout)

        def release(self, n=1):
            with self._lock:
                return self._core(self.release, True, n=n)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('BoundedSemaphore')
            return self._core.compatibility_repr()


    @BoundInnerClass
    class Event(base.Primitive):
        def __init__(self, scenario):
            score = scenario._core
            core = score.EventCore(self)
            super().__init__(core)

        def is_set(self):
            with self._lock:
                return self._core(self.is_set, True)

        def isSet(self):
            with self._lock:
                return self._core(self.isSet, True)

        def set(self):
            with self._lock:
                return self._core(self.set, True)

        def clear(self):
            with self._lock:
                return self._core(self.clear, True)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, True, timeout=timeout)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('Event')
            return self._core.compatibility_repr()

    @BoundInnerClass
    class Barrier(base.BarrierPrimitive):
        def __init__(self, scenario, parties, action=None, timeout=None):
            score = scenario._core
            core = score.BarrierCore(self, parties, action, timeout)
            super().__init__(core)

        def wait(self, timeout=None):
            with self._lock:
                return self._core(self.wait, True, timeout=timeout)

        def reset(self):
            with self._lock:
                return self._core(self.reset, True)

        def abort(self):
            with self._lock:
                return self._core(self.abort, True)

        def __repr__(self):
            if self._core.use_fancy_repr:
                return self._core.fancy_repr('Barrier')
            return self._core.compatibility_repr()

    # User-facing aliases for the API wrapper classes, surfaced on the
    # Scenario API so callers can isinstance-check the objects they
    # receive without reaching into the nested core.  Each is the
    # *unwrapped* class (not the BoundInnerClass descriptor): a stable
    # isinstance target that resolves to the same object at both
    # Scenario.X and scenario.X.  The descriptor itself can't be used
    # -- it tries to rebind against Scenario on instance access, and
    # its base lives deep in the core, so that fails.
    #
    # The primitive APIs aren't subclassed, so @base() sits below
    # @BoundInnerClass on each and registers the unwrapped class
    # directly.  TransactionAPI is subclassed by every per-method tx
    # class (which need the bound form), so it registers twice: once
    # unwrapped as UnboundTransactionAPI (for this alias) and once
    # wrapped as TransactionAPI (for those subclasses to inherit).
    #
    # Naming: bare name where it's free (Transaction -- no primitive
    # owns it), API-suffix where the cooked primitive already owns the
    # base name (LockAPI, etc.).  The cooked primitives, Raw handles,
    # Driver, and Dispatch are already surfaced as direct members
    # above, so they're not repeated here.
    LockAPI = base.LockAPI
    RLockAPI = base.RLockAPI
    ConditionAPI = base.ConditionAPI
    EventAPI = base.EventAPI
    SemaphoreAPI = base.SemaphoreAPI
    BoundedSemaphoreAPI = base.BoundedSemaphoreAPI
    BarrierAPI = base.BarrierAPI
    Transaction = base.UnboundTransactionAPI


# Internal aliases on the scenario core, purely for our own
# convenience: framework code checks isinstance against the transaction
# classes, and without these it would spell out the full nested path
# every time (the tx API wrapper and the tx core classes otherwise sit
# several attributes deep inside Core).  These are NOT
# user-facing -- the user-facing aliases live on Scenario itself.  They
# reach the same unwrapped classes via plain attribute access, so they
# work as isinstance targets both off the class
# (Scenario._ScenarioCore.X) and off a score instance (score.X).
#
# Note to future selves: this is an ad-hoc convenience set, not an
# interface.  Add or remove entries here freely as internal needs
# change -- each is just a reference to an existing class, and nothing
# depends on the set being complete.  Current entries cover the tx API
# wrapper and the four tx core classes, which is everything the present
# isinstance checks need.
_core_cls = Scenario._ScenarioCore.Core
Scenario._ScenarioCore.TxAPI = _core_cls.API.TransactionAPI
Scenario._ScenarioCore.Transaction = _core_cls.Transaction
Scenario._ScenarioCore.TimeoutTransaction = _core_cls.TimeoutTransaction
Scenario._ScenarioCore.WaitingTransaction = _core_cls.WaitingTransaction
Scenario._ScenarioCore.StallingTransaction = _core_cls.StallingTransaction
del _core_cls


mm()
