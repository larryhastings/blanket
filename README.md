# blanket

## Deterministic multithreaded testing for Python

##### Copyright 2025-2026 by Larry Hastings

> *Your test should be effectively single-threaded*.
> *If it isn't, you haven't blanketed hard enough*.
> *Slow it down*.

## Overview

**blanket** is a library for writing deterministic tests of multithreaded
Python code.

Have you ever tried to write a regression test for code running on
multiple threads?  It's a real headache.  Threading bugs are caused
by race conditions and ordering mistakes--the failures arise from a
specific sequence of how two or more threads interact.  But this
sequence is out of your control!  When you synchronize multiple threads,
you use "synchronization primitives" from the `threading` module--`Lock`,
`Event`, and the like.  But those are out of your control--you don't pick
which thread gets the lock.  Instead, your operating system's scheduler
effectively decides which thread gets the lock, seemingly at random.  This
makes it exceedingly hard to create a reproducible test case for
threading-related bugs.  And if you can't reproduce it reliably, you
can't debug it, and your unit test suite can't actually test it.

This problem only gets harder if you're shooting for 100% coverage.
Code paths that handle rare race conditions are by definition *rare*.
But if you can't write a test that reliably reproduces the condition,
how do you test it in your test suite?  How do you get to 100%?

And this problem is only going to get harder. Python's "nogil" mode
will become the default--someday soon--and more and more code will
have to become multithreaded-aware.  Code that was reliable with the
GIL may start exhibiting bugs it never used to.

Enter **blanket**.  **blanket** takes away the randomness of the
synchronization primitives and puts *you* in control.
Using **blanket**, you can control the behavior of the primitives--instead
of the `Lock` randomly choosing which thread it gives itself to next,
*you* choose.  When you write your test using **blanket**, every test
becomes 100% deterministic, reliably reproducing even the most obscure
race condition.  Every time.  100% coverage restored!

One design choice worth mentioning up front: **blanket** *wraps*
the real `threading.Lock`, `threading.Condition`, `queue.Queue`, and so
on, rather than reimplementing them. Your tests use the real primitives,
which means they're guaranteed to behave like the real thing--because
they *are* the real thing, just under **blanket** control.  But for
this to work, your code has to replace the real `threading` and `queue`
primitives with **blanket**-wrapped versions.

**blanket** 1.1 supports CPython 3.10 or newer.  The release
test matrix covers CPython 3.10 through 3.14.  It depends on
[my **big** library,](https://github.com/larryhastings/big)
and on the
[**bytecode**](https://pypi.org/project/bytecode/) module used by
the injector.  **blanket** is 100% pure Python.

The current version is **1.1**.

## Quickstart

Here's a small Python program that exercises three threads, one
shared `Lock`, and a `Barrier(3)`. The OS scheduler decides who
gets the lock first, second, and third, and who exits the barrier
first, second, and third. That's six possibilities times six
possibilities--thirty-six different orderings, and you have
absolutely no control over which one you get on any particular run.

```
import random
import threading

lock = threading.Lock()
barrier = threading.Barrier(3)

def worker(name):
    with lock:
        print(f"worker {name} got the lock")
    barrier.wait()
    print(f"worker {name} is past the barrier")

A = threading.Thread(target=worker, args=('A',))
B = threading.Thread(target=worker, args=('B',))
C = threading.Thread(target=worker, args=('C',))

threads = [A, B, C]
random.shuffle(threads)
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Run that a few times. You'll almost certainly get a
different ordering each time you run it.  The order of
operations is out of your control.

Let's rewrite it using **blanket**:

```
import blanket
import random
import threading

scenario = blanket.Scenario()

lock = scenario.Lock()
barrier = scenario.Barrier(3)

def worker(name):
    with lock:
        print(f"worker {name} got the lock")
    barrier.wait()
    print(f"worker {name} is past the barrier")

A = threading.Thread(target=worker, args=('A',))
B = threading.Thread(target=worker, args=('B',))
C = threading.Thread(target=worker, args=('C',))

threads = [A, B, C]
random.shuffle(threads)

lock_api = scenario.api(lock)
barrier_api = scenario.api(barrier)

with scenario:
    for t in threads:
        t.start()
    list(lock_api.relay(B, A, C))
    lock_api.unblock(lock.release, C)
    with barrier_api.cycle(C, A, B):
        pass
for t in threads:
    t.join()
```

Notice how little had to change.  We replaced the lock and
barrier with **blanket** versions, then `relay` controlled
the order of who got the lock, `unblock` let the final
`lock.release` call on C run, and `cycle` walked the
three threads through the barrier, even controlling what order
they resumed in.  And now--the randomness is *gone*.
Every time you run this version, you'll get the same
output, on any machine:

```
worker B got the lock
worker A got the lock
worker C got the lock
worker C is past the barrier
worker A is past the barrier
worker B is past the barrier
```


## Why blanket Works

The trouble with testing multithreaded code is that
synchronization primitives--locks, condition variables,
semaphores, barriers, and so on--are non-deterministic.
When you call one to synchronize, you don't know when
it will return, and what other threads it let run before
you.  Obviously these primitives are crucial to making
multithreaded code possible--but they come with a
terrible cost.

Thankfully, we can fix it!  **blanket** replaces the threading
synchronization primitives with perfect duplicates--but
these don't randomly choose who runs next.  Instead, they
stop and wait for further instructions.  Every method call
on a **blanket** primitive--`lock.acquire()`, `condition.wait()`,
`event.set()`, all of it--first passes through a sentinel
point we call the *scheduler block*.  The call waits there
until your code, which we call *the scheduler*, gives it
permission to run.  By controlling the order in which the
scheduler lets these calls run--in **blanket** parlance,
by controlling the *tempo*--you indirectly control the state
of the synchronization primitives, and thus the behavior of
your entire program.


## Terminology

**blanket** is a new library, doing kind of a new thing.  So we have
to establish some new terms we're going to use for working with **blanket**.

A *scenario* is the top-level object that owns a set of **blanket**
synchronization primitives and any number of managed worker threads.
You create one by calling `blanket.Scenario()`. Most of what you
do with **blanket** is do things to (or through) a scenario.

To *enter the scenario* means to enter the `with scenario:` block.
Almost everything interesting about driving worker threads requires
being inside that block.

The *scheduler* is the thread that does the work of driving the
worker threads. Almost always this is your test's main thread:
your main thread enters the scenario, calls **blanket** APIs to
drive the workers around, and then exits the scenario. The role
of scheduler is *temporary*--it only applies while the main thread
is inside the `with scenario:` block.

A *worker thread* is any thread you've registered with the scenario
via `scenario.thread(target)`. Worker threads are the threads
that actually exercise the code you're testing. They use **blanket**'s
synchronization primitives normally--`with lock:`, `event.wait()`,
and so on. The threads don't know they're being scripted.

A *primitive*, or *primitive handle*, is a **blanket** `Lock`,
`RLock`, `Condition`, `Semaphore`, `BoundedSemaphore`, `Event`,
`Barrier`, or queue object. You construct them on the scenario:
`scenario.Lock()`, `scenario.Barrier(3)`, `scenario.Queue()`, etc.
Each primitive is wired into the scenario, making every method call
on it observable and steerable by the scheduler.

A *raw*, or *raw handle*, is a parallel handle for the same underlying
synchronization object.  The difference is, these calls don't bother
with **blanket**'s regulation.  Calls to methods on raw handles just
call the real method, immediately, and return immediately when it's
done.  You can get a raw handle via `scenario.raws[primitive]` or
`scenario.raw(primitive)`. The raw handle is useful inside the
`with scenario:` block when you want a call to skip regulation--either
because the scheduler itself needs to act on the primitive, or
because you're giving the handle to a worker or subsystem you don't
want to script.  (Outside the scenario, you don't need raws at
all--**blanket** primitives are unregulated outside the
`with scenario:` block.)

An *actual* handle is a real handle to a real `threading` module
synchronization primitive.  **blanket** implements its primitives
by wrapping the real thing; when you call `acquire` on a **blanket**
`Lock`, at its heart **blanket** will call `acquire` on a real
`threading.Lock`.  You never see these when you work with **blanket**,
but it's an implementation detail worth knowing about.

A *transaction* is the wrapper object **blanket** creates around
each method call on a primitive. Every `lock.acquire()`,
every `condition.wait()`, every `barrier.wait()` becomes a transaction.
The transaction has a state, a current method, a thread, and a
small handful of operations the scheduler can invoke on it. The
transaction is the unit at which **blanket** does its work.

The *tempo* is the linearized sequence of synchronization method
calls that the scheduler is permitting to complete. The fundamental
move in **blanket** is "decide what the tempo should be, then make
it so."

A *signal* is an observable condition the scheduler can `wait()` on:
a thread terminating, a transaction reaching a particular state,
a particular method being called, and so on.

*Parking* is the act of making a thread stop and wait.  (This
term is borrowed from Java, where you make a thread *park*,
waiting for a *permit* that allows it to resume.)  In **blanket**,
we say that we "park" a thread, which generally means a transaction
has entered one of its "parking states".

There are several places where the scheduler can choose to park
a transaction.  To make it easier to talk about, **blanket**
gives them special names:

* the *scheduler block*, which happens *before* calling the
  actual method.
* the *scheduler stall*, a specific mid-transaction park
  only used for certain transactions.
* the *scheduler pause*, which happens *after* calling the
  actual method.

We'll learn more about these later.

There is a fourth, called the *actual wait*, which is the
parking state inside the actual method call.  When you call
`lock.acquire()` on a real `threading.Lock`, if you have to wait
for the lock to become available, you "park", and **blanket**
gives this parking state the name the "actual wait".  The
difference is, **blanket** has no control over this parking
state; the scheduler can't directly control it, it's managed
by the actual primitive.

## Getting Started

### Requirements

**blanket** 1.1 supports CPython 3.10 or newer.  I currently test
it on CPython 3.10 through 3.14.  It depends on
[**big**](https://github.com/larryhastings/big) and
[**bytecode**.](https://pypi.org/project/bytecode/) That's it.

### The Shape Of A blanket Test

Every **blanket** test follows the same overall shape:

```
# 1. Create the scenario, the primitives, and the worker threads.
scenario = blanket.Scenario()
lock = scenario.Lock()

def worker():
    with lock:
        ...

t = scenario.thread(worker)
t2 = scenario.thread(worker)

# 2. Enter the scenario.  Your main thread is now the scheduler.
with scenario:
    ...

# 3. Exit the scenario, and test the resulting state of the system.
assert ...
# or
self.assertTrue(...)
```

Notice that all the setup happens *before* entering the scenario.
This is on purpose. The scheduler-on-main-thread role only applies
inside `with scenario:`. Setup that doesn't need scheduler control
(creating primitives, defining worker functions, registering threads)
should happen before your main thread becomes the scheduler.

Notice also: the worker thread code is *the actual production code
you're testing.* The workers use `with lock:` and `lock.acquire()`
just like they would in production. **blanket** doesn't require
you to modify the code under test.  All you have to do is substitute
the real synchronization objects with their **blanket** equivalents.
That's all **blanket** needs to work its magic.

### A Note On Naming

The examples in this document follow a small, consistent naming
convention that matches how I write **blanket** code in practice:

- Threads get single uppercase letters: `A`, `B`, `C`, ..., `Z`.
- A `Lock` primitive is `lock`; its API object is `lock_api`.
  Similarly `cond`/`cond_api`, `event`/`event_api`, and so on.
- Multiple primitives of the same kind get descriptive names
  (`reader_lock`, `writer_lock`), or, worst case, numbers.
- The scenario is `scenario`, or `s` when brevity matters.
- If a test has only a single primitive, I often just call its
  API object `api`.
- I frequently abbreviate "transaction" as `tx`.

Of course, you're under no obligation to follow these conventions
in your own code.  That's just what I use in my own code, and what
I'll use here in the documentation.

## The Scenario

The `Scenario` is the top-level **blanket** object. The class is
at module scope in **blanket** and takes no arguments:

```
scenario = blanket.Scenario()
```

This object is the central manager for everything you do
with **blanket**.  It contains the classes for the replacement
primitives (`scenario.Lock()`, etc.), it can create and
manage worker threads for you (created using `scenario.thread()`),
and it has helper methods useful when running a test.  Using a
scenario, you can:

- inspect a thread's current transaction (`scenario.transaction`)
- wait for something to happen (`scenario.wait`)
- drive worker threads through method calls with the middle-level APIs: `scenario.park`, `scenario.skip`, `scenario.block`, `scenario.pause`, `scenario.Driver`, `scenario.Chain`, and `scenario.Dispatch`
- monkey-patch another module so it uses **blanket** `threading` and `queue` primitives (`scenario.inject`)

But that's just a taste.  We'll go over all the things you can do with
a scenario over the course of this document--there's a lot of 'em.

### Entering The Scenario

Most of the interesting **blanket** APIs require that you be
*inside the scenario*--inside a `with scenario:` block.
This is where your main thread "becomes the scheduler".

When you first create your scenario, you can construct primitives,
register and even start worker threads.  But you can't control
anything; the synchronization primitives behave just like normal
synchronization primitives.

But that's only until you enter the scenario.  Once you enter
the scenario, the primitives automatically stop every time any
thread calls a method on them.  They're waiting for instructions
from you--you have now "become the scheduler".  Until you "exit
the scenario", you have control over when the primitive methods
run, and that gives you all the control you need.

You can enter a scenario more than once.  Outside the scenario,
things behave like normal.  Inside the scenario, you're in
control.


### Worker Threads

As a rule, your tests should be written with two or more
"worker threads" doing the actual work, and your main thread
entering the scenario and becoming the scheduler.  (Why two?
If you only have one worker thread... what do you need
cross-thread synchronization for?!)

The worker threads do the actual work of the test.  They'll
call into your code to exercise it, and must use **blanket**
synchronization primitives to synchronize with each other.
Done correctly, they should be completely unaware they're
running inside **blanket**.

You *can* create your worker threads the normal way, with
`threading.Thread`.  However, **blanket** provides a helper
that makes creating threads easy: `scenario.thread(target, *args, **kwargs)`.
This creates the thread, passing in the `*args` and `**kwargs`
you specified, and it returns the thread handle.

Threads created by `scenario.thread` are also automatically
managed for you by the scenario, so we call them
"managed threads".  Here are the extras you get for free
with a managed thread:

* If you create a managed thread before entering the scenario,
  the scenario automatically starts the thread for you when you
  enter the scenario.
* If you create a managed thread while inside the scenario,
  `scenario.thread` starts the thread immediately.
* When you exit the scenario, the scenario will `join` all
  managed threads, waiting until each one terminates.

Once running, as long as the worker is only running ordinary
Python code, it runs at full speed, just like any thread would.
The interesting thing happens the moment the worker calls a method
on a **blanket** primitive: the call waits, sleeping until the
scenario gives it permission to proceed.  That's the magic
that makes **blanket** work.


### The Primitives

A scenario supplies regulated versions of the synchronization objects
most Python programs use from `threading`:

* `Lock`
* `RLock`
* `Condition`
* `Barrier`
* `Event`
* `Semaphore`
* `BoundedSemaphore`

In 1.1, a scenario also supplies regulated versions of the queue
classes from `queue`:

* `SimpleQueue`, on Python versions whose stdlib has `queue.SimpleQueue`
* `Queue`
* `LifoQueue`
* `PriorityQueue`

The objects they return behave like the stdlib objects, with the same
methods taking the same arguments for the Python version you're running.
**blanket** tries hard to impersonate the stdlib exactly.  For example,
`RLock.locked`, `Condition.locked`, `queue.SimpleQueue`, and
`Queue.shutdown` only appear on the corresponding **blanket** objects
when they appear on the real stdlib objects.

**blanket** doesn't currently regulate the `asyncio` synchronization
primitives.  They're similar to the `threading` primitives, but they live
on an event loop and aren't for synchronizing OS threads.  That's a
different problem.  If **blanket** ever grows support for `asyncio`, it'll
probably want a separate async-flavored design, not a quick copy of the
threading API.

When you create a `scenario.Lock()` or `scenario.Queue()`, you get back
a *primitive handle*, or just a *primitive* for short.  Method calls on
the primitive behave exactly like the real thing--until you enter the
scenario.  Once you do, the primitive is *regulated*: it blocks when it's
called, to let you control it.
Outside the `with scenario:` block, **blanket** primitives are
*unregulated*: calls pass straight through to the real primitive,
what we call the *actual* primitive. This means outside the scenario
you can just make ordinary calls into the primitives to change their
state:

```
event = scenario.Event()

# this works fine; we're outside the scenario
event.set()

with scenario:
    # inside the scenario--
    # don't call a method on the primitive here!
    ...

# also fine, we're outside the scenario again
event.clear()
```

Also, for every primitive handle, there's a matching *raw handle*,
or *raw* for short, which you can get from the scenario.

```
raw_lock = scenario.raws[some_random_lock]
# or:
raw_lock = scenario.raw(some_random_lock)
```

The *raw* is a second handle to the same internal objects; methods on
the *primitive* and the *raw* both change the internal state of the
object in the same way. The only difference is that the raw handle is
*always unregulated*; when you call a method on it, it always runs
immediately.

When is this useful?  Well, what if you need to change the state of a
primitive while *inside* the scenario?  You might want to tweak a
semaphore in the middle of a test, bumping up its value by calling
`release`. But if you just call `release` on the primitive, it'll be a
regulated call--and meanwhile, you're the guy who's supposed to be
calling into **blanket** and letting these calls make progress. You'd be
deadlocked!

Instead, just use the raw handle:

```
with scenario:
    ...

    # totally fine
    raw = scenario.raw(my_semaphore)
    raw.release()
```

You might also want to give a *worker thread* (or some other piece of
subsystem code) an unregulated handle, even though other code in the same
test is using the regulated handle. Regulation follows the handle, not
the primitive, so different references to the same underlying object can
be regulated independently. Imagine a test that exercises three
subsystems A, B, and C, sharing a lock, and you only want **blanket** to
control synchronization in A and C--maybe B is incidental machinery you
don't care about managing. You can give B a raw handle to the lock; B
will use the lock at full speed, while A and C's calls on the same lock
produce transactions and flow through the scheduler.

Outside the scenario you don't need raws--the regulated handle is already
unregulated. Raws are only for when you're inside the scenario.

### Masquerade

By default, the `repr` on a `scenario.Lock()` looks exactly like
the `repr` on a real `threading.Lock()`. And
`isinstance(scenario.Lock(), threading.Lock)` returns `True`!  We
say that **blanket**'s synchronization primitives *masquerade* as
real primitives. The point is to let the code under test be
completely unaware that anything unusual is going on.  If any
code examines the lock for some reason, it'll think it's a real
`threading.Lock`--it'll never know the difference.

If you set a name on the primitive--via its API object,
`lock_api.name = "..."`--we drop the masquerade, and the
`repr` switches to a nicer "fancy" version.  That gives
you control; if you don't need the masquerade, and it'd be
helpful for your locks to have names for print-style debugging,
you can simply name your locks and get better diagnostics.

There are two ways to tell a **blanket** primitive from
a real one.  First, in the `repr` of every **blanket** primitive,
even when it's masquerading, it uppercases the hexadecimal
`id` at the end.

```
>>> import threading
>>> import blanket
>>> real = threading.Lock()
>>> scenario = blanket.Scenario()
>>> blanket = scenario.Lock()
>>> real
<unlocked _thread.lock object at 0x78c990475650>
>>> blanket
<unlocked _thread.lock object at 0X78C9905B2CF0>
```

See?  In the real lock, the id at the end starts with
`0x`, and the a-z letters are lowercase.  With the
**blanket** lock, the end starts with `0X`, and the a-z
letters are in uppercase.

Second, if you need to tell programmatically, check to see
if the object is an instance of the scenario class:

```
if isinstance(lock, scenario.Lock):
    print("hey!  this is a blanket lock!  cool!")
```


## The Three Layers Of The API

> *Higher level interfaces are implemented using*
> *lower level interfaces.  The user should be able*
> *to reimplement any medium or high level interface*
> *using the tools we give them.*

The **blanket** API is layered.  Each layer is implemented in terms
of the one below it, and the higher layers implement common patterns
and handle the messy details for you.

The three layers are:

- **The low-level API**: transactions, and the universal
  synchronization function `scenario.wait`.
- **The middle-level API**: methods and classes that drive
  threads through sequences of method calls: `scenario.park`,
  `scenario.skip`, `scenario.block`, `scenario.pause`, and the
  `Driver`/`Chain`/`Dispatch` subsystem.
- **The high-level API**: per-primitive helper methods for
  common patterns in multithreaded programming: `assign`, `relay`,
  `cycle`, `allocate`, and `deliver`.

You should spend most of your time at the high level, dropping
down to the middle level occasionally, and reaching into the low level
only for tests that need surgical precision. But it's worth being
familiar with all three.  As the classic computer science aphorism
says: *all abstractions leak*.  So it's helpful to understand all
three levels, even if you mostly stay at the top.


### The Low-Level API

The low-level API has two components, and together they comprise
the foundation **blanket** is built on: transactions,
and `scenario.wait`.  Take either one away and it's impossible
for **blanket** to work.


#### Transactions

Every method call on a **blanket** primitive becomes a *transaction*.
A transaction encapsulates:

- the *primitive* the method call was made on,
- the *method* being called, which is a "bound method object" (`lock.acquire`),
- the *thread* doing the calling,
- the *state* the call is currently in,
- and a few operations the scheduler can perform on it.

While a thread is currently running a transaction, you can get
a handle to that transaction by calling `scenario.transaction(thread)`
or by evaluating `scenario.transactions[thread]`.  Once a transaction
finishes (once it reaches a "terminal state"), it gets unregistered
from these two places.

#### The Transaction Lifecycle

A transaction is a state machine.  Every transaction moves through
a sequence of states, from the moment it's created to the moment it terminates.

In order from first to last, those states are:

- **`BLOCKED`** - the *scheduler block*; entry park, where every
  transaction starts.
- **`COMMIT`** - a parking state for timeout-bearing transactions.
- **`WAITING`** - a parking state for transactions blocked inside
  the underlying real primitive.
- **`STALLED`** - the *scheduler stall*; post-primitive park.
- **`RESUMED`** - a transit state, past `WAITING` (or `COMMIT`).
- **`COMMITTED`** - a transit state; the work has been committed.
- **`PAUSED`** - the *scheduler pause*; general-purpose park,
  applicable at any point.
- **`EXITING`** - a transit state on the way to terminal.
- **`RETURNED`** - terminal; the method returned normally.
- **`RAISED`** - terminal; the method raised an exception.

A transaction *always* progresses strictly *forward* through
these states, from `BLOCKED` to `RETURNED` (or `RAISED`).  It may
skip a state--`PAUSED` only gets visited if someone asks for the
pause--but it never goes backwards.

For the full story on what each state means in detail, including
what each primitive's methods actually do in each state, see the
*Transaction State Reference* near the end of this document.

#### The Four Parking States

Of the states above, four are *parking states*: `BLOCKED`,
`COMMIT`/`WAITING` (one or the other, depending on the transaction
kind), `STALLED`, and `PAUSED`. These are the four places along
the lifecycle where the transaction can come to rest and wait for
the scheduler to release it (with the caveat that `WAITING` is
released by the underlying primitive, not by the scheduler).

Why have four of them? Because the scheduler often wants different
things at different moments. Sometimes you want to hold the call
*before* it's done anything, so you can cause
the threads to call a method in a certain order.  That's what the
scheduler *block* is for.  Sometimes the actual method will park
itself; we report that with the **`COMMIT`** and **`WAITING`** states,
although we can't control those parking states directly.
The scheduler *stall* specifically lets you regulate who acquires
the underlying lock of a `Condition` and when.  And finally,
sometimes you want to hold a call after it's finished, to control
when that thread resumes after a blocking call and goes back to
doing work--that's what the scheduler *pause* is for.

#### Per-Transaction Operations

There are also a bunch of method calls you can make
on a transaction directly.  You'll typically get at these via
`transaction.method`, where `transaction` is the wrapper object
returned to you by the higher-level API:

- `transaction.unblock()` releases the transaction from a scheduler
  block, letting it proceed.  There are also `unpause` and `unstall`
  methods.


There are three functions that only apply to transactions
representing a method that takes a `timeout` argument:

- `transaction.expire()` - force the method call to time out and fail.
- `transaction.disregard()` - tell the transaction "ignore the timeout,"
  causing it to act as if no timeout was specified.
- `transaction.revert()` - restore the original timeout value passed
  in by the user.  Undoes `expire` and `disregard`.


#### scenario.wait

`scenario.wait(*items, timeout=None, all=False)` is the universal blocker. It
blocks the scheduler until any of the *items* you supply *signals*. With
`all=True`, it accumulates signaled items until every item has signaled at
least once, using one shared timeout; if that timeout expires it returns the
partial `frozenset` accumulated so far.
A wide range of objects can be items: bound methods on primitives
(signals while any thread is inside that method), regulated primitives
(signals while any thread is using that primitive), a thread (signals
while the thread has an active transaction), a transaction (signals once
the transaction has completed), and the various signal-token classes
documented below.

We'll see a lot more about `wait` in the *Signals And wait* section
to come.



### The Middle Level

The middle-level API drives threads through sequences of method calls.
There are four scenario methods and three driver classes.

The scenario methods all take a *thread spec* followed by one or more
methods.  A thread spec is normally just a thread:

```
scenario.skip(A, lock.acquire)
```

When you're steering work created inside an existing transaction--for
example, a primitive call made by a `Condition.wait_for` predicate--the
thread spec may instead be a strict two-tuple:

```
scenario.skip((A, base_tx), lock.acquire)
```

The tuple means: look for the named call on thread `A`, but only as a
child, grandchild, etc. of `base_tx`.  The base transaction itself is the
scope; it is not the target.

#### park

`scenario.park(*args)` drives one or more named threads to specified
methods, stopping each one at the scheduler block on the method you
specified.  `park` is *lenient*: it skips unrelated transactions until it
finds the named call.

```
with scenario:
    t = scenario.thread(worker)
    result = scenario.park(t, lock.acquire)
    # t is now parked on lock.acquire.
    # result[t] is the transaction at the scheduler block.
    result[t].unblock()
```

#### skip

`scenario.skip(*args)` drives the named threads *through* one or more
method calls each.  `skip` is *strict*: the named calls must be the next
transactions, in the order you named them.  It auto-skips child
transactions.

```
scenario.skip(t, lock.acquire, lock.release)
# t has now completed both lock.acquire and lock.release.
```

#### block

`scenario.block(*args)` is the strict sibling of `park`.  It expects the
named call to be the next transaction, and leaves that transaction parked
at `BLOCKED`.

```
blocked = scenario.block(t, lock.acquire)
tx = blocked[t]
# tx.state is State.BLOCKED
```

Use `block` when the next call had better be exactly the call you named.
Use `park` when you're willing to drive past unrelated work until the
named call appears.

#### pause

`scenario.pause(*args)` is the strict sibling of `skip` that stops after
the actual method has run.  It drives the named call to `PAUSED`, so the
call has committed but the worker thread has not yet resumed ordinary
Python execution.

```
paused = scenario.pause(t, event.wait)
tx = paused[t]
tx.unpause()           # release your pause request
```

#### Driver

`scenario.Driver`, `scenario.Chain`, and `scenario.Dispatch` are objects
used to drive one or more transactions running in one or more threads.
They give you direct, manual control over the underlying driver state
machine. Most tests don't need them, but the few that do tend to *really*
need them.

A `Driver` attaches to a single worker thread. You construct one with
`scenario.Driver(thread)`, or with `scenario.Driver(thread, base_tx)` to
drive only descendant transactions under `base_tx`. You can also pass
`route=` to give the Driver a little per-driver scheduler. Drivers are
lazy; nothing really happens until you "drive" one, either by calling
the object `driver()`, or by giving it to a `Dispatch` and iterating over
the dispatch.

This is a breaking change in 1.1: a Driver no longer silently looks
for the next transaction on its own.  Even scanning for the current
transaction is explicit: call `driver.scan()`, then drive the Driver.
Nested work is visible too; if a child transaction appears while the
Driver is carrying out a directive, the Driver reports back instead of
silently skipping it.  Most high-level imperatives still opt into
automatic child skipping where that is the right behavior, but code using
`Driver` directly should expect to steer nested work.

A Driver gives you a set of imperatives--`scan()`, `finish()`,
`until(state)`, `block()`, `commit()`, `wait(*signals)`, `reenter()`,
`resume()`, `stall()`, `pause()`, and `route()`--each of which requests
a scan, a state transition, a passive signal wait, a callback edge, or a
route install.
Again, this isn't done eagerly; the `Driver` remembers the request, then
makes it happen the next time it's driven.  If you stage a second
imperative before driving, the second one replaces the first one.
`refresh()` is the exception: it acts immediately, re-baselining the
Driver after deliberate out-of-band driving.

A `Chain` is an ordered sequence of `Driver` objects. Adding a Chain to a
Dispatch promotes the chain's first driver; when that driver reaches its
next ask point, the next driver in the chain takes its place; and so on,
until the chain is empty. You can also iterate over a Chain directly, or
pop drivers off the head manually with `chain.promote()`.

A `Dispatch` is an iterator over drivers that need attention. You add
drivers (or chains of drivers) to it via `dispatch.add`. Each time you
call `next(dispatch)`, it returns whichever driver needs the scheduler's
attention next, having woken via a single `scenario.wait` on the union of
every driver's signals.

The typical pattern looks something like this:

```
with scenario:
    d1 = scenario.Driver(t1)
    d2 = scenario.Driver(t2)
    d1.scan()
    d2.scan()
    dispatch = scenario.Dispatch()
    dispatch.add(d1)
    dispatch.add(d2)
    for d in dispatch:
        # d is a Driver that has completed its scan.
        # Stage the next directive explicitly.
        d.finish()
        d()
```

Note that if you only need to interact with one driver, you can skip the
`Dispatch` object.  Calling the driver object drives it in isolation:

```
with scenario:
    d = scenario.Driver(t1)
    d.scan()
    d()
    d.finish()
    d()
    # d has been driven and you can now inspect it
```

`park`, `skip`, `block`, `pause`, and everything in the high-level API
are all implemented using `Driver` objects.

### The High-Level API

The high-level API is a collection of methods on the per-primitive
*API objects*.  You can get the API object for a particular primitive
via `scenario.api(primitive)`:

```
lock_api = scenario.api(lock)
cond_api = scenario.api(cond)
```

Each API object has helper methods appropriate to its primitive.
They are tailor-made idiomatic shortcuts that handle common
usage patterns with that primitive in multithreaded code.

- **`assign(thread, acquirer=None, *, pause=False)`** -
  available on `Lock` and `RLock` API objects, and on `Condition`
  API objects for the condition's underlying lock. Manages one
  `acquire` call, and maybe one `release` call.  With one
  argument, the lock must not be locked, and that thread
  must call `acquire`, which will succeed.  With two
  arguments, the first thread calls `release`, and the
  second thread calls `acquire`, and is guaranteed to succeed.

- **`relay(initial, *acquirers, pause=False)`** - available on `Lock`
  and `RLock` API objects. Chain acquiring and releasing the
  lock through an ordered sequence of threads. The `initial`
  thread can call either `acquire` or both `acquire` followed
  by `release`; every thread after `initial` but before the last
  one must call `acquire` followed by `release`, and the
  last thread must call `acquire`, at which point `relay` is done.
  Returns an iterator yielding each acquirer thread after
  its `acquire` call succeeds, so you can manage what that
  thread does once it acquires the lock.

- **`cycle(*threads)`** - on `Condition`, `Event`,
  and `Barrier` API objects.  Drives a set of threads through a
  wait/notify cycle; one or more "wait" calls (calling a `wait`
  or `wait_for` method), which sleep until the "notify" call
  (`Condition.notify`, `Condition.notify_all`, `Event.set`,
  or the last `Barrier.wait` call which opens the barrier).
  `cycle` returns an object which should be used as a context
  manager (`with api.cycle(A, B, C):`).  When `cycle` returns,
  all the method calls have been called, and all the waiters
  are waiting for you to take over.  You can call methods
  on the cycle object to wake or pause them in any order
  (`wake(thread, ...)`, `pause(thread, ...)`).  `Condition.cycle` also
  has `wait(thread, ...)`, for driving false-predicate `wait_for`
  wakeups back into the waiting phase.

- **`allocate(*threads, pause=False)`** - on `Semaphore` and
  `BoundedSemaphore` API objects.  Drive a script of semaphore
  `acquire` and `release` calls. `threads` mixes acquirers and
  releasers; **blanket** figures out which is which based on what
  method the thread calls.  Only one semaphore call is actually
  running at a time; the rest stay parked at `BLOCKED`.  If an
  `acquire` has to wait for a later `release`, that's fine.  If
  the script you provide can't make progress, then `allocate`
  blocks too.  Returns an iterator, yielding each acquirer thread
  after its acquire has succeeded.

- **`deliver(*threads)`** - on queue API objects.  Drive a sequence
  of `get`, `put`, `get_nowait`, and `put_nowait` calls on that
  queue.  `deliver` is strict: every named participant's next queue
  call must be one of those four methods on that same queue.  Only
  one queue call is actually running at a time; the rest stay parked
  at `BLOCKED`.  If the script you provide can't make progress,
  then `deliver` blocks too.  Returns the queue transactions in the
  same order as the arguments.


In addition to these tailor-made helpers, the API objects
also provide high-level helpers to manage timeouts:
`expire`, `disregard`, and `revert`.  These are convenience
methods that call the method on the transaction for you.
You pass in the primitive method the thread should be calling,
and the list of threads, and it changes the timeout behavior
for you:

```
lock_api.expire(lock.acquire, t1, t2, t3)
```


## Signals And wait

The `wait` method on a scenario is the universal blocker. It blocks the
scheduler until one of the items you give it *signals*, meaning, the
condition it represents becomes true.  The design for `wait` borrows
heavily from Win32's wonderful
[`WaitForMultipleObjects`](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects)
function, which does basically the same thing.

In 1.1, signals are self-reporting objects.  A signal implements
`sample(scenario)`, which returns whether the signal is currently high.
`scenario.wait` samples its inputs immediately; if nothing is high, it
parks until **blanket** emits a matching wakeup.  **blanket** no longer
keeps a long-lived set of every signal that ever fired, so old
uninteresting signal objects don't stay alive just because they were once
true.

The items you can give to `wait` cover a lot of conditions:

- A **thread** signals while the thread has an active transaction.
- A **transaction** signals once the transaction has completed.
- A **regulated primitive**, e.g. `lock` or `q`, signals while any thread
  has an active transaction using that primitive.
- A **bound method on a primitive**, e.g. `lock.acquire`, signals while
  any thread has an active transaction on that method.
- A **`Call(thread_or_tuple, method, state=None)`** instance signals
  while a particular thread is calling `method`.  `thread_or_tuple` may
  be either a bare thread or `(thread, base_tx)` to restrict the match to
  descendant transactions under `base_tx`.
- A **`Use(thread_or_tuple, primitive)`** instance signals while a
  particular thread is using any method on `primitive`.  It also accepts
  the `(thread, base_tx)` scoped form.
- A **`Terminated(thread)`** instance signals once the thread has
  terminated.
- A **`Not(x)`** instance signals the opposite of `x`.
  `Not(Terminated(x))` signals when the thread has *not* terminated;
  `Not(lock)` signals when no active transaction is using `lock`.
- A **`Nested(transaction)`** instance signals while that transaction has
  a child transaction.
- An **`Action(transaction)`** instance signals while a `Barrier.wait`
  transaction is running its action callback.
- A **`Predicate(transaction)`** instance signals while a
  `Condition.wait_for` transaction is running its predicate callback.
- A **`Reached(transaction, state)`** instance signals while the
  transaction's state is at or past `state`.
- A **`TransactionState`** subclass instance: `Blocked(tx)`,
  `Waiting(tx)`, `Stalled(tx)`, `Resumed(tx)`, `Committed(tx)`,
  `Paused(tx)`, `Exiting(tx)`, `Returned(tx)`, `Raised(tx)`,
  `Commit(tx)`. Signals only while that transaction is in that state.

You can pass in as many of these as you like.  `wait` returns as soon as
*any* of them signals, and it returns a `set()` containing all the
original items that signaled.  So if you want "either thread A terminates
or thread B reaches `WAITING`," you write:

```
signaled = scenario.wait(Terminated(A), Reached(b_tx, State.WAITING))
```

and now you can examine `signaled` to see which one signaled.

(What if you want to wait until *all* the items have signaled? Just call
`wait` multiple times, once for each item.)

`wait` also supports a `timeout` keyword-only parameter, which if
provided is the longest you will wait, specified in seconds:

```
signaled = scenario.wait(t, timeout=5.0)
```

By default `timeout` is `None`, which means "no timeout, wait forever".
If a `wait` call times out, it returns an empty set.

## Other Topics

A handful of small things worth knowing.

### Monkey-Patching

Sometimes the code you want to test does its synchronization work via
`import threading` or `import queue`, and references like
`threading.Lock` or `queue.Queue`.  Or maybe even `from threading import
Lock` and `from queue import Queue`.  You don't own the source, you can't
modify it so you pass in pre-constructed primitives, and you'd rather not
patch each reference by hand.  For that case, the scenario has an
`inject` method.

`scenario.inject(module)` monkey-patches supported `threading` and
`queue` primitive references in `module` so that, while the patch is in
place, calls like `target_module.threading.Lock()` and
`target_module.queue.Queue()` construct **blanket** primitives on the
scenario instead of real stdlib primitives. Example:

```
import target_module

scenario = blanket.Scenario()
with scenario.inject(target_module):
    ...
    # Inside this block, target_module's threading.Lock,
    # threading.Condition, queue.Queue, etc. construct blanket
    # primitives bound to scenario.
    ...
```

It handles two reference patterns:

- Names bound directly to a primitive class, e.g. `from threading import
  Lock`, `from queue import Queue`, or `Mutex = threading.Lock`. Each
  such name is rebound to the corresponding scenario primitive class.
  (This is done by examining the value, not the name; something else that
  happens to be named `Lock` will be left alone.)
- A module attribute whose value is the `threading` or `queue` module
  itself, e.g. `import threading`. That attribute is replaced with a
  small stand-in object whose regulated names are the scenario's
  primitives, and whose other attribute lookups fall through to the real
  module.  So `target_module.threading.Lock()` constructs a **blanket**
  `Lock`, but `target_module.threading.Thread` is still
  `threading.Thread`; similarly, `target_module.queue.Queue()` constructs
  a **blanket** `Queue`, but `target_module.queue.Empty` is still the
  real `queue.Empty`.

The injection is returned as a handle, usable as a context manager (as
above) or closed explicitly with `.close()`. On close, the original
references are restored.

If `scenario.inject(module)` can't find anything to patch, it raises
`ValueError`--almost certainly a sign that you've pointed it at the wrong
module, or the target imports `threading` / `queue` lazily inside a
function (so the import hasn't happened yet at the time inject runs).

### Nested Transactions

Most transactions are roots.  A thread calls `lock.acquire`, **blanket**
creates a transaction for that call, and `scenario.transaction(thread)`
returns that transaction until it exits.  Simple enough.

Nested transactions happen when the method **blanket** is regulating calls
back into user code, and that user code calls another regulated primitive.
There are two important cases:

- `Barrier.wait` can run the barrier's `action` callback.
- `Condition.wait_for` runs the predicate callback, possibly more than once.

If that callback calls a **blanket** primitive, the new transaction is a
child of the callback-running transaction.  You can see this directly:

```
outer = scenario.transaction(thread)
# Later, while the callback is running and has called another primitive:
inner = scenario.transaction(thread)
assert inner.parent is outer
```

The top transaction on the thread is always the child-est one.  So
`scenario.transaction(thread)` returns `inner` in the example above.
The parent chain is how **blanket** remembers that `inner` happened
inside `outer`, not merely later on the same thread.

This matters because sometimes you want to drive the nested call, not the
next ordinary call the thread makes after the outer transaction finishes.
For that, APIs that select a future transaction by thread and method
also accept a strict two-tuple:

```
scenario.skip((thread, outer), lock.acquire)
scenario.block((thread, outer), q.get)
scenario.api(lock).unblock(lock.acquire, (thread, outer))
scenario.api(q).deliver((thread, outer), producer)
```

Read `(thread, outer)` as: look for a matching child, grandchild, etc.
under `outer`.  Do not select `outer` itself.  The tuple is a scope, not
a transaction object.

The same idea exists in signals:

```
scenario.wait(Call((thread, outer), lock.acquire))
scenario.wait(Use((thread, outer), lock))
```

And, if you're down at the Driver level, you can construct the scoped
Driver directly:

```
driver = scenario.Driver(thread, outer)
```

A plain `Driver(thread)` no longer silently skips nested transactions in
1.1.  If it runs into one, it reports back with the child selected as
`driver.tx`; read `driver.tx.parent` when you need to distinguish a child
from a top-level transaction.  A scoped Driver that can't reach nested
work because the base transaction is parked in the wrong place, or because
the base transaction has already gone away, ends in `impasse`.  That's
**blanket** saying: with the scope you gave me, there is no transaction
here that I can drive.

The practical advice: when you're testing callbacks, keep the outer
transaction in a variable, and use `(thread, outer)` whenever you mean
"the thing this callback is about to do."


### Parked Threads At Scenario Exit

When the scheduler exits the scenario, **blanket** does some cleanup
automatically. It happens in this order:

1. Any still-active `Driver` is closed.
2. The scenario flips to unregulated: from this point on, calls
   on the **blanket** primitives behave just like calls on the
   underlying real primitives--they don't park.
3. Every transaction that's currently parked at a blanket-controlled
   parking state (`BLOCKED`, `STALLED`, `PAUSED`) is unparked,
   so its worker can resume, and finish the call natively against
   the now-unregulated primitive.
4. Every managed worker thread is `join()`ed.

The upshot: in the usual case, you don't need to manually drive
parked workers to termination before exiting the scenario. The
exit unparks them and they should finish on their own.

There are still cases where exit can hang. A thread parked at
`WAITING`--asleep inside a real `condition.wait`, `lock.acquire`,
or `barrier.wait`--isn't unparked by **blanket**, because it isn't
blanket's to wake. The OS-level wait will return only when the
underlying primitive's natural wake condition fires (a notify, an
event set, the last barrier party arriving) or its timeout expires.
If nothing in your test ever provides that wake, the join hangs.

If you want explicit control over how a particular thread ends--for
example, driving it through specific final method calls before exit,
or expiring its timeout instead of waiting for it to fire--use the
middle-level APIs (`skip`, `block`, `pause`, `park`) or a `Driver`
directly inside the scenario, before exit. The auto-unpark at exit is
only for the "just let the worker finish on its own" case.

### Setup, Teardown, And Raws

For *setup and teardown* of synchronization state, you almost never
need raw handles--**blanket** primitives are unregulated outside
the `with scenario:` block, so calls on the regulated handle just
pass through:

```
event = scenario.Event()
event.set()                # outside the scenario; passes through

lock = scenario.Lock()
lock.acquire()             # also passes through
```

Raws are for the two cases where regulation would otherwise apply
but you don't want it to: the scheduler making an unregulated call
from inside the scenario (e.g., `scenario.raw(sem).release()`), or
handing an unregulated handle to a worker or subsystem you
specifically don't want to script. Regulation tracks the handle,
not the primitive, so a regulated handle and a raw handle to the
same underlying primitive can coexist in the same test.

### Lazy Imperatives

The `Driver` imperatives (`scan`, `finish`, `until`, `block`, `commit`,
`wait`, `reenter`, `resume`, `stall`, `pause`, and `route`) are *lazy*.
They request a scan, a state transition, a passive signal wait, a callback
edge, or a route install, but they don't fire the underlying work until
the driver is actually driven--by calling `driver()`, or by a `Dispatch`
driving it. A driver carries at most one staged imperative at a time:
calling a second imperative before the first one is driven replaces the
first. The point of laziness isn't to let you stack imperatives; it's to
separate *what should happen next* from *when it happens*, so that
`Dispatch` can be the one to fire the work in coordination with whatever
other drivers are also active.

If you're working at the middle level, this rarely matters--`park`
and `skip` handle the driving for you. But if you reach for explicit
`Driver` / `Chain` / `Dispatch`, knowing the laziness rule will
save you confusion.  `refresh()` is not lazy: it updates the Driver's
model immediately and stages nothing.

## The Injector

There's one more piece to **blanket** that doesn't fit anywhere
in the scenario story: a bytecode injector, in the
`blanket.injector` submodule. This part is small and specialized,
serving a comparatively rare use case--if you need it, you need it,
and if you don't need it you can skip this section entirely.

### What The Injector Is For

The whole **blanket** scenario-based model rests on the assumption
that the code under test uses synchronization primitives. **blanket**
wraps the primitives, regulates the calls, and the scheduler steers
from there.

But what if the code *doesn't* use synchronization primitives?
What if you're some sort of galaxy-brained programming god writing
lockless data structures?  What if you have code that relies on
specific Python operations being atomic at the bytecode level
(like `dict[k] = v` or `list.append`)?  There's nothing for
**blanket** to wrap, nothing for the scheduler to steer.  The
two threads happily race ahead at whatever rate the interpreter
runs them, and the test goes back to being nondeterministic.

That's what the injector is for. It lets you take an existing
Python function and modify it, producing a *new* function that's
identical to the original except: you've inserted a single
function call inside the function's bytecode, at a location you
specify.  You then arrange for that injected call to be a
**blanket** synchronization point--say, a call to 
`event.wait` (on a real `threading` event)--and that gives you
back control over the function making progress.

In short: with the injector, you insert synchronization points
*into* the code under test, giving the scheduler something to
control.

If you need to, you can mix injected code with code
using **blanket** primitives.  The two are orthogonal
techniques and compose perfectly.


### Locations

A `Location` object represents a point in a function's bytecode
where we can inject a call to a callable. You construct one via
one of four classmethods:

- **`Location.position(function, line, column=1)`** - by source
  position. `line` is relative to the start of the function (1-based);
  `column` is 1-based. On Python 3.11+, column is precise; on Python
  3.10 and earlier, only `column=1` is supported.

- **`Location.text(function, text, *, skip=0, after=None)`** - by
  source text match. Finds the first occurrence of `text` in the
  function's source code, optionally skipping `skip` earlier matches,
  optionally requiring the match to come after another `Location`.
  On Python 3.10 and earlier, `text` must match at the start of a
  line (after indentation).

- **`Location.token(function, token, *, skip=0, after=None)`** - by
  Python token. Finds the first occurrence of `token` (a string)
  in the function's tokens. Same `skip` and `after` semantics.

- **`Location.bytecode(function, bytecode, *, skip=0, after=None)`** - by
  bytecode instruction name, e.g. `"RETURN_VALUE"`.  Same `skip` and
  `after` semantics as `Location.text` and `Location.token`.  The escape
  hatch for when source-level locations aren't the right tool.

`Location` objects support equality, hashing, rich comparison
(`<`, `<=`, etc.) when they're from the same function,
and a useful `repr`.


### inject_call

Once you have a `Location`, you can inject a call using
`inject_call`:

```
from blanket.injector import Location, inject_call

def target(x):
    y = x * 2
    return y + 1

def callback():
    print("hello from the injection point")

loc = Location.text(target, 'y = x * 2')
new_target = inject_call(callback, loc)
new_target(5)
# prints "hello from the injection point", returns 11
```

`inject_call` doesn't modify the original function. It builds a
new function object with the same code, plus the inserted call at
the location you specified. The injected callable is bound into
the new function's globals under its `__name__` (with collision
resolution); you can override the name via `name=`.

This works on functions and methods!  It's up to you what to
do about methods on a class; you can create a subclass where
you replace a method with the injected version, or you can
overwrite the original by just setting the attribute on the
class.  Up to you.


## Tools Similar To blanket

To my knowledge, there isn't anything else that does quite what
**blanket** does. There is a substantial and growing body of academic
and industrial work on the problem of testing concurrent code, and
some of it is pretty similar to **blanket**.

The biggest related field is called *stateless model checking*,
or SMC. Tools in this area include Microsoft's CHESS,
AWS's Shuttle, Rust's Loom, GenMC, Nidhugg, and many others; the
field has been an active research area for two decades and counting.
The SMC approach is to *automate* the exploration of thread
interleavings: the tool runs your test repeatedly, each time making
different scheduling choices, with the goal of systematically
(or stochastically!)
covering as many distinct interleavings as it can. Modern SMC
tools use sophisticated techniques like *dynamic partial-order
reduction* to prune redundant orderings, and probabilistic strategies
like the *Probabilistic Concurrency Testing* (PCT) algorithm to
bias the search toward likely bugs. The trade-off is that SMC tools
generally re-implement the synchronization primitives themselves,
so they can inspect and control execution at a fine grain. That
means your program is being tested against the model checker's
reimplementation of the primitive, rather than the real primitive.

Microsoft's *Coyote* sits a little closer to **blanket** in spirit,
and is worth calling out. Coyote is an SMC tool for .NET programs
that, in addition to exploring interleavings, *records* the
sequence of scheduling decisions that led to any bug it finds. The
recording can then be replayed to reproduce the bug deterministically.
In effect, Coyote automates the production of something resembling
a **blanket** scheduler script. As far as I can tell, though, the
recordings are machine-generated and machine-replayed--they don't
appear to be designed to be written or edited by hand the way a
**blanket** script is.

But what really sets **blanket** apart from these other tools is intent.
The other tools are aimed primarily at *discovering* concurrency bugs.
**blanket** is designed for *recreating* known concurrency
scenarios--declaratively, by hand, in code you write yourself.
As far as I know, that's something new.

Could you use **blanket** for SMC?  I think you could!
But **blanket** isn't optimized for raw speed, and SMC tools
probably want something faster.  Alternatively, a hypothetical
SMC for Python could produce **blanket** scheduler code as output,
akin to the Coyote recording file: once it discovered a bug, it
could hypothetically write a **blanket** scheduler script that
reproduces the bug, and you could copy that script into your
regression suite.

## Under The Hood

This section is for the curious. None of it is required reading
to use **blanket**, but if you want to understand *why* the API
is shaped the way it is, here's what's actually going on inside.

### Core Objects

A **blanket** primitive isn't actually one object--it's four.
When you call `scenario.Lock()`, you get back a *primitive handle*
(masquerading as a real `threading.Lock`).  Alongside it, the
scenario builds an *API object* (which you get via `scenario.api(lock)`)
and a *raw handle* (`scenario.raw(lock)`).  These three user-facing
objects are actually thin wrappers around a fourth internal-only
object we call a *core*, in this case a `LockCore`.

The core is where the actual work happens.  Each wrapper's methods
do essentially the same dance: acquire a lock, possibly box or
unbox a few arguments, call the corresponding core method, then
release the lock and return.  The core does the real work
underneath: bookkeeping the transactions, signalling state changes,
manipulating the underlying real primitive.

Why three wrappers around each core?  Each one represents a different
interaction posture.  The primitive handle masquerades as a real
`threading.Lock` so the code under test doesn't know it's been
swapped in.  The API object is the scheduler's surface, with methods
like `assign`, `relay`, `unblock`.  The raw handle is the unregulated
escape hatch.  All three point at the same core, and all three funnel
their work through it.

The scenario object itself follows the same pattern.  The user-facing
`Scenario` is a wrapper; its core is internally called *score*--an
abbreviation of *scenario core*.  That naming convention shows up
in a few places in the docs, the source, and stack traces.

(You never need to think about cores when writing **blanket** tests.
The wrappers cover everything.  But if you ever see `LockCore.acquire`
or `score.transactions` in a stack trace, you now know what's going
on.)

### score.lock

The lock the wrappers all acquire is `score.lock`, owned by the
scenario core.  There's exactly one of them.  Every regulated
operation in **blanket**--every method call on a primitive, every
API-object method, every Driver/Chain/Dispatch operation, every
scheduler-side manipulation of a transaction--enters under
`score.lock`.

That sounds slow.  In practice it isn't, because the lock is only
lightly contended: the scheduler thread does its work, then releases
the lock and waits for something to happen; worker threads grab
the lock momentarily as they transit through their transaction
states and then either release it and proceed or release it and
park.  Nobody holds it for long.

The trade-off is deliberate: **blanket** trades a little performance
for a lot of safety and determinism.  One lock means one consistent
view of the world, no inter-component races inside **blanket**
itself, and a much simpler invariant story for the implementation.
(For perspective: CPython runs the entire interpreter under a
single lock--the GIL--and Python isn't *that* slow.)

### The Scheduler Block And Pause

The core trick is straightforward: every regulated method call
starts by acquiring `score.lock` and consulting the scenario about
whether to proceed. The default answer is no.

When a worker calls `lock.acquire()`, **blanket** doesn't immediately
call the real `lock.acquire()`. Instead, it builds a transaction
object (state `BLOCKED`), parks the worker on it, and signals the
scheduler that a new transaction exists. The worker is now asleep
inside **blanket**, holding zero locks, holding none of the underlying
primitive's state. The scheduler is free to look at the transaction,
inspect what method it's on, see what other transactions exist, and
decide what to do.

When the scheduler decides "yes, go", it calls `transaction.unblock()`.
The transaction transitions out of `BLOCKED`, and the worker wakes
up. The real `lock.acquire()` is invoked. If it returns immediately
(uncontended), the transaction proceeds through `COMMITTED` and
`EXITING` to `RETURNED`. If it would have blocked (contended), the
transaction transitions through `WAITING`--the worker really is
asleep inside the real `threading.Lock`'s acquire--until the lock
becomes available, then `RESUMED` and so on.

The same pattern applies to every regulated method. The work always
happens in the real underlying primitive; **blanket** just decides
when the worker is allowed to attempt the work.

This is why we say **blanket** wraps the real primitives rather than
replacing them: the semantics of `Lock.acquire()` come straight
from `threading.Lock`. We don't reimplement any of it. We just
add gates.


### A Walk Through Condition.wait

Let's walk through a `Condition.wait()` end-to-end, because it's the
most interesting of the lifecycle paths.

Thread T calls `cond.wait()` on a **blanket** `Condition`. **blanket**
constructs a transaction in `BLOCKED`. T is asleep. Scheduler wakes,
sees the transaction, calls `unblock`. The transaction transitions
to `COMMIT`--it's a `TimeoutTransaction` (because `Condition.wait`
accepts a timeout), so it parks here briefly. The scheduler can
choose to expire/disregard at this point; assume it just unblocks.

The transaction transitions onward to `WAITING`. Internally, the
real `threading.Condition.wait()` is called. T is now asleep
*inside* the real condition variable, which has released the
underlying lock and is genuinely blocked on a wait. The scheduler
can observe that the transaction is in `WAITING`, but it can't
directly wake it up--only a `notify()` (or a timeout) can do that.

Meanwhile, thread S calls `cond.notify()` on the same condition.
**blanket** constructs a notify transaction, drives it through the
scheduler block, the notify executes, and T's wait wakes up.

T's wait returns inside the real condition variable. T now needs
to re-acquire the underlying lock to honor `Condition.wait`'s
contract. This re-acquire is itself a synchronization event, so
the transaction enters `STALLED` while **blanket** decides whether
to let the re-acquire happen. (This is the scheduler stall in action:
T has come out of its primitive-side wait, but hasn't committed
its post-wake work, and the scheduler can intervene here.)

Scheduler unblocks the stall. The re-acquire happens. The
transaction transitions through `COMMITTED`, `EXITING`, and finally
`RETURNED`. T is now past `cond.wait()` and continues with whatever
came next.

That's the full path. Notice how at every park, the scheduler can
observe and intervene; and notice how the "real work" is always
done by the underlying `threading.Condition`.

### Why The Driver Is Lazy

The `Driver` state machine stages a single pending imperative
rather than firing it eagerly.  If you stage another imperative before
firing the first one, the later imperative wins.  The point isn't to
batch work; it's to delay setting state on the Driver and the
transaction until we're actively driving them.

Making the `Driver` lazy was important to making `Chain` useful.
If you use `Chain` to drive a number of threads serially,
you can't eagerly start setting states on the waiting threads
or unparking them.  If you unpark them, they'll start making
progress immediately--but the point of using a `Chain` is to
force those threads to make progress serially.  Those subsequent
threads must wait patiently until it's their turn!

This is also why the high-level API methods that return iterators
(`relay`, `allocate`, `cycle`) *are* iterators: each yield is a
natural point to fire one staged step and pause, to let the scheduler
do whatever it needs to do before continuing on to the next thread.

### Faithful Semantics, By Construction

A theme worth restating: **blanket** has no opinion about what
synchronization primitives *mean*. It does no reimplementation.
Every `lock.acquire()` is a real `threading.Lock.acquire()` underneath.
Every `condition.wait()` is a real `threading.Condition.wait()`.
Every `barrier.wait()` is a real `threading.Barrier.wait()`.

This is by design. The point of **blanket** is to give your tests
*reliable behavior*, and that behavior should be faithful to the
real primitives--because the goal is to test the code under test,
not a model of it. If `Condition.wait` has some subtle corner case,
**blanket** will exhibit that subtle corner case, because **blanket**
is using the same `Condition.wait`.

## API Reference

### Module-level

**`blanket.Scenario`**

The scenario class. See the *Scenario* section below for the full
surface.

**`blanket.ThreadOrderingError(ValueError)`**

Raised when **blanket** observes a thread ordering that violates
the script--e.g. a thread completes a method call that the
scheduler script said should be parked.

**`blanket.CompetingDriversError(ValueError)`**

Raised when a thread already has an active `Driver` and another
Driver tries to drive it.  Only one Driver can be active on a
thread at a time; the conflict is detected on first use rather
than at construction.

**`blanket.State`**

The sentinel class for transaction states. Has the following
class-level constants, one per transaction state:

```
State.BLOCKED
State.COMMIT
State.WAITING
State.STALLED
State.RESUMED
State.COMMITTED
State.PAUSED
State.EXITING
State.RETURNED
State.RAISED
```

Each is comparable (`<`, `<=`, etc.) by lifecycle order. Each has
a `.name` (the string `"BLOCKED"`, etc.) and `.index` (the
numerical lifecycle position).

`State.terminal_states` is the frozenset `{State.RETURNED, State.RAISED}`.

**`blanket.Signaling`**

The marker base class for explicit signal objects.  `scenario.wait` also
accepts public shorthand forms--bare threads, bare regulated primitives,
and bare bound methods on regulated primitives--and boxes them internally
before waiting.

**`blanket.Reached(transaction, state)`**

Signal token. Signals while `transaction`'s state is at or past `state`.
Useful for "wait until the transaction has reached at least this point."

**`blanket.Call(thread_or_tuple, method, state=None)`**

Signal token. Signals while a particular thread is in a transaction on
`method`.  `thread_or_tuple` may be either a bare thread or the strict
two-tuple `(thread, base_tx)`, which scopes the match to descendant
transactions underneath `base_tx`.  The base transaction itself is not the
target.

**`blanket.Use(thread_or_tuple, primitive)`**

Signal token. Signals while a particular thread has any transaction on
`primitive` in its call chain.  It supports the same `(thread, base_tx)`
scoped form as `Call`.  ("Use" is the noun form here--it rhymes with
"moose", not "booze".)

**`blanket.Not(token)`**

Signal token. Signals while `token` is *not* signaling.  `token` may be
another explicit signal object, or one of the shorthand items accepted by
`scenario.wait`, such as a thread, a primitive, or a bound primitive
method.

**`blanket.Terminated(thread)`**

Signal token. Signals once `thread` has terminated.

**`blanket.Nested(transaction)`**

Signal token. Signals while `transaction` has a child transaction in
flight.

**`blanket.Action(transaction)`**

Signal token. Signals while a `Barrier.wait` transaction is running its
user-supplied action callback.

**`blanket.Predicate(transaction)`**

Signal token. Signals while a `Condition.wait_for` transaction is running
its predicate callback.

**`blanket.TransactionState(transaction, state)`**

Signal token base class for "transaction is exactly in this state."
Has the following subclasses, one per concrete transaction state:

```
Blocked(tx)
Commit(tx)
Waiting(tx)
Stalled(tx)
Resumed(tx)
Committed(tx)
Paused(tx)
Exiting(tx)
Returned(tx)
Raised(tx)
```

Each signals while `tx.state is` the corresponding state.

**`blanket.TimeoutState`**

A tuple-subclass `(value, time, timed_out)` describing the current
timeout state of a transaction: the user's specified timeout value
(or the synthetic value derived from an `expire` or `disregard`),
the deadline-time it computes to, and whether the timeout has
fired. Returned by various transaction APIs.

**`Scenario.Transaction`** / **`scenario.Transaction`**

The class of the transaction-wrapper objects returned by the
scheduler-facing API (e.g. `scenario.transaction(t)`). Useful
for `isinstance` checks. The methods on transaction objects are
documented in the *TransactionAPI* subsection of `Scenario` below.

### Scenario

**`Scenario()`**

Construct a new scenario.

**`scenario.name`**

The scenario's name (a string), used in `repr()`. Settable.

**`scenario.threading`**

A scenario-bound impersonator for the `threading` module.  Regulated
primitive names are replaced with this scenario's primitives; everything
else falls through to the real module.

**`scenario.queue`**

A scenario-bound impersonator for the `queue` module.  Queue classes
available in the running Python's stdlib are replaced with this
scenario's regulated queue classes; everything else, such as `Empty`,
`Full`, and `ShutDown` where present, falls through to the real module.

**`scenario.reset()`**

Clear the completed-transaction log.  Structural state is left alone:
registered primitives, raw handles, managed threads, and live waiters are
not reset.  Called automatically on scenario entry.

**`scenario.apis`**

Read-only mapping from primitive to API object.

**`scenario.api(primitive)`**

Equivalent to `scenario.apis[primitive]`.

**`scenario.raws`**

Read-only mapping from primitive to raw handle.

**`scenario.raw(primitive)`**

Equivalent to `scenario.raws[primitive]`.

**`scenario.log`**

A read-only list-like view of completed transactions, in completion
order.  It supports `clear()` but not list mutation methods such as
`append` or item assignment.

**`scenario.managed`**

A read-only set-like view of registered worker threads.

**`scenario.thread(target, *args, **kwargs)`**

Create and register a managed worker thread. If the scenario has
been entered, the thread starts immediately; if not, the thread
is registered and starts when the scenario is entered. Returns
a `threading.Thread`.

**`scenario.transactions`**

Read-only mapping from thread to current transaction.

**`scenario.transaction(thread)`**

Equivalent to `scenario.transactions.get(thread)`. Returns `None`
if the thread has no active transaction.

**`scenario.wait(*items, timeout=None, all=False)`**

Block until any of `items` signals. See the *Signals And wait* section
for the supported item types. If `timeout` expires, returns an empty set.
With `all=True`, wait until every item has signaled at least once, using a
single shared timeout; timeout returns the partial `frozenset`.

**`scenario.park(*args)`**

Leniently drive named threads to specified methods, parking each at the
scheduler block. Arguments come in `(thread_spec, method)` pairs, where a
thread spec is either a thread or `(thread, base_tx)`:

```
scenario.park(A, lock.acquire, B, lock.release)
scenario.park((A, parent_tx), lock.locked)
```

Each thread may appear at most once. Returns a dict mapping thread to the
transaction at the scheduler block.

**`scenario.skip(*args)`**

Strictly drive named threads *through* specified methods. Arguments are
flat: thread spec, then one or more methods for that thread, then
optionally another thread spec, etc.:

```
scenario.skip(A, lock.acquire, lock.release, B, lock.acquire)
scenario.skip((A, parent_tx), lock.locked)
```

Returns a dict mapping thread to the last transaction.

**`scenario.block(*args)`**

Strictly drive named threads to specified methods and leave each matching
transaction parked at `BLOCKED`.

**`scenario.pause(*args)`**

Strictly drive named threads through specified methods and leave each
matching transaction parked at `PAUSED`.

**`scenario.__enter__()` / `scenario.__exit__(...)`**

Enter and exit the scenario context. Inside the context, the calling
thread takes the role of the scheduler.

On exit, in order: any still-active `Driver` is closed; the scenario
flips to unregulated (subsequent calls on the primitives pass straight
through to the underlying real primitives); every transaction currently
parked at a blanket-controlled park (`BLOCKED`, `STALLED`, `PAUSED`) is
released, so its worker can resume and finish natively; every managed
worker thread is `join()`ed.  Completed transactions remain in the log
for post-mortem inspection until the next scenario entry or explicit
`reset()`.

### The Primitives

Each is constructed as a method on the scenario, e.g. `scenario.Lock()`.
All faithfully implement the public surface of the corresponding stdlib
type on the Python version you're running.

Threading primitives:

**`scenario.Lock()`** - `acquire(blocking=True, timeout=-1)`,
`release()`, `locked()`, `__enter__`/`__exit__`.

**`scenario.RLock()`** - `acquire(blocking=True, timeout=-1)`,
`release()`, `locked()` where supported, `__enter__`/`__exit__`.

**`scenario.Condition(lock=None)`** - `acquire(...)`, `release()`,
`locked()` where supported, `wait(timeout=None)`,
`wait_for(predicate, timeout=None)`, `notify(n=1)`, `notify_all()`,
`__enter__`/`__exit__`.

**`scenario.Semaphore(value=1)`** - `acquire(blocking=True, timeout=None)`,
`release(n=1)` on Python versions whose stdlib supports `n` and
`release()` on older versions, `__enter__`/`__exit__`.

**`scenario.BoundedSemaphore(value=1)`** - same as Semaphore; `release`
raises if it would exceed initial value.

**`scenario.Event()`** - `is_set()`, `set()`, `clear()`,
`wait(timeout=None)`.

**`scenario.Barrier(parties, action=None, timeout=None)`** -
`wait(timeout=None)`, `reset()`, `abort()`, plus the `parties`,
`n_waiting`, `broken` properties.  If you provide `action`, **blanket**
calls it with the opener's `Barrier.wait` transaction API object.

Queue primitives:

**`scenario.SimpleQueue()`** - where supported by the stdlib:
`put(item, block=True, timeout=None)`, `put_nowait(item)`,
`get(block=True, timeout=None)`, `get_nowait()`, `qsize()`, `empty()`.
`SimpleQueue` is implemented in C in the stdlib, so **blanket** cannot
observe its internal wait state; it regulates the method calls but does
not expose the richer `WAITING` / `STALLED` transitions available on the
Python-level queues.

**`scenario.Queue(maxsize=0)`**, **`scenario.LifoQueue(maxsize=0)`**,
and **`scenario.PriorityQueue(maxsize=0)`** - `put`, `put_nowait`,
`get`, `get_nowait`, `task_done`, `join`, `qsize`, `empty`, `full`, and
`shutdown(immediate=False)` where supported by the stdlib.

In addition, every primitive has a `name` property (settable).

### Per-Primitive API Objects

Each primitive has a corresponding API object available via
`scenario.api(primitive)`. The API object is the scheduler-facing side
of the primitive.  It has a better `repr`, a settable `name`, a `raw`
property for the raw handle, transaction lookup helpers, and a handful
of methods for driving transactions on that primitive.

Every API object has:

- **`api.name`** - the primitive's diagnostic name.  Setting this also
  switches the primitive's `repr` from stdlib masquerade mode to
  **blanket**'s fancy named `repr`.
- **`api.raw`** - the raw, unregulated handle for the primitive.
- **`api.transactions`** - read-only mapping from thread to current
  transaction on this primitive.  This is a plain mapping keyed by bare
  thread; it deliberately does not accept `(thread, base_tx)` tuples.
- **`api.transaction(thread)`** - equivalent to
  `api.transactions.get(thread)`. Returns `None` if the thread has no
  current transaction on this primitive.
- **`api.unblock(method, *thread_specs, pause=False)`** - unblock the
  named threads' transactions on `method`. `method` is the bound method
  on the primitive.
- **`api.unpause(method, *thread_specs)`** - clear the scheduler pause bit
  on the named threads' transactions.  If **blanket** is not also holding
  the transaction at `PAUSED`, it unparks.

Every current primitive API object also has timeout helpers.  They only
make sense for timeout-bearing methods, and the transaction itself raises
if you ask for timeout surgery on a method that can't time out.

- **`api.expire(method, *thread_specs)`** - expire the named threads'
  transactions on `method`.
- **`api.disregard(method, *thread_specs)`** - disregard the named
  threads' timeouts on `method`.
- **`api.revert(method, *thread_specs)`** - undo any prior `expire` or
  `disregard`, restoring the user's original timeout. Operates on
  transactions in `BLOCKED`.

A *thread spec* is either a bare thread or the strict two-tuple
`(thread, base_tx)`.  The tuple scopes that participant to descendant
transactions under `base_tx`; the base transaction itself is not driven.
This thread-spec grammar is accepted by the API methods above and by the
higher-level helpers below.

`Condition` API objects also have:

- **`api.unstall(method, *thread_specs)`** - release the named threads'
  transactions on `method` from the `STALLED` park.  This is mainly used
  after a notify on a `Condition.wait` transaction that's stalled
  mid-commit, to let the worker proceed into the internal lock
  re-acquire.
- **`api.assign(thread_spec, acquirer=None, *, pause=False)`** - the same
  high-level lock assignment helper exposed by `Lock` and `RLock`,
  applied to the condition's underlying lock.

Lock and RLock API objects also have:

- **`api.assign(thread_spec, acquirer=None, *, pause=False)`** - assign
  the lock to `thread_spec`. With one argument, that participant simply
  acquires. With two arguments, the first participant releases and the
  second acquires.
- **`api.relay(initial, *acquirers, pause=False)`** - chain the lock
  through the named participants. `initial` may be either the current
  holder (parked at release/BLOCKED) or an acquirer on an unheld lock
  (parked at acquire/BLOCKED); each `acquirer` takes the lock in turn
  after `initial`. Returns an iterator yielding each acquirer as it takes
  the lock.

Semaphore and BoundedSemaphore API objects also have:

- **`api.allocate(*thread_specs, pause=False)`** - drive the named
  participants through semaphore traffic. Each participant's next
  semaphore call must be `acquire` or `release` on that semaphore.
  `allocate` runs only one of those calls at a time; the others remain
  at `BLOCKED` until chosen. Duplicate thread specs are allowed; a
  thread's later occurrence is not surfaced until the earlier occurrence
  has finished its semaphore transaction. The returned iterator yields
  each acquirer thread after its acquire has succeeded.

Queue, LifoQueue, PriorityQueue, and SimpleQueue API objects also have:

- **`api.deliver(*thread_specs)`** - drive the named participants through
  queue traffic. Each participant's next queue call must be `get`, `put`,
  `get_nowait`, or `put_nowait` on that queue.  `deliver` runs only one
  of those calls at a time; the others remain at `BLOCKED` until chosen.
  Duplicate thread specs are allowed; a thread's later occurrence is not
  surfaced until the earlier occurrence has finished its queue
  transaction. Returns a tuple of transaction API objects, in the same
  order as the arguments.

Condition, Event, and Barrier API objects also have:

- **`api.cycle(*thread_specs)`** - construct a `Cycle` over the named
  participants.  All participants except the last should be making
  `wait` or `wait_for` calls.  The last participant is the "opener", and
  should be calling some sort of *notify* function: `Condition.notify`,
  `Condition.notify_all`, `Event.set`, or in the case of `Barrier` it
  should be the last `waiter`, which opens the barrier.

  The cycle is a context manager and exposes `wake`, `pause`, `iter`,
  `close`, and is callable for wake-and-close shorthand.  `Condition`
  cycles also expose `wait`, which drives selected `wait_for` waiters
  back into the waiting phase after a false predicate.

  `Barrier.cycle` and `Condition.cycle` accept a keyword-only
  `scheduler` callback in the cases where user callbacks can reenter
  **blanket**: a barrier action or a `Condition.wait_for` predicate.
  The callback is called as `scheduler(tx)`, where `tx` is the
  `Barrier.wait` or `Condition.wait_for` transaction running the callback.
  Use `tx.thread` to disambiguate which waiter/action is asking for
  scheduler help.

### TransactionAPI

The wrapper object the scheduler-facing API hands you for individual
transactions.

Properties:

- **`tx.method`** - the bound method this transaction is on.
- **`tx.thread`** - the thread this transaction is on.
- **`tx.state`** - the current state (a `State` constant).
- **`tx.done`** - `True` if the transaction has terminated.
- **`tx.kwargs`** - read-only proxy for the call's keyword arguments.
- **`tx.start_time`** - the time the transaction was constructed.
- **`tx.end_time`** - the time the transaction terminated (`None`
  until terminal).
- **`tx.result`** - the value returned by the actual method,
  or, the exception raised by the actual method if it raised.
  (`None` until terminal.)
- **`tx.succeeded`** - `True` if the transaction "succeeded",
  which is defined as "returned a value and did not indicate it
  timed out".
  `False` if it raised or timed out, `None` while not yet terminal.
- **`tx.failed`** - The opposite of `succeeded`. (and `None` if
  `succeeded` is `None`.)
- **`tx.pause`** - read/write boolean.  This is the scheduler-owned pause
  request.  Set it to `True` to ask **blanket** to hold the transaction
  at `PAUSED`; set it to `False` to withdraw that request.
- **`tx.paused`** - read-only boolean. `True` iff either you or
  **blanket** is currently asking the transaction to stay at `PAUSED`.
- **`tx.parent`** - the parent transaction, if any.  Only used
  for nested transactions, such as the `Condition.wait` inside
  a `Condition.wait_for`).  Usually `None` indicating no parent.
- **`tx.depth`** - a count of how many transactions this transaction
  is nested inside, usually 0.
- **`tx.log`** - tuple of `(time, state)` entries recording the
  transaction's state-transition history. Useful for retrospective
  queries like "did this transaction visit `PAUSED`? and if so, when?"
- **`tx.timeout`** - a `TimeoutState` describing whether a timeout
  was specified and its current status.  `None` if there was no timeout.
  Only defined on transactions that can time out (the method has
  a `timeout` parameter).

Methods:

- **`tx.visited(*states)`** - return `True` if the transaction has
  visited every state listed.  This is a convenience query over `tx.log`;
  it does not impose an ordering requirement.  Calling it with no states
  raises `ValueError`.
- **`tx.wait(state=None)`** - block the scheduler until the transaction
  reaches `state`, or until it terminates if `state` is omitted. Returns
  the transaction's current state.
- **`tx.unblock()`** - unblock the transaction from the scheduler
  block.
- **`tx.unpause()`** - equivalent to setting `tx.pause = False`, then
  asserting that the transaction is no longer paused.  If **blanket** is
  still holding the transaction at `PAUSED`, this raises.
- **`tx.unstall()`** - unblock the transaction from a stall.
- **`tx.expire()`** - force the transaction to time out, when it runs.
  Can only be called on transactions that can time out,
  while the transaction is in `BLOCKED` state.
- **`tx.disregard()`** - force the transaction to never expire.
  Can only be called on transactions that can time out,
  while the transaction is in `BLOCKED` state.
- **`tx.revert()`** - reset the timeout to what the user specified,
  overriding an `expire` or `disregard` call.
  Can only be called on transactions that can time out,
  while the transaction is in `BLOCKED` state.

### Scenario.Driver

**`scenario.Driver(thread, base_tx=None, *, route=None)`**

Construct a Driver attached to `thread`.  If `base_tx` is provided, the
Driver is scoped to descendant transactions under that base transaction.
If `route` is provided, it must be callable, accept the Driver, and
return an iterator; the route steers the Driver until the route returns.
Most routes are generator functions, but a stateful callable object that
returns itself and implements `__next__` is also supported.
A Driver "drives" a thread, which is to say, it causes method calls made
on primitives by the thread to make progress. You can tell the Driver
what you want the thread, or the tx running on the thread, to do, and the
Driver will make it happen and report back when it's successful--or if
some unexpected thing happened and it can no longer make progress on your
request.

Drivers are lazy and acquire their active driving slot only when they are
actually driven.  Constructing more than one Driver for a thread is okay;
trying to drive the same thread with two active Drivers at the same time
raises `CompetingDriversError`.

Breaking change in 1.1: Driver no longer does anything on its own.
A fresh Driver starts in `undirected`; call `scan()` and then drive it to
select the thread's current published transaction.  Driver also no longer
silently skips nested transactions by default.  If a child transaction
appears while Driver is carrying out a directive, it reports back with
that child selected as `driver.tx`.  The high-level APIs still drive
through child transactions where their public contract requires it.

Properties:

- **`driver.thread`** - the thread.
- **`driver.base_tx`** - the base transaction, or `None`.
- **`driver.state`** - the driver state.  A fresh Driver starts in
  `undirected`.
- **`driver.tx`** - the current transaction (`None` if none).
- **`driver.txs`** - tuple of all transactions seen so far.
- **`driver.status`** - a `DriverStatus` snapshot describing the most
  recent completed Driver directive, or `None` if no directive has
  completed yet or the Driver is currently driving.
- **`driver.log`** - tuple of `DriverStatus` snapshots, one for each
  completed Driver directive, including directives consumed internally
  by a route.
- **`driver.waited`** - a `frozenset` containing the final set of
  signals the Driver waited on during the most recent completed drive.
- **`driver.signaled`** - a `frozenset` containing the signals from
  `driver.waited` that were high and woke the Driver.
- **`driver.motivation`** - a `frozenset` containing the signals that
  explain the Driver's new state.  These sets always satisfy
  `driver.motivation <= driver.signaled <= driver.waited`.  For
  `driver.wait(*signals)`, success means `driver.motivation` is the
  subset of the asserted signals that actually signaled; intrinsic
  thread termination means it is `{Terminated(driver.thread)}`.

`DriverStatus` is an immutable tuple-style record with these fields:
`directive`, `directive_args`, `state`, `tx`, `waited`, `signaled`, and
`motivation`.  Its signal sets have the same frozenset invariant:
`motivation <= signaled <= waited`.

- **`driver.directive`** - the most recently completed Driver method, as a
  bound method, or `None` if `driver.status` is `None`.
- **`driver.directive_args`** - the positional arguments from the most
  recently completed directive, as a tuple.
- **`driver.routed`** - `True` while a route is installed and still
  active.
- **`driver.done`** - `True` if in a terminal driver state.  Only
  `terminated` is terminal, so in practice this means "the thread is gone."
  A `raised` Driver is *not* done -- it's recoverable via `scan()`.

Driver has these states:

- `undirected`, no directive is staged.  This is the initial state, and
  also the state after trying to drive with an empty directive slot.
- `driving`, a directive is currently being executed.  You normally won't
  see this state from scheduler code, because the worker thread has
  control while the Driver is driving.
- `success`, the last committed directive reached its target.  This covers
  a successful scan, park, finish, `until`, callback edge, and signal wait.
- `nested`, a child transaction surfaced while the Driver was carrying out
  a non-scan directive.  The child is selected as `driver.tx`; tell the
  Driver what to do with it, then scan back to the parent as needed.
- `returned`, `until(raised)` expected the transaction to raise, but it
  returned cleanly instead.
- `overshot`, the transaction self-progressed past an externally-controlled
  target (`COMMIT` or `WAITING`) during the window since the Driver last
  looked.
- `persisted`, `until(terminated)` expected the thread to stop transacting,
  but the thread began another transaction instead.
- `mutated`, another actor moved a thread past a blanket-controlled point
  this Driver had parked it at.  Recover by scanning again.
- `raised`, the driven transaction transitioned to `RAISED` -- an
  *unanticipated* raise (an expected one, via `until(raised)`, lands in
  `success`).  This is **not** a terminal state: an uncaught exception would
  have killed the thread (landing in `terminated`), so reaching `raised`
  means the worker raised but is still alive and may catch it and carry on.
  Recover by scanning again, exactly like `mutated`; the raise is surfaced
  in the `DriverStatus` without bricking the Driver.
- `terminated`, the thread terminated.  This is the one true terminal state
  -- the only one the Driver can never transition out of, because there is
  no longer a thread to produce transactions to drive.
- `impasse`, a scoped Driver can't currently reach nested work because the
  base transaction is parked in a blanket-controlled state or has gone away.

Driver also publishes `driving_states` and `terminal_states` as
`frozenset` objects containing the public driving and dead-end states.

Driver supports several "imperatives"; these are instructions for what
you want the driver to accomplish when driving the thread.  Note that
these simply stage a directive; Driver doesn't change any state on a
transaction until you let it start driving.  There is one pending slot,
and the last staged imperative wins:

- **`driver.scan(base_tx=None)`** - find a published transaction.  With no
  argument, scan under the Driver's scope floor; with a transaction, scan
  for a child of that transaction.  This is how a fresh Driver starts doing
  useful work.  After driving a transaction to completion, call `scan()`
  again to pick up the next one.
- **`driver.finish()`** - drive the current transaction to a terminal
  transaction state.
- **`driver.until(state)`** - drive toward one of `driver.terminated`,
  `driver.raised`, or `driver.impasse`, and report `success` if that
  requested state is reached.
- **`driver.block()`** - park the worker at the scheduler block, without
  unblocking.
- **`driver.commit()`** - drive the current transaction to `COMMIT`
  (timeout-bearing only).
- **`driver.wait(*signals)`** - passively wait until one of the named
  signals fires. The signal set is normalized the same way `scenario.wait`
  normalizes signals. `wait` does not drive the worker and does not need a
  current transaction; its value over `scenario.wait` is that a waiting
  Driver can participate in a `Dispatch` while other Drivers keep moving.
  After it returns, inspect `driver.waited`, `driver.signaled`, and
  `driver.motivation`.
- **`driver.reenter()`** - drive until the current transaction enters a
  user callback that can itself ask **blanket** for scheduler help.
- **`driver.resume()`** - after `reenter()`, drive until that callback
  returns to the transaction.
- **`driver.stall()`** - drive the current transaction to `STALLED`
  (stalling-supporting only).
- **`driver.pause()`** - drive the current transaction to `PAUSED`.
- **`driver.route(route)`** - install a route. A route is callable with
  the shape `route(driver)` and must return an iterator.  A generator
  function is the usual spelling; a callable object may also return itself
  and store its own state while implementing `__next__`. At each point
  where the Driver would normally ask the caller for instructions, the
  Driver calls the route instead. The route inspects the Driver, issues a
  Driver imperative, and yields. When the route returns, the Driver goes
  back to normal behavior and reports to its caller at the current ask
  point. Routes compose naturally with `yield from`, and branch naturally
  with ordinary `if` / `while` statements. `route(X)` inside a route is a
  hard hand-off to a new route; `yield from subroute(driver)` is a
  subroutine call that comes back.  A route may issue one directive and
  then yield; yielding without a directive raises.

`driver.wait(*signals)` always includes `Terminated(driver.thread)` as its
intrinsic ender. If the thread terminates and you listed that signal, the
wait succeeds like any other listed signal; if you didn't list it, the
Driver enters `terminated`. Transaction exit is not intrinsic to
`wait`--if you care about a transaction, list the transaction or a
transaction-specific signal yourself.

Other methods:

- **`driver.refresh()`** - immediately re-sample the Driver's live thread
  state after intentional out-of-band driving. It stages nothing and
  drives nothing; it is the "I moved this myself, on purpose" operation.
- Calling the driver itself (a la `driver()`) commits the staged directive
  and drives until that directive reaches an ask point: success, nested,
  raised, termination, impasse, or a route ending.  Calling it with no
  staged directive raises and leaves the Driver `undirected`.

### Scenario.Chain

**`scenario.Chain(*drivers)`**

Construct a Chain over zero or more `Driver`s.

Properties:

- **`chain.pending`** - tuple of the pending Drivers.

Methods:

- **`chain.append(driver)`** - add a driver to the pending list.
- **`chain.remove(driver)`** - remove a driver.
- **`chain.promote()`** - pop and return the pending-head Driver
  without driving it.  Returns `None` if pending is empty.  The
  returned Driver is unowned; the caller must register it with
  a Dispatch (or close it) before letting it go out of scope.
  Useful for custom iteration patterns.
- **`driver in chain`** - membership test.
- **`for d in chain:`** - iterate, yielding the Driver at the
  head of `pending` each time, driving it forward to its next ask point
  before moving to the next.
- **`len(chain)`** - total count of drivers.
- **`bool(chain)`** - true if any drivers remain.
- **`chain.close()`** - close every Driver owned by this Chain.

### Scenario.Dispatch

**`scenario.Dispatch()`**

Construct an empty Dispatch.

Methods:

- **`dispatch.add(driver_or_chain)`** - add a driver or chain.
- **`dispatch.update(items)`** - add several.
- **`dispatch.remove(item)`** - remove. Raises `KeyError` if missing.
- **`dispatch.discard(item)`** - remove. No error if missing.
- **`item in dispatch`** - membership test.
- **`for d in dispatch:`** - iterate yielding Drivers as they need
  attention. The iterator runs until the dispatch is empty.
- **`dispatch.close()`** - close every Driver and Chain owned by this
  Dispatch.

### Scenario.inject

**`scenario.inject(module)`**

Monkey-patch supported `threading` and `queue` primitive references in
`module`. Returns an `Injection` handle, which is also a context manager.
Raises `ValueError` if no patchable references are found.

`Injection.close()` restores the pre-inject references.

It's possible to patch a module twice!  If you ever do that,
undo the patches in reverse order.  If you run `scenarioA.inject(X)`
and then `scenarioB.inject(X)` on the same module `X`,
you must un-inject B before un-injecting A.

### blanket.injector

A submodule of **blanket**, containing two things:
the `Location` class, and `inject_call`.

**`Location(function, start, stop)`**

Direct constructor. Usually you use one of the classmethods below.

**`Location.position(function, line, column=1)`**

Find an injection location by source position. `line` is relative
to the start of the function (1-based). On Python 3.10 and earlier,
only `column=1` is supported.

**`Location.text(function, text, *, skip=0, after=None)`**

Find an injection location by source text match.

**`Location.token(function, token, *, skip=0, after=None)`**

Find an injection location by Python token.

**`Location.bytecode(function, bytecode, *, skip=0, after=None)`**

Find an injection location by bytecode instruction name.

**`inject_call(injected_function, location, *, name='')`**

Build a new function that's a copy of `location.function` with a
call to `injected_function` inserted at `location`. The original
function isn't modified.

Use case: execute `ev = threading.Event()`, and inject a call to
`ev.wait` in the middle of a function.  Call the function from
another thread.  You know the thread is now parked at the `ev.wait()`
call, and will only resume when you call `ev.set()`.

## Method Reference

This is the full state-by-state and method-by-method reference for
**blanket** regulated methods and the transactions they use.
If you're only using the high-level APIs, you probably don't need
this; this section is for when you want to know
exactly what each state means and which states a given method visits.

### Transaction States

Transactions are implemented as state machines.  The machine has a total
of ten states, including the start state (**`BLOCKED`**) to the terminal
states (**`RETURNED`** and **`RAISED`**).  These states are *ordered;*
the transaction state machine only transitions to a subsequent state.
(There are no back-transitions; a transaction will never transition
from a later state back to an earlier one.)  Every transaction starts at
**`BLOCKED`** and ends at either **`RETURNED`** or **`RAISED`**; the
states in between depend on which method was called.

- **`BLOCKED`** - the *scheduler block*.  The initial state--every
  transaction starts here.  Always visited.  When the thread calls a
  **blanket** method wrapper, say `lock.acquire()`, that starts a
  transaction, which immediately goes into **`BLOCKED`** state.  Note
  that no real work has been done yet--the actual method on the real
  underlying synchronization primitive hasn't been touched yet.  The
  scheduler can inspect the transaction, call `expire` or `disregard`
  (if the method call supports timeouts), and ultimately must "unblock"
  the transaction to let it make progress.

- **`COMMIT`** - Signifies calling the actual method on the underlying
  synchronization primitive.  When you call `lock.acquire`, you're
  calling a wrapper; when the transaction enters this state, that means
  it's called the real `acquire` method on the real `Lock` object.
  Depending on the method called, this *can* be a "parking state",
  meaning the transaction will block--some methods never block in this
  state, some methods might block or might not.  The scheduler can't
  directly unpark a transaction parked in **`COMMIT`** state; the
  method will unpark itself when its conditions are met.  For example,
  if `lock_q` is currently locked, and thread A calls `lock_q.acquire`,
  that transaction will park in **`COMMIT`** state until some other
  thread unlocks `lock_q` and thread A's call to `lock_q.acquire` returns.
  Transactions that have been forced to time out using `expire` still
  enter **`COMMIT`** state, but they do so with an immediate timeout.

- **`WAITING`** - Similar to **`COMMIT`** state: a special parking
  state reached while calling the actual method on the actual primitive,
  not directly under the control of the scheduler.
  **`WAITING`** is only visited by a few transactions, and those
  transactions represent blocking method calls that can get into a race.
  The classic example here is `Condition.wait` vs `Condition.notify`.
  If thread A is calling `cond_x.wait`, and thread B is calling `cond_x.notify`,
  will the notify call wake up the wait call, or not?  You can tell
  by examining the state of the transaction on thread A.  If thread
  A's transaction has reached **`WAITING`** state, it has registered
  itself as a "waiter" internally on the primitive, and will be awakened
  by a subsequent "notify" call.

- **`STALLED`** - the *scheduler stall*.  Only visited by `Condition.wait`
  transactions.  The underlying method call has woken up, and
  transitioned out of `WAITING` state, and is now attempting to
  reacquire the underlying lock.  When the scheduler *unstalls*
  the transaction (unparks the transaction from **`STALLED`** state)
  the `Condition.wait` call will immediately call `acquire` on that
  condition's underlying lock.

- **`RESUMED`** - a transitory state.  Only visited by `Condition.wait`
  transactions.  The transaction has been "unstalled"; at the moment
  the transaction enters this state, it's likely still executing the
  actual method on the actual primitive, but there are no more parking
  states, and the actual method should return soon.

- **`COMMITTED`** - a transitory state.  Always visited.  The transaction
  has finished calling the actual method on the actual primitive, and
  you may now examine the `result` on the transaction.

- **`PAUSED`** - the *scheduler pause*.  A parking state managed
  by *blanket*.  The scheduler can request that a transaction park
  here with `tx.pause = True`; *blanket* high-level APIs also often cause
  a transaction to park in **`PAUSED`** state.  The
  transaction parked in **`PAUSED`** state will be held in that state
  as long as any functionality is requesting that state.

- **`EXITING`** - a transitory state.  Always visited.  The transaction
  is nearly done, and will immediately transition either **`RETURNED`**
  or **`RAISED`**.

- **`RETURNED`** - a terminal state.  The method returned a value.

- **`RAISED`** - a terminal state.  The method raised an exception.

### Notable Primitive Methods

This subsection documents all the primitive methods that do anything
unusual.  Methods not listed here are conventional and don't do anything
interesting.  (Also known as "boring" transactions.)

#### Lock and RLock

- `acquire`: Can time out.  Parks in **`COMMIT`** state if the lock
  is already locked.

#### Condition

- `__init__`: *blanket* imposes an additional restriction on `Condition`
  objects not imposed by the `threading` module.  You can pass in your
  own lock object in to `threading.Condition`, and it doesn't even need
  to be a real `Lock` or `RLock`; `threading.Condition` allows duck-typed
  lock objects, and introspects the object to discover which methods it
  supports.  *blanket* is more restrictive: although you can pass your
  own lock object into `scenario.Condition`, it *must* be an instance
  of either `scenario.Lock` or `scenario.RLock` from the same scenario.

- `acquire`: Identical to `acquire` on a `Lock` or `RLock`.
  (Literally calls `acquire` on the underlying lock, which must
  be a `Lock` or `RLock`.)
  Can time out.  Parks in **`COMMIT`** state if the lock is already locked.

- `wait`: Can time out.  Parks in **`WAITING`** state after registering
  as a waiter internally and releasing the underlying lock.  If a
  `Condition.wait` transaction has reached **`WAITING`** state, it's
  eligible to be awakened by a `Condition.notify` or `Condition.notify_all`
  call running on another thread.  Parks in `STALLED` state after the
  underlying `wait` call is awakened by a `notify` call, immediately
  before attempting to reacquire the condition's underlying lock.

  `wait` releases and reacquires the Condition's underlying lock,
  but those operations are not visible as separate nested transactions.
  It's the same `Condition.wait` transaction the whole time: it parks in
  **`WAITING`** after releasing the lock, then parks in **`STALLED`**
  after it wakes up and before it reacquires the lock.

- `wait_for`: The most complex method in the `threading` module, and
  therefore the most complex transaction in *blanket*.  Can time out.

  `wait_for` enters **`COMMIT`** state before calling the actual
  `wait_for` method on the actual `Condition` object, and stays there
  until `wait_for` returns.  The real `wait_for` method is implemented
  as a loop.  Inside the loop, it initially calls the `predicate` callable.
  If the `predicate` returns a true value, it returns that value.  If the
  `predicate` callable returns a false value, `wait_for` calls `self.wait`
  then loops.  This means `wait_for` can potentially call the `predicate`
  and `self.wait` an arbitrary number of times.

  Whenever `wait_for` calls its `predicate` callback function, *blanket*
  notifies the scheduler by signaling `Predicate(tx)`.  This is important
  in case the `predicate` makes method calls on other primitives; the
  scheduler will need to manage those method calls, too.  If the predicate
  does call any other primitive methods, these will be *nested* transactions.

  When `wait_for` calls `self.wait`, *blanket* creates a *nested transaction*
  for that `Condition.wait` call.  This is how the scheduler
  regulates access to the underlying lock in the middle of a `wait_for` call.
  If thread A calls `wait_for`,
  thread A will release the underlying
  lock when the scheduler *unblocks* the nested `wait` transaction,
  and
  thread A will attempt to reacquire the underlying
  lock when the scheduler *unstalls* the nested `wait` transaction.
  (And, again: this can potentially happen *multiple times,* if the
  `wait_for` call loops and reattempts the predicate and calls `wait`
  multiple times.)

#### Semaphore and BoundedSemaphore

- `acquire`:  Can time out.  Parks in **`WAITING`** state when the
  semaphore counter is zero.

#### Event

- `wait`:  Can time out.  Parks in **`WAITING`** state when the
  event isn't set.

#### Barrier

- `wait`:  Can time out.  Parks in **`WAITING`** state while
  waiting for the barrier to open.  The final call to `wait`--the
  call that opens the barrier, nicknamed the "opener"--does *not*
  enter **`WAITING`** state.

  If the barrier was constructed with an `action` callback,
  blanket will signal `Action(tx)` while calling the callback.
  If the callback calls any regulated methods on *blanket*
  primitives, these will create *nested transactions* on that
  thread.


## Running The Tests

**blanket**'s tests are plain old `unittest` tests.  The standard
workflow is to run the test-suite driver directly from the repository
root:

```
python tests/test_all.py
```

`test_all.py` is the human-friendly runner.  It runs the common test
modules plus the version-specific test files that make sense on the
Python you're currently using.

For coverage, run that same driver.  Pick the config for the Python
version you're running, and make sure that config is used for `run`,
`html`, and `report`--otherwise the report step will fall back to the
generic `.coveragerc` and you'll wonder why the numbers changed.

For example, on Python 3.13:

```
export COVERAGE_RCFILE=.coveragerc.py313
coverage erase
coverage run tests/test_all.py
coverage html
coverage report -m
```

On Python 3.14, use `.coveragerc.py314`.  There are configs for Python
3.10 through 3.14.  They measure both **blanket** and the tests, and set
`fail_under = 100`.  The version-specific configs omit test files that
`test_all.py` correctly doesn't run on that interpreter; for example,
the Python 3.10 config omits the Python 3.11+ injector tests and the
Python 3.13+ queue-shutdown tests.

If you want to use coverage's `sysmon` core on a Python that supports it,
set `COVERAGE_CORE=sysmon` too.  That's a coverage implementation detail,
not a **blanket** requirement.

You can also run discovery as a sanity check:

```
python -m unittest discover -s tests -p 'test_*.py'
```

But the official test and coverage path is `tests/test_all.py`.


## Changelog

**1.1**

- Added regulated `queue` support.  `Scenario` now supplies
  `Queue`, `LifoQueue`, and `PriorityQueue`, and `SimpleQueue` on 3.7+.
  These are real stdlib queues underneath, with
  **blanket** regulation wrapped around their public methods.
    - `scenario.inject()` now supports `queue` as well as `threading`.
    - The high-level API for queue objects is `deliver()`.  It's something
      like the `allocate` API for semaphores; you pass in a sequence of
      thread handles, where each one will next call `get` or `put` (or
      an equivalent call like a `nowait` version), and `deliver` orchestrates
      the calls together.

- Added "base_tx" support.  For all APIs where you pass in a thread handle,
  you can now also pass in a `(thread, base_tx)` tuple.  The tuple form
  means the API will operate strictly on child transactions of the `base_tx`
  transaction.  This works with `park`, `skip`,
  `block`, `pause`, `assign`, `relay`, `allocate`, `deliver`, `cycle`,
  the per-primitive transaction helpers (`unblock`, `unpause`, `expire`,
  `disregard`, `revert`, and `ConditionAPI.unstall`), and the `Call` and
  `Use` signal tokens.

- Changes to `Driver`:

    - Reworked `Driver` around explicit staged directives.  A fresh Driver
      starts in `undirected` and does nothing until directed; use `scan()` to
      find the thread's current published transaction.  Driver's public states
      are now `undirected`, `driving`, `success`, `nested`, `returned`,
      `overshot`, `persisted`, `mutated`, `raised`, `terminated`, and
      `impasse`.

    - Changed nested-transaction semantics: nested transactions are no longer
      automatically silently skipped by Driver.  When the driver detects a
      nested transaction, it reports back with that child selected as
      `driver.tx`; the scheduler decides what to do next.  Driver-level
      `skip()` and every `autoskip=` parameter were removed.

    - Driver directives are staged into one pending slot, and the last staged
      directive wins.

    - `raised` is no longer a terminal Driver state.  `terminated` is now the
      only terminal state -- the only one a Driver can never leave.  A
      transaction raising does not kill its thread (an *uncaught* raise would,
      landing in `terminated`), so a worker that catches the exception and
      keeps running is now drivable past the raise: `scan()` recovers and
      picks up its next transaction, the same way you recover from `mutated`.

    - `Driver.wait` is now the Driver-level companion to `scenario.wait`:
      `driver.wait(*signals)` passively waits until one of the named signals
      fires. It doesn't release the worker through parking states and it
      doesn't require a current transaction. Driver now publishes
      `waited`, `signaled`, and `motivation` after a drive, and `refresh()`
      lets you bless deliberate out-of-band driving before observing with
      `wait`.

    - Added Driver routes.  Construct a Driver with `route=...`, or install one
      later with `driver.route(route)`, where `route` is a generator function
      accepting the Driver.  Routes get the Driver at each ask point, issue one
      imperative, and yield; they can compose with `yield from` and branch with
      ordinary Python control flow.

    - `Driver` is now more relaxed about multiple drivers operating on the
      same thread.  You can now have as many as you like, provided that only
      one is active at a time.

    - Modernized Driver consumers to use routes for their deep-driving and
      callback choreography.  `Condition.cycle()` no longer drops into raw
      `score.wait` for `wait_for` predicate routing, and the old hidden
      `listen_predicate` / `listen_action` Driver flags are gone.

    - Added Driver observability: `DriverStatus`, `driver.status`, and
      `driver.log`.  Added `tx.visited(*states)` as a convenience query over
      a transaction's state-transition log.

    - Reworked pause handling around two pause bits: a scheduler-owned
      `tx.pause` bit and a blanket-owned `blanket_pause` bit.  `tx.pause` is
      again a boolean property; `tx.paused` reports whether either bit is
      holding the transaction at `PAUSED`; `tx.unpause()` clears the user
      bit and raises if blanket is still holding the transaction.  The
      old public `tx.pausing` and `tx.unpark()` APIs were removed.

    - Added `scenario.wait(..., all=True)`, which waits for every supplied
      item to signal at least once under one shared timeout and returns the
      accumulated `frozenset`.

- Changes to `cycle`:

    - Majorly reworked `Condition.cycle()`.  It now handles plain `wait()` and
      `wait_for()` more carefully, including the case where the first
      `wait_for` predicate call succeeds immediately and no nested wait ever
      occurs.  It exposes ready waiters, supports `cycle.wait()` for driving
      false-predicate `wait_for` wakeups back into the waiting phase, and
      manages predicate reentry via `scheduler(tx)`.

    - `Barrier.cycle()` and `Condition.cycle()` scheduler callbacks now receive
      the transaction being reentered: `scheduler(tx)`.  This matters when one
      cycle object is managing multiple `wait_for` calls or a barrier action;
      `tx.thread` tells you which thread is asking for scheduler help.

- Changes to other high-level APIs:

    - Reworked `SemaphoreAPI.allocate()`.  It now uses the same traffic-script
      model as `deliver()`: each participant's next semaphore operation must
      be `acquire` or `release`, duplicate threads are allowed, and only one
      semaphore transaction runs at a time.  The old preflight "prove this
      batch can complete" check is no longer viable; an impossible script
      now blocks, just like the program you're modeling would block.

    - Removed `scenario.finish()`.  It was too vague--"finish whatever this
      thread has left to do" sounds convenient, but it overlaps badly with
      the more explicit Driver and middle-level APIs.

    - Removed the old `wait=` parameter from `scenario.park()` and
      `scenario.skip()`.  Added the explicit `scenario.block()` and
      `scenario.pause()` helpers instead.  `park()` is the lenient one
      (skip unrelated work until the named call appears); `block()` is the
      strict one (the named call must be next).  `skip()` drives the
      transaction to a terminal state;
      `pause()` drives the transaction to `PAUSED` state.

- Changes to `scenario.wait` and signals:

    - Reworked signal handling.  Signals are now self-reporting objects with
      `sample(scenario)`.  **blanket** no longer keeps references to signaled
      objects forever, so old signal objects that go out of scope will be
      collected normally.

    - Removed external wrappers for objects like `Primitive` or `BoundMethod`
      or `Thread` when working with `scenario.wait()`  These objects are now
      accepted directly as aggregate signals, by `scenario.wait`, `Not`, etc.

    - `Not(...)` now works with the normal signal forms, including bare
      threads, bound primitive methods, and bare primitives.  Raw handles and
      raw bound methods normalize to their corresponding regulated handles.

    - Added `Action(tx)` and `Predicate(tx)` signal support for user callbacks:
      `Action(tx)` is high while a `Barrier.wait` transaction is running the
      barrier action, and `Predicate(tx)` is high while a `Condition.wait_for`
      transaction is running its predicate.

- Tightened stdlib fidelity.  Version-specific APIs such as
  `queue.SimpleQueue`, `queue.Queue.shutdown`, `queue.ShutDown`,
  `threading.Condition.locked`, `threading.RLock.locked`,
  `threading.Semaphore.release(n=...)`, and CPython's old `Lock` legacy
  aliases appear on **blanket** objects only when the running stdlib has
  the corresponding feature.

- Added public scenario-side API aliases for `isinstance` checks, including
  `scenario.Transaction`, `scenario.LockAPI`, `scenario.ConditionAPI`,
  `scenario.QueueAPI`, and the rest of the primitive API wrapper types.

- `scenario.log` is again read-only, apart from `clear()`.  Scenario exit
  now leaves the completed transaction log available for post-mortem
  inspection; the log is cleared on the next scenario entry or explicit
  `reset()`.

- Updated the injector for newer CPython bytecode shapes while preserving
  the Python 3.10-and-earlier line-only source-location behavior.
  `Location.bytecode()` now documents the instruction-name API, and the
  old line/column behavior is covered by version-specific tests.

- Updated the test suite substantially.  The project now has
  version-specific coverage configs for CPython 3.10 through 3.14, using
  `tests/test_all.py` as the official test driver.

**1.0** *2026/05/14*

- Initial release!
