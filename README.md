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
makes it exceedingly hard to create a reproducable test case for
threading-related bugs.  And if you can't reproduce it reliably, you
can't debug it, and your unit test suite can't actually test it.

This problem only gets harder if you're shooting for 100% coverage.
Code paths that handle rare race conditions are by definition *rare*.
But if you can't write a test that reliably reproduces the condition,
how do you test it in your test suite?  How do you get to 100%?

And this problem is only going to get harder. Python's "free threading" mode
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
the real `threading.Lock`, `threading.Condition`, and so on,
rather than reimplementing them. Your tests use the real primitives,
which means they're guaranteed to behave like the real thing--because
they *are* the real thing, just under **blanket** control.  But for
this to work, your code has to replace the real `threading` module
primitives with **blanket**-wrapped versions.

**blanket** requires Python 3.7 or newer.  It depends on
[my **big** library,](https://github.com/larryhastings/big)
and the optional bytecode injector needs the
[**bytecode**](https://pypi.org/project/bytecode/) module.
**blanket** is 100% pure Python.

The current version is **1.0**.

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
`RLock`, `Condition`, `Semaphore`, `BoundedSemaphore`, `Event`, or
`Barrier`. You construct them on the scenario: `scenario.Lock()`,
`scenario.Barrier(3)`, etc. Each primitive is wired into the
scenario, making every method call on it observable and steerable
by the scheduler.

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

**blanket** requires Python 3.7 or newer, and depends on
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
- drive worker threads through method calls with the middle-level APIs: `scenario.park`, `scenario.skip`, `scenario.finish`, `scenario.Driver`, `scenario.Chain`, and `scenario.Dispatch`
- monkey-patch another module so it uses **blanket** synchronization primitives (`scenario.inject`)

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
the scenario, the primtiives automatically stop every time any
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


### The Seven Primitives

The `threading` module contains seven synchronization primitives:

* `Lock`
* `RLock`
* `Condition`
* `Barrier`
* `Event`
* `Semaphore`
* `BoundedSemaphore`

A **blanket** scenario object also contains these same seven
primitives, with the same names.  The objects they return
behave the same, with the same methods taking the same arguments.

When you construct create a `scenario.Lock()`, you get back a
*primitive handle*, or just a *primitive* for short.  Method calls
on the primitive behave exactly like the real thing--until you
enter the scenario.  Once you do, the primitive is *regulated*:
it blocks when it's called, to let you control it.
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

The *raw* is a second
handle to the same internal objects; methods on the *primitive* and
the *raw* both change the internal state of the object in the same way.
The only difference is that the raw handle is *always unregulated*;
when you call a method on it, it always runs immediately.

When is this useful?
Well, what if you need to change the state of a primitive while
*inside* the scenario?  You might want to tweak a semaphore
in the middle of a test, bumping up its value by calling `release`.
But if you just call `release` on the primitive, it'll be a
regulated call--and meanwhile, you're the guy who's supposed
to be calling into **blanket** and letting these calls make
progress.  You'd be deadlocked!

Instead, just use the raw handle:

```
with scenario:
    ...

    # totally fine
    raw = scenario.raw(my_semaphore)
    raw.release()
```

You might also want to give a *worker thread* (or some
other piece of subsystem code) an unregulated handle,
even though other code in the same test is using the regulated
handle.  Regulation follows the handle, not the primitive,
so different references to the same underlying object can be
regulated independently.  Imagine a test that exercises three
subsystems A, B, and C, sharing a lock, and you only want
**blanket** to control synchronization in A and C--maybe B is
incidental machinery you don't care about managing. You can give B
a raw handle to the lock; B will use the lock at full speed,
while A and C's calls on the same lock produce transactions and
flow through the scheduler.

Outside the scenario you don't need raws--the regulated handle is
already unregulated. Raws are only for when you're inside the scenario.


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
  `scenario.skip`, `scenario.finish`, and the
  `Driver`/`Chain`/`Dispatch` subsystem.
- **The high-level API**: per-primitive helper methods for
  common patterns in multithreaded programming: `assign`, `relay`, `cycle`, and
  `allocate`.

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

`scenario.wait(*items, timeout=None)` is the universal blocker. It
blocks the scheduler until any of the *items* you supply *signals*.
A wide range of objects can be items: bound methods on primitives
(signals while any thread is inside that method), the scenario
itself (signals while a thread has an active visible transaction),
a thread (signals while the thread has an active transaction),
a transaction (signals once the transaction has completed), and
the various signal-token classes documented below.

We'll see a lot more about `wait` in the *Signals And wait* section
to come.



### The Middle Level

The middle-level API drives threads through sequences of method
calls. There are three free functions and three classes.

#### park

`scenario.park(*args, wait=False)` drives one or more
supplied threads to specified methods, stopping each one at the
scheduler block on the method you specified. After `park` returns,
each supplied thread is parked, with its current transaction available
for inspection or manipulation:

```
with scenario:
    t = scenario.thread(worker)
    result = scenario.park(t, lock.acquire)
    # t is now parked on lock.acquire.
    # result[t] is the transaction at the scheduler block.
    result[t].unblock()
```

#### skip

`scenario.skip(*args, wait=False)` is similar, but it
drives the named threads *through* one or more method calls each.
After `skip` returns, the named threads have completed all the
method calls you specified:

```
scenario.skip(t, lock.acquire, lock.release)
# t has now completed both lock.acquire and lock.release.
```

#### finish

`scenario.finish(*threads)` drives each named thread
to a terminal state--best-effort, on the theory that you just want
them out of the way. Useful inside the scenario for cleaning up
specific parked threads before you move on to the next phase of the
test. You usually don't need to call `finish` just before scenario exit:
the exit cleans up for you, unparking blanket-parked threads automatically
and "join"-ing the managed threads.  You only need `finish` when you
want explicit control--driving a thread through specific final method
calls, or expiring its timeout instead of letting it sit.

#### Driver

`scenario.Driver`, `scenario.Chain`, and `scenario.Dispatch` are
objects used to drive one or more transactions running in one or
more threads.  They give you direct, manual control over the
underlying driver state machine. Most tests don't need them, but
the few that do tend to *really* need them.

A `Driver` attaches to a single worker thread. You construct one
with `scenario.Driver(thread)`. Drivers are lazy; nothing really
happens until you "drive" it, either by calling the object `driver()`,
or by giving it to a `Dispatch` and iterating over the dispatch.
(This means two Drivers can be constructed for the same thread
without immediately conflicting.  But only one `Driver` can drive
a thread at a time; trying to drive a thread with two `Driver`
objects at the same time is an error.)

A Driver gives you a set of imperatives--`skip()`, `finish()`,
`block()`, `commit()`, `wait()`, `stall()`, `pause()`--each of
which requests a state transition or series of transitions.
Again, this isn't done eagerly; the `Driver` remembers the request,
then makes it happen the next time it's driven.  You can only call
one imperative at a time; if you call a second imperative
before the first one has been driven, the driver raises.

A `Chain` is an ordered sequence of `Driver` objects.  Adding a
Chain to a Dispatch activates the chain's first driver; when that
driver reaches a terminal state, the next driver in the chain
takes its place; and so on, until the chain is empty.  You can
also iterate over a Chain directly, or pop drivers off the head
manually with `chain.promote()`.

A `Dispatch` is an iterator over drivers that need attention. You
add drivers (or chains of drivers) to it via `dispatch.add`. Each
time you call `next(dispatch)`, it returns whichever driver
needs the scheduler's attention next, having woken via a single
`scenario.wait` on the union of every driver's signals.

The typical pattern looks something like this:

```
with scenario:
    d1 = scenario.Driver(t1)
    d2 = scenario.Driver(t2)
    d1.skip()
    d2.skip()
    dispatch = scenario.Dispatch()
    dispatch.add(d1)
    dispatch.add(d2)
    for d in dispatch:
        # d is a Driver that needs attention
        ...
```

Note that if you only need to interact with one driver,
you can skip the `Dispatch` object.  Calling the driver
object drives it in isolation:

```
with scenario:
    d = scenario.Driver(t1)
    d.finish()
    d()
    # d has been driven and you can now inspect it
```

`park`, `skip`, `finish`, and everything in the high-level API
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
  available on `Lock` and `RLock` API objects. Manages one
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
  one must must call `acquire` followed by `release`, and the
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
  (`wake(thread, ...)`, `pause(thread, ...)`).

- **`allocate(*threads, pause=False)`** - on `Semaphore` and
  `BoundedSemaphore` API objects.  Drive an ordered sequence of
  semaphore acquires and releases. `threads` mixes acquirers and
  releasers; **blanket** figures out which is which based on what
  method the thread calls.  Returns an iterator, yielding each
  thread after its call has finished.


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

The `wait` method on a scenario is the universal blocker. It blocks
the scheduler until one of the items you give it "signals",
meaning, the condition it represents becomes true.  The design
for `wait` borrows heavily from Win32's wonderful
[`WaitForMultipleObjects`](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects)
function, which does basically the same thing.

The items you can give it cover a lot of conditions:

- A **thread** signals while the thread has an active transaction.
- A **transaction** Signals once the transaction has completed.
- A **bound method on a primitive**, e.g. `lock.acquire`. Signals
  while any thread has an active transaction on that method.
- A **`Call(thread, method)`** instance. Signals while `thread`
  is calling `method`.
- A **`Use(thread, primitive)`** instance. Signals while `thread`
  is calling any method on `primitive`.
- A **`Terminated(thread)`** instance. Signals once the thread has
  terminated.
- A **`Not(x)`** instance, where `x` is one of the above. Signals
  the opposite of the original; `Not(Terminated(x))` signals when
  the thread has *not* terminated.
- A **`Nested(transaction)`** instance. Signals while that
  transaction has a "child" or "nested" transaction.
  (Relevant for `Condition.wait_for`, which calls `Condition.wait`
  internally).
- An **`Action(transaction)`** instance. Signals while that
  transaction is in its *action phase*--temporarily pushed off
  its thread's transaction chain so that child transactions
  created during that window appear as fresh roots rather than
  children. Currently the only producer is `Barrier.wait`, which
  pushes itself around the user-supplied action callback.
- A **`Reached(transaction, state)`** instance. Signals while the
  transaction's state is at or past `state`.
- A **`TransactionState`** subclass instance: `Blocked(tx)`,
  `Waiting(tx)`, `Stalled(tx)`, `Resumed(tx)`, `Committed(tx)`,
  `Paused(tx)`, `Exiting(tx)`, `Returned(tx)`, `Raised(tx)`,
  `Commit(tx)`. Signals only while that transaction is in that state.
- The **scenario itself**. Signals while you've entered the scenario.

You can pass in as many of these as you like.  `wait` returns as soon
as *any* of them signals, and it returns a `set()` containing all the
items that signaled.  So if you want "either thread A terminates
or thread B reaches `WAITING`," you write:

```
signaled = scenario.wait(Terminated(A), Reached(b_tx, State.WAITING))
```

and now you can examine `signaled` to see which one signaled.

(What if you want to wait until *all* the items have signaled?
Just call `wait` multiple times, once for each item.)

`wait` also supports a `timeout` keyword-only parameter,
which if provided is the longest you will wait, specified
in seconds:

```
scenario.wait(t, timeout=5.0)
```

By default `timeout` is `None`, which means "no timeout, wait forever".
If a `wait` call times out, it raises `TimeoutError`.


## Monkey-Patching

Sometimes the code you want to test does its threading work via
`import threading` and references like `threading.Lock`.  Or maybe
even `from threading import Lock`.  You don't own the source,
you can't modify it so you pass in a pre-constructed `Lock`,
and you'd rather not patch each reference by hand.  For that case,
the scenario has an `inject` method.

`scenario.inject(module)` monkey-patches threading-primitive
references in `module` so that, while the patch is in place,
calls like `target_module.threading.Lock()` construct **blanket**
primitives on the scenario instead of real `threading` primitives.
Example:

```
import target_module

scenario = blanket.Scenario()
with scenario.inject(target_module):
    ...
    # Inside this block, target_module's threading.Lock,
    # threading.Condition, etc. all construct blanket primitives
    # bound to scenario.
    ...
```


It handles two reference patterns:

- Names bound directly to a threading primitive class, e.g.
  `from threading import Lock` or `Mutex = threading.Lock`. Each
  such name is rebound to the corresponding scenario primitive
  class.  (This is done by examining the value, not the name;
  something else that happens to be named `Lock` will be left alone.)
- A module attribute whose value is the `threading` module itself,
  e.g. `import threading`. That attribute is replaced with a small
  stand-in object whose `.Lock` / `.RLock` / etc. are the scenario's
  primitives, and whose other attribute lookups fall through to
  the real `threading` module.  So `target_module.threading.Lock()`
  constructs a **blanket** `Lock`, but `target_module.threading.Thread`
  is still `threading.Thread`.

The injection is returned as a handle, usable as a context manager
(as above) or closed explicitly with `.close()`. On close, the
original references are restored.

If `scenario.inject(module)` can't find anything to patch, it
raises `ValueError`--almost certainly a sign that you've pointed
it at the wrong module, or the target imports `threading` lazily
inside a function (so the import hasn't happened yet at the time
inject runs).



## Subtle Behaviors And Tips

A handful of small things worth knowing.

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
or expiring its timeout instead of waiting for it to fire--you can
still call `scenario.finish(*threads)` (or use a `Driver` directly)
inside the scenario, before exit. The auto-unpark at exit is only for
the "just let the worker finish on its own" case.

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

The `Driver` imperatives (`skip`, `finish`, `block`, `commit`,
`wait`, `stall`, `pause`) are *lazy*. They request a state
transition, but they don't fire the underlying work until the driver
is actually driven--by calling `driver()`, or by a `Dispatch`
driving it. A driver carries at most one staged imperative at a
time: calling a second imperative before the first one is driven
raises. The point of laziness isn't to let you stack imperatives;
it's to separate *what should happen next* from *when it happens*,
so that `Dispatch` can be the one to fire the work in coordination
with whatever other drivers are also active.

If you're working at the middle level, this rarely matters--`park`
and `skip` handle the driving for you. But if you reach for explicit
`Driver` / `Chain` / `Dispatch`, knowing the laziness rule will
save you confusion.

## The Injector

There's one more piece to **blanket** that doesn't fit anywhere
in the scenario story: a bytecode injector, in the
`blanket.injector` submodule. This part is small, optional, and
serves a comparatively rare use case--if you need it, you need it,
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
identical to the original except: you've inserted single inserted
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

- **`Location.bytecode(function, offset, stop=None)`** - by raw
  bytecode offset. The escape hatch for when you've built the
  function with a bytecode-manipulation library and the other
  methods can't see your source.

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
scenario core.  There's exactly one of them per scenario.  Every regulated
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
rather than firing it eagerly. The point isn't to let you batch
multiple intents (you can't--a second imperative on a driver with
one still pending raises), it's simply to delay setting state on
the Driver and the transaction until we're actively driving them.

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

**`blanket.Signaled`**

The marker base class for signal-token objects. All signal tokens
are subclasses of `Signaled`. Mostly of interest if you're writing
code that needs to dispatch on whether a given object is a signal
token; everyday users won't reference it.

**`blanket.Reached(transaction, state)`**

Signal token. Signals while `transaction`'s state is at or past
`state`. Useful for "wait until the transaction has reached at
least this point."

**`blanket.Call(method, thread)`**

Signal token. Signals while `thread` is in a transaction on
`method`.

**`blanket.Use(thread, primitive)`**

Signal token. Signals while `thread` has any transaction on
`primitive` in its call chain. ("Use" is the noun form here--it
rhymes with "moose", not "booze".)

**`blanket.Not(token)`**

Signal token. Signals while `token` is *not* signaling.

**`blanket.Terminated(thread)`**

Signal token. Signals once `thread` has terminated.

**`blanket.Nested(transaction)`**

Signal token. Signals while `transaction` has a child transaction
in flight.

**`blanket.Action(transaction)`**

Signal token. Signals while `transaction` is in its *action phase*
--temporarily pushed off its thread's transaction chain, so that
any child transactions created during the push window appear as
fresh roots (parent `None`) to the rest of **blanket**. Currently
the only producer is `Barrier.wait`, which pushes itself around
the user-supplied action callback.

**`blanket.TransactionState(transaction, state)`**

Signal token base class for "transaction is exactly in this state."
Has the following subclasses, one per non-transit state:

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

**`blanket.TransactionAPI`**

The class of the transaction-wrapper objects returned by the
scheduler-facing API (e.g. `scenario.transaction(t)`). Useful
for `isinstance` checks. The methods on a `TransactionAPI` are
documented in the *Transactions* subsection of `Scenario` below.

### Scenario

**`Scenario()`**

Construct a new scenario.

**`scenario.name`**

The scenario's name (a string), used in `repr()`. Settable.

**`scenario.reset()`**

Clear accumulated working state from the scenario--terminated
transactions, the waiters reverse index, the log. Leaves structural
state alone (primitives, threads, family signal sets). Safe to
call repeatedly. Called automatically on scenario entry.

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
order.

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

**`scenario.wait(*items, timeout=None)`**

Block until any of `items` signals. See the *Signals And wait*
section for the supported item types. Raises `TimeoutError` if
`timeout` expires.

**`scenario.park(*args, wait=False)`**

Drive named threads to specified methods, parking each at the
scheduler block. Arguments come in `(thread, method)` pairs:

```
scenario.park(A, lock.acquire, B, lock.release)
```

Each thread may appear at most once. Returns a dict mapping
thread to the transaction at the scheduler block. With `wait=True`,
also drives each call through to completion and returns the
completed transactions.

**`scenario.skip(*args, wait=False)`**

Drive named threads *through* specified methods. Arguments are
flat: thread, then one or more methods for that thread, then
optionally another thread, etc.:

```
scenario.skip(A, lock.acquire, lock.release, B, lock.acquire)
```

Returns a dict mapping thread to the last transaction. With
`wait=True`, waits for the last method on every thread to finish.

**`scenario.finish(*threads)`**

Best-effort drive each named thread to a terminal state. Loops
until every named thread has terminated. May deadlock if a thread
is parked on something the test never provides.

**`scenario.__enter__()` / `scenario.__exit__(...)`**

Enter and exit the scenario context. Inside the context, the
calling thread takes the role of the scheduler.

On exit, in order: any still-active `Driver` is closed; the
scenario flips to unregulated (subsequent calls on the primitives
pass straight through to the underlying real primitives); every
transaction currently parked at a blanket-controlled park
(`BLOCKED`, `STALLED`, `PAUSED`) is released, so its worker can
resume and finish natively; every managed worker thread is
`join()`ed; transient state is reset.

### The Seven Primitives

Each is constructed as a method on the scenario, e.g.
`scenario.Lock()`. All faithfully implement the public surface of
the corresponding `threading` type:

**`scenario.Lock()`** - `acquire(blocking=True, timeout=-1)`,
`release()`, `locked()`, `__enter__`/`__exit__`.

**`scenario.RLock()`** - `acquire(blocking=True, timeout=-1)`,
`release()`, `locked()` (where supported), `__enter__`/`__exit__`.

**`scenario.Condition(lock=None)`** - `acquire(...)`, `release()`,
`wait(timeout=None)`, `wait_for(predicate, timeout=None)`,
`notify(n=1)`, `notify_all()`, `__enter__`/`__exit__`.

**`scenario.Semaphore(value=1)`** - `acquire(blocking=True, timeout=None)`,
`release(n=1)`, `__enter__`/`__exit__`.

**`scenario.BoundedSemaphore(value=1)`** - same as Semaphore;
`release` raises if it would exceed initial value.

**`scenario.Event()`** - `is_set()`, `set()`, `clear()`,
`wait(timeout=None)`.

**`scenario.Barrier(parties, action=None, timeout=None)`** - `wait(timeout=None)`,
`reset()`, `abort()`, plus the `parties`, `n_waiting`, `broken` properties.

In addition, every primitive has a `name` property (settable).

### Per-Primitive API Objects

Each primitive has a corresponding API object available via
`scenario.api(primitive)`. The API object has these methods:

Every API object:

- **`api.unblock(method, *threads, pause=False)`** - unblock the
  named threads' transactions on `method`. `method` is the bound
  method on the primitive.
- **`api.unstall(method, *threads)`** - release the named threads'
  transactions on `method` from the `STALLED` park. Used after a
  notify on a `Condition.wait` transaction that's stalled mid-commit,
  to let the worker proceed into the internal lock re-acquire.
- **`api.unpause(method, *threads)`** - decrement the pause counter
  on the named threads' transactions; transactions whose counter
  reaches zero unpark from `PAUSED`.
- **`api.expire(method, *threads)`** - expire the named threads'
  transactions on `method` (only meaningful for timeout-bearing
  methods).
- **`api.disregard(method, *threads)`** - disregard the named threads'
  timeouts on `method`.
- **`api.revert(method, *threads)`** - undo any prior `expire` or
  `disregard` on the named threads' transactions, restoring the
  user's original timeout. Operates on transactions in `BLOCKED`.

Lock and RLock API objects also have:

- **`api.assign(thread, acquirer=None, *, pause=False)`** - assign
  the lock to `thread`. With one argument, `thread` simply acquires.
  With two arguments, `thread` releases and `acquirer` acquires.
- **`api.relay(initial, *acquirers, pause=False)`** - chain the
  lock through the named threads. `initial` may be either the
  current holder (parked at release/BLOCKED) or an acquirer on an
  unheld lock (parked at acquire/BLOCKED); each `acquirer` takes
  the lock in turn after `initial`. Returns an iterator yielding
  each acquirer as it takes the lock.

Semaphore and BoundedSemaphore API objects also have:

- **`api.allocate(*threads, pause=False)`** - drive an ordered
  sequence of semaphore acquires and releases.

Condition, Event, and Barrier API objects also have:

- **`api.cycle(*threads)`** - construct a `Cycle` over
  the named threads.  All the threads except the last thread
  should be making `wait` or `wait_for` calls.  The last thread
  is the "opener", and should be calling some sort of *notify*
  function: `Condition.notify`, `Condition.notify_all`, `Event.set`,
  or in the case of `Barrier` it should be the last `waiter`, which
  opens the barrier.

  The cycle is a context manager and exposes
  `wake`, `pause`, `iter`, `close`, and is callable for
  wake-and-close shorthand. `wake` and `pause` are polymorphic on
  argument count: with at least one thread named, they drive those
  threads and return a tuple; with no arguments, they drive the
  first remaining waiter (per spec order) and return that single
  thread, raising `ValueError` if the cycle is empty.

  `Barrier.cycle` actually takes one keyword-only parameter,
  `barrier_api.cycle(*threads, scheduler=None)`.  If specified,
  `scheduler` should be a callable; it will be called after
  allowing the last `wait` call to execute, which runs the
  barrier's "action" if any.  If the "action" calls methods
  on **blanket**-regulated primitives, you'll need to run
  scheduler code to control the execution of those primitives;
  this `scheduler` callback is the right place to run that code.

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
- **`tx.pause`** - read/write boolean. Setting `tx.pause = True`
  tells the transaction that you want it to pause at `PAUSED` state.
- **`tx.pausing`** - read-only; `True` if either you *or* **blanket**
  *itself* have asked the transaction to pause.
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

- **`tx.unblock()`** - unblock the transaction from the scheduler
  block.
- **`tx.unpause()`** - equivalent to `tx.pause = False`, but also,
  unparks the transaction if the scheduler pause if you were the
  only party requesting **`PAUSING`** state.  (**blanket** can turn
  on pausing state too, and a transaction parked in **`PAUSING`**
  state won't unblock until all parties give it permission to resume.)
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
- **`tx.push()`** - temporarily remove this transaction from its
  thread's transaction chain.  The push happens at the time of
  the call; the returned object is callable, and calling it
  (or using it as a context manager and exiting it) pops.
  Preconditions: the transaction must be a chain root
  (`tx.parent` is `None`) and not already pushed.  While pushed,
  any child transactions appear as fresh roots rather than
  children, and `Action(tx)` signals high.  Currently used by
  `Barrier.wait` around its action callback to hide the barrier
  transaction from any primitive calls inside the action.

### Scenario.Driver

**`scenario.Driver(thread)`**

Construct a Driver attached to `thread`.  A Driver
"drives" a thread, which is to say, it causes method
calls made on primitives by the thread to make progress.
You can tell the Driver what you want the thread, or the
tx running on the thread, to do, and the Driver will make
it happen and report back when it's successful--or if
some unexpected thing happened (the thread terminated!)
and it can no longer make progress on your request.

Properties:

- **`driver.thread`** - the thread.
- **`driver.state`** - the driver state (`None` until first
  driven).
- **`driver.tx`** - the current transaction (`None` if none).
- **`driver.txs`** - tuple of all transactions seen so far.
- **`driver.done`** - `True` if in a terminal driver state.

Driver has these states:

- `idle`, no tx is observed on the thread.
- `active`, a tx is observed on the thread, and the driver hasn't
  been instructed to drive it.
- `skipping`, driver has been instructed to "skip" (drive to
  completion) the tx on the thread.  driver will automatically
  unpark the tx from any scheduler-controlled park state.
  after the tx finishes, driver transitions back to `idle`.
- `parking`, driver has been instructed to "park" the tx in a
  particular tx state.  driver will automatically unpark the
  tx from any scheduler-controlled park state, until either it
  reaches the requested tx state, or it overshoots it or finishes,
  in which case driver raises an error.  if tx parks in the desired
  state, driver transitions to `parked`.
- `nesting`, driver has been instructed to drive the tx until
  a nested transaction springs into existence on top of it
  (the parent has a child).  driver transitions back to `idle`
  when this happens, leaving the nested tx as the new current
  tx on the thread.
- `finishing`, driver has been instructed to drive the tx until
  it finishes (reaches a terminal state).  when the tx transitions
  to `RETURNED` state, driver transitions to `finished`.
- `parked`, terminal state driver transitions to after successfully `parking`.
- `finished`, terminal state driver transitions to after successfully `finishing`.
- `raised`, terminal state driver transitions to if the tx transitions
   to `RAISED` state.
- `terminated`, terminal state driver transitions to if the thread terminates.

If Driver is in a terminal state, you can reuse it.  If there is no
tx on the thread, it will transition back to `idle` and wait for a tx
to spring into existence; if a tx is active on the thread, it will transition
to `active` and return immediately.  If you call an imperative then
drive it again, it will transition back into the appropriate driving
state (`finishing` or `parking`) and continue from there.

Driver also publishes the sets `driving_states`, `active_states`,
and `terminal_states`, which are `frozenset` objects containing
those states.

Driver supports several "imperatives"; these are instructions for what
you want the driver to accomplish when driving the thread.  Note that
these simply set internal state, instructing Driver what you want done;
Driver doesn't change any state on a transaction until you let it
start driving:

- **`driver.skip()`** - let the current transaction complete normally.
- **`driver.finish()`** - drive the worker to a terminal driver state.
- **`driver.block()`** - park the worker at the scheduler block,
  without unblocking.
- **`driver.commit()`** - drive the current transaction to `COMMIT`
  (timeout-bearing only).
- **`driver.wait()`** - drive the current transaction to `WAITING`
  (waiting-supporting only).
- **`driver.stall()`** - drive the current transaction to `STALLED`
  (stalling-supporting only).
- **`driver.pause()`** - drive the current transaction to `PAUSED`.
- **`driver.nested()`** - drive the current transaction until a
  nested (child) transaction is created on top of it; used to
  wait through a transaction's "action" phase, e.g. the user-
  supplied action callback on `Barrier.wait`.

Other methods:

- Calling the driver itself (a la `driver()`) drives the driver until
  the driver needs further instructions: it has succeeded in your
  requested imperative, it can no longer succeed with your
  requested imperative, or a new transaction has started and it doesn't
  know what you want done.
- **`driver.close()`** tells the driver to stop managing that thread.

A driver only actively manages a thread while it's driving.  When
you call the driver--or add it to a Chain or Dispatch, and iterate over
that object--it "owns" the thread, and you can't attach a second Driver
to the same thread.  You can't drive one thread from two Driver objects
at once.


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
  head of `pending` each time, driving it forward and waiting
  for it to reach a terminal state before moving to the next.
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

Monkey-patch threading-primitive references in `module`. Returns
an `Injection` handle, which is also a context manager. Raises
`ValueError` if no patchable references are found.

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

**`Location.bytecode(function, offset, stop=None)`**

Find an injection location by raw bytecode offset.

**`inject_call(injected_function, location, *, name='')`**

Build a new function that's a copy of `location.function` with a
call to `injected_function` inserted at `location`. The original
function isn't modified.

Use case: execute `ev = threading.Event()`, and inject a call to
`ev.wait` in the middle of a function.  Call the function from
another thread.  You know the thread is now parked at the `ev.wait()`
call, and will only resume when you call `ev.set()`.

## Transaction State Reference

This is the full state-by-state and method-by-method reference for
**blanket** transactions.  Most users will only need the breezy
overview in *Threads And Transactions* near the top of the document;
this section is for when you want to know exactly what each state
means and which states a given method visits.

### Transaction States

- **`BLOCKED`** - the *scheduler block*.  Every transaction starts
  here.  The method has been called, but no work has happened yet:
  the worker is asleep inside **blanket** and the underlying real
  primitive hasn't been touched.  The scheduler can inspect the
  transaction, expire or disregard a pending timeout, and ultimately
  unblock to let it proceed.

- **`COMMIT`** - a parking state for timeout-bearing transactions
  (`Lock.acquire(timeout=...)`, `Condition.wait(timeout=...)`,
  `Barrier.wait(timeout=...)`, and so on).  The transaction is
  committed to its action and is about to attempt it.  The scheduler
  can choose to hold the transaction here for explicit
  commit-vs-timeout decisions, then unblock to proceed.

- **`WAITING`** - a parking state for transactions blocked inside
  the underlying real primitive.  The transaction has called into
  the real `condition.wait()`, the real contended `lock.acquire()`,
  the real `barrier.wait()`, etc., and is now genuinely asleep
  inside the primitive.  This state is *not* directly under the
  scheduler's control--it's the underlying primitive's to manage.
  The scheduler can observe that the transaction is in `WAITING`
  and use that as a signal, but it can't `unblock` it.  The primitive
  has to choose to wake up, by way of a notify, a release, the last
  barrier party arriving, or a timeout expiring.

  (For `Condition.wait` specifically, `WAITING` is also where the
  underlying lock is dropped while waiting--the real `condition.wait`
  releases the lock as part of waiting and re-acquires it on wake.
  The re-acquire is itself a transaction, nested inside the `wait`.)

- **`STALLED`** - the *scheduler stall*.  The transaction has woken
  up from `WAITING` (or from `COMMIT`) and is now back under the
  scheduler's direct control, but hasn't yet been permitted to
  proceed to its commit work.  This is where you can intercept a
  thread *after* it's woken up from a real primitive wait but
  *before* it's done any post-wake bookkeeping.

- **`RESUMED`** - a transit state.  The transaction has come out
  of `WAITING` (or `COMMIT`) and is in flight again.

- **`COMMITTED`** - a transit state.  The transaction has executed
  its commit work and is heading toward exit.

- **`PAUSED`** - the *scheduler pause*.  A general-purpose park
  point applicable at any point along the lifecycle: the scheduler
  can request a transaction park here by setting `tx.pause = True`,
  and the transaction won't continue until every party that set
  the flag has cleared it.  Used heavily by the high-level helpers.

- **`EXITING`** - a transit state.  The transaction is on its final
  flight to terminal.

- **`RETURNED`** - terminal.  The method returned normally.

- **`RAISED`** - terminal.  The method raised an exception.

### Method States

This subsection documents, primitive by primitive, anything unusual
about each method's transaction.  Methods not listed are mundane
(they pass through the lifecycle without surprises).

#### Lock

- `acquire(blocking=True, timeout=-1)`: timeout-bearing
  (visits `COMMIT`).  Has a real `WAITING` when the lock is
  contended: the transaction is asleep inside the real
  `threading.Lock.acquire`, waiting for the lock to become
  available.  Wakes on lock availability or timeout.
- `release()`: passes through without parking after the scheduler
  block; doesn't visit `WAITING`, `COMMIT`, or `STALLED`.

#### RLock

The same as `Lock`, with reentrancy handled by the underlying real
`threading.RLock`.

#### Condition

- `acquire`, `release`: the same as `Lock.acquire`/`Lock.release`.
- `wait(timeout=None)`: timeout-bearing (visits `COMMIT`).  Has a
  real `WAITING` (asleep inside `threading.Condition.wait`, with
  the underlying lock dropped).  Visits `STALLED` post-wake, before
  the internal lock re-acquire is allowed to proceed.  The lock
  re-acquire is a nested transaction; while it's running,
  `Nested(wait_tx)` signals high.
- `wait_for(predicate, timeout=None)`: timeout-bearing.  Always
  nests at least one `Condition.wait` transaction inside.  If the
  user-supplied `predicate` calls primitive methods, those become
  nested transactions too.
- `notify(n=1)`, `notify_all()`: pass through after the scheduler
  block.

#### Semaphore

- `acquire(blocking=True, timeout=None)`: timeout-bearing.  Has
  a real `WAITING` when the semaphore counter is zero.
- `release(n=1)`: passes through.

#### BoundedSemaphore

The same as `Semaphore`, except `release` raises if it would
exceed the initial value.

#### Event

- `wait(timeout=None)`: timeout-bearing.  Has a real `WAITING`
  while the event isn't set.
- `is_set()`, `set()`, `clear()`: pass through.

#### Barrier

- `wait(timeout=None)`: timeout-bearing.  Has a real `WAITING`
  for the first `parties - 1` arrivers, until the last party
  arrives.  If the barrier was constructed with an `action`
  callback, the final arrival's transaction (the *opener*)
  pushes itself off its thread's transaction chain before running
  the action.  While pushed, `Action(opener_tx)` signals high,
  and any primitive calls made by the action appear as fresh
  root transactions rather than children of the opener.  This
  means signal tokens like `Nested(opener_tx)` do *not* fire
  during the action--use `Action(opener_tx)` instead.

  Cycle validation: constructing a `Cycle` on a `Barrier` with
  a `scheduler` argument (asking the cycle to drive an
  externally-supplied scheduler-cycle for the action) requires
  the barrier to have been built with an `action`.  Without one,
  the constructor raises.
- `reset()`, `abort()`: pass through.

## Changelog

**0.1** *2026/05/14*

- Initial release!
