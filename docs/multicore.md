<!--
SPDX-FileCopyrightText: 2026 TONE3000
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
SPDX-License-Identifier: MIT
-->

# Multi-core processing: the realtime worker pool

`WorkerPool` spreads independent work across a pool of realtime helper
threads. Passing a pool to `Engine::process` (or through
`ProcessContext::pool` to `Runtime::process`) turns multi-core processing on;
passing none runs everything serially on the calling thread. Whether to
create a pool, and whether to expose it as a user setting, is the host's
call. Two kinds of work fork:

- **NAM voices and oversampling phases.** An oversampled `Engine` holds N
  independent instances of the same model over disjoint per-phase buffers
  (see [oversampling.md](oversampling.md)), and a stereo engine holds two
  voices, one per channel, each with its own full set of phase instances (a
  mono engine has one voice). The engine forks the flat voice by phase set as
  one group (at most 2 x 8 = 16 jobs, the pool's `kMaxJobs`), so an 8x stereo
  model still costs about one native-rate model per core instead of sixteen
  on one. The voices are as independent as the phases (separate models,
  separate buffers), and `EngineTest.PhaseForkMatchesSerialBitExact` and
  `EngineTest.TwoVoiceForkMatchesSerialBitExact` pin the engine fork as
  bit-identical to the serial loop, at every factor the suite is
  instantiated for. `Runtime.StereoNamForksAcrossThePoolBitIdentically`
  pins the same through a `Runtime`.
- **Whatever else the host can split.** The pool is a general fork/join
  primitive for jobs with disjoint state, and a host can use it to run
  independent signal paths side by side. Ampira, the original host, forks the
  two branches of its parallel sections this way: one branch goes to a worker,
  the other runs on the audio thread, and both join before the sections
  merge. That fork belongs to the host, not the library. It does interact with
  the engine fork, though: a host job can itself call `Engine::process` with
  the pool, and the engine then forks its voices and phases from inside that
  job (see nesting below).

Implementation is `include/nam_loader/WorkerPool.h` (header-only). The
contract is pinned by `tests/worker_pool_tests.cpp` (the scheduling
primitive: every job exactly once, nesting, degradation, churn), while
`tests/engine_tests.cpp` and `tests/runtime_tests.cpp` check that the audio
is bit-identical to serial.

## Design rules

The pool is an extension of the audio callback, not a general scheduler:

- Workers only ever run strictly inside a `forkJoin()` that the forking
  (audio) thread is waiting in. Neither the audio thread nor a job takes a
  lock: the state the jobs touch is owned by the code that forked them, which
  is why the host must not swap or destroy that state while a block is being
  processed. A host that replaces a runtime builds the new one elsewhere and
  swaps it in between blocks, so workers never see a runtime change under
  them.
- A job is a plain function pointer plus a context struct on the forking
  thread's stack (alive until `forkJoin()` returns). Forking allocates
  nothing.
- `forkJoin(fn, ctxs, count)` runs every job exactly once and returns only
  when all are complete. The caller always executes job 0 itself, so
  completion never depends on worker liveness (`WorkerPool.RunsEveryJobExactlyOnce`,
  `WorkerPool.RunsInlineWithoutWorkers`).
- Nesting is supported one level deep (a host job forking an `Engine`'s
  voices and phases), and stays deadlock-free by construction: a joiner never
  parks, it claims and runs its own group's unclaimed jobs inline and only
  spins on jobs another thread is actively executing, so every waiter makes
  progress itself (`WorkerPool.NestedForkJoinRunsEveryLeaf`).

## Job protocol

All lock-free, one atomic state per slot in a fixed registry (slots are pool
memory, never freed; contexts on caller stacks are only dereferenced while
the owning `forkJoin` is still on that stack):

```
Free --publisher CAS--> Building --> Armed --worker CAS--> Claimed --> Done
                                       |                                |
                                       +--join CAS -> run inline -> Free <-+ (join recycles)
```

The Armed-to-Claimed transition is a compare-exchange raced between the
workers and the publisher's join loop: whoever wins runs the job. That race
is the safety valve. If every worker is descheduled or busy, the joiner
steals its jobs back and runs them inline, so the callback degrades to
exactly the serial cost instead of stalling. A full registry (deep
concurrent nesting) makes `forkJoin` run the overflow inline immediately:
less parallel, never blocked (`WorkerPool.RegistryOverflowFallsBackInline`).
This is why multi-core is pure scheduling: every schedule runs the same code
on the same buffers, and the output is bit-identical regardless.

## Scheduling and wakeups

The threads run at realtime priority (time-constraint on macOS, MMCSS
"Pro Audio" on Windows); the pool sizes itself to one worker per spare
hardware core. Linux refuses realtime scheduling to processes without
rtprio privileges, so there the workers fall back to normal
highest-priority threads: still parallel, and the join steal absorbs any
scheduling misses. On macOS workers also join the audio device's workgroup
(`os_workgroup`) when the host provides one, through
`WorkerPool::setAudioWorkgroup`; without that, Apple Silicon parks them on
efficiency cores and joins miss deadlines. Workgroup tokens are thread-affine,
so each worker re-joins from its own loop whenever the device changes.

Wakeups are precise, not broadcast: parked workers advertise themselves in
a bitmask and `forkJoin` signals only as many of them as it published jobs.
Broadcasting to the full pool costs more scheduler time than the forked
work saves (a dozen realtime threads wake to find the one job already
claimed), enough to cancel the fork's speedup outright on a 14-core
machine. The park/publish race is closed with a seq_cst fence pairing
(worker: set bit, then re-check for armed jobs; publisher: arm jobs, then
read the mask), and the join steal bounds a doubly-missed wakeup to one
serial-cost block. Parked workers also re-check on a 100 ms timeout as a
last resort. That timeout was 1 ms originally, and with every wake
signalled explicitly it only added kernel time: 13 parked workers timing
out 1,000 times a second cost about 11% of a core while the plugin was
idle, more than the neural net itself. Profiled on a 14-core M-series Mac
with `ps -M`: the worker threads showed ~0.8% system time each and ~0.1%
user time; at 100 ms they show 0.0%.

One easy-to-miss detail: FTZ/DAZ denormal flags are per-thread CPU state.
Each worker sets `ScopedNoDenormals` in its own loop; without it, NAM decay
tails hit denormal range at roughly 100x cost on workers, and the phase
fork would break bit-identity with the serial loop, which runs under the
audio thread's flags. A host that forks its own jobs through the pool gets
this for free, but the audio thread must run with the same flags for the
bit-identity to hold.

## Failure containment

Exceptions can't cross threads, so a NAM phase job traps its own and the
joining thread rethrows once every phase has completed (`Engine::process`
throws `std::runtime_error` on the calling thread). `Runtime::process` catches
it: it restores the dry input, passes it through from then on, and sets
`hasFailed()` (`Runtime.EngineExceptionPassesDry`). The same containment
applies when the engine runs serially, so a forked throw behaves exactly like
a serial one. What to do about a failed runtime (bypass it, report it, retire
it) is the host's job.

## Without a pool

With no `WorkerPool` (or a pool started with zero workers) nothing forks and
`Engine::process` runs its voices and phases sequentially on the audio thread.
Oversampled engines still work, just on one core. A host that forks its own
independent paths decides separately whether each fork is worthwhile; the
library only guarantees that either choice gives the same output.
