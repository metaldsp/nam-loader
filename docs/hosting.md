<!--
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini

SPDX-License-Identifier: MIT
-->

# Hosting runtimes

NamLoader gives you a `Runtime`: one model, ready to run. It deliberately does
not decide when a `Runtime` is built, how it reaches the audio thread, or how
one replaces another. That is the host's job, and the library does not
implement it.

This page describes the pattern Ampira, the library's original host, uses. It
is a recommendation with concrete numbers and the reasons behind them, not an
API. Where the library supports the pattern, it says so.

For terms, see [CONTEXT.md](../CONTEXT.md). For how a host sets the gain flags
on each call, see [calibration.md](calibration.md).

## Why a runtime is never edited in place

A `Runtime` is built for fixed facts (`RuntimeSpec`): the oversampling factor,
the maximum block size, the channel count and the slimmable size. The model is fixed at
construction too. So any change of
these is a new `Runtime`, never a mutation of the running one.

Rebuilding the live engine behind a mute or a dip in the wet level was
considered and rejected in Ampira: it cannot cover an edit that reshapes the
whole signal path, and the gap is audible. The pattern instead is to build the
replacement beside the running one and swap them with a true crossfade.

## The lifecycle

1. **Build** off the audio thread.
2. **Publish** to the audio thread without a lock.
3. **Pre-roll** the new runtime, muted, on the live input.
4. **Crossfade** from the old runtime to the new one.
5. **Retire** the old runtime off the audio thread.

### What forces a rebuild

Anything in `RuntimeSpec`, and the model itself:

- a new model;
- a change of oversampling factor;
- a change of channel count (for example mono to stereo input);
- a change of slimmable size.

Anything that travels in `ProcessContext` does not rebuild: input
calibration, its dBu level, normalization, the target loudness and
`calibratedHandOff` are read on each call, and the post-model gain glides to
the new value over 50 ms. Power and bypass belong to the host's own layer.

A change of host sample rate or maximum block size changes the sizes the
runtime was built for, so rebuild then too. This normally happens when the
host prepares audio, where there is nothing playing to crossfade from; build
synchronously and install the runtime with no fade. If models are still
loading at that point, the first runtime can fade in from silence once they
are ready.

### Build

`loadModel` and `Runtime::create` allocate, parse and prewarm, so they run on
a loader or pool thread. A `Model` is immutable and shared, so building a
second `Runtime` for the same model costs instantiation, not a re-parse.

Size the build from the boundary:

- Set `RuntimeSpec::maxBlockSize` to `RateBoundary::getMaxBaseBlockSize()`
  times the oversampling factor. The boundary can hand the base-rate function
  slightly more frames than the plain rate ratio suggests, and `process` must
  never be given more than `maxBlockSize` frames.
- Use one `RateBoundary` for the whole graph, not one per runtime, so the old
  and new runtimes are heard through the same resampler during the fade.

`Runtime::create` returns null on an unusable spec or a failed build. In that
case keep playing the current runtime, log once, and retry on the next change.

If edits arrive faster than builds finish (a slider dragged through a
threshold that forces a rebuild), debounce. Ampira waits 50 ms, builds only the
latest request, cancels a build that was superseded while it ran, and posts only
the newest completed one.

### Publish

Hand the finished runtime to the audio thread through a lock-free single-slot
mailbox: an atomic pointer that the building thread exchanges and that only the
audio thread empties.

- A newer runtime replaces one that has not been adopted yet. The latest wins.
  The displaced one was never seen by the audio thread, so the publisher can
  destroy it on its own thread.
- The audio thread adopts a posted runtime at the start of a block, never in
  the middle of a fade. Edits that arrive during a transition wait, and only
  the last counts.
- Adopt a runtime only when the retire queue has room (see below). Otherwise
  leave it in the mailbox; a newer post still replaces it.
- Refuse a runtime older than the one playing, which can only be a build that
  lost a race. Ampira tags each build with a monotonic generation.

### Pre-roll

Before a new runtime is audible, run it on the live input with its output
discarded. The model's receptive field has to fill first: a NAM model starts
with an empty history, and until it has seen enough input its output does not
reflect the signal. Fading in an unfilled model would play that start-up
transient.

`Runtime::getPrewarmSamples()` returns the receptive field in samples at the
oversampled rate (the model's prewarm count times the phase count). To
compare it with a time budget, divide by `48000 * factor`.

Ampira clamps the pre-roll to between 20 ms and 400 ms (counted at the base
rate). The pre-roll costs a muted runtime's worth of processing on top of the
playing one, so the ceiling bounds that cost for an unusually long receptive
field. Ampira's figure is the largest receptive field in the whole processing
graph, together with the delay lines of its parallel-path alignment stages, so
a crossfade never plays an unfilled delay line; for a bare `Runtime` the
receptive field is the thing to cover.

`Runtime.ReportsPrewarmSamples` shows the figure for phase-safe and
non-phase-safe models.

### Crossfade

After the pre-roll, run both runtimes on the same input for 40 ms and blend
their outputs per sample:

```
out = old * cos^2(t) + new * sin^2(t),   t from 0 to pi/2 across the fade
```

The two weights sum to 1. This is an equal-gain crossfade rather than an
equal-power one, which is right because old and new process the same input and
their outputs are strongly correlated: summing them with weights that add to
1 keeps the level steady, where equal-power weights would bump it in the
middle. Count the fade in base-rate samples inside the `RateBoundary`
callback, where blocks can have varying sizes, so the same resampler carries
old and new.

Why a true crossfade, not a cut: a cut jumps between two different waveforms
and clicks, and a cut or short mute in the wet signal is audible as a gap. A
true crossfade lets the old sound complete while the new one takes over.

Other points to settle during the fade:

- **Levels.** The new `Runtime` starts its post-model gain at the value it
  will hold, so it does not glide in from unity beside the old one
  (`Runtime.FirstBlockStartsAtTheGainItWillHold`).
- **Latency.** All runtimes of one instance must report the same latency, so a
  swap never shifts the host's delay compensation. A `Runtime` and the
  `Oversampler` add none; the `RateBoundary` latency is constant for a given
  host rate.
- **Meters and state outside the runtime.** Ampira switches its audible
  output format at the start of the fade, and publishes meters from the old
  runtime until the midpoint and from the new one after it.
- **The first runtime.** With nothing to crossfade from, fade it in from
  silence.

Expect CPU to roughly double while two runtimes overlap, plus the pre-roll's
muted run. Any state a runtime does not own (a reverb tail, say) starts empty
in the replacement; a NAM model is warmed over its receptive field, not over
arbitrary history.

### Retire

Never destroy a `Runtime` on the audio thread: destruction frees memory. When
a fade finishes, push the old runtime into a fixed-capacity single-producer
queue, and drain it from a non-audio thread, such as the message thread's
housekeeping timer. Ampira uses 8 slots, a power of two. Because it adopts a
runtime only while a slot is free, and each fade retires exactly one, a push
cannot find the queue full. The drain must be serialized between callers,
because the queue has a single consumer and two concurrent drains would delete
the same runtime twice.

## Failure handling

An engine exception never escapes `Runtime::process`. The first exception
restores that block's dry input and sets `hasFailed()`; every later call
leaves the signal as it came in
(`Runtime.EngineExceptionPassesDry`,
`Runtime.EngineExceptionRestoresTheDryInputAfterCalibration`).

The flag is a plain `bool`, written on the audio thread. Read it from the audio
thread after `process`, and report it to the rest of the host through your own
atomic. Then, off the audio thread:

- mark the model as failed and tell the user;
- do not rebuild the same model in a loop;
- build a replacement without it (or with another model) and publish it
  through the same path.

The same applies to build failures: `Runtime::create` returning null leaves
the audible runtime playing, and the next change retries.

## Checklist

- `loadModel`, `Runtime::create` and all destruction: off the audio thread.
- `maxBlockSize` is `getMaxBaseBlockSize() * factor`.
- Publish with an atomic exchange; latest wins; adopt at block starts.
- Pre-roll by the receptive field, clamped to your own bounds, muted.
- Crossfade 40 ms with weights that sum to 1; equal latency for all runtimes.
- Retire through a bounded queue; drain off the audio thread.
- Poll `hasFailed()` on the audio thread and replace the runtime.
