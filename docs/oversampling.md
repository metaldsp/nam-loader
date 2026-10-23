<!--
SPDX-FileCopyrightText: 2026 TONE3000
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
SPDX-License-Identifier: MIT
-->

# Oversampling: the rate raiser

Oversampling runs the models at 2x, 4x, or 8x the 48 kHz base rate
(`kBaseSampleRate`). The point is aliasing: NAM models are nonlinear, so they
generate harmonics, and any harmonic above Nyquist folds back into the audible
band as an inharmonic artifact. Raising the rate gives those harmonics
somewhere legal to land; the decimation filter then removes everything above
the base Nyquist on the way back down.

The factor is chosen per `Runtime` (`RuntimeSpec::oversampleFactor`) when it
is built, and the `Engine` and the `Oversampler` wrapped around it must agree
on it. Changing the factor therefore means building a new `Runtime` and
`Oversampler` for the new rate and swapping them in, rather than re-preparing
a live engine; publishing and crossfading that swap is the host's job. Ampira,
the original host, does exactly that: each prepared program owns its own
oversampler and crossfades to a rebuilt one when the user changes the factor.
Everything a host wraps in the oversampled region runs at the same rate, so
time-based parameters mean the same thing at every factor, and the whole graph
still crosses the host-rate boundary once
([ADR 0001](adr/0001-one-resampling-boundary.md)).

Implementation lives in `include/nam_loader/Oversampler.h` (header-only) and
`include/nam_loader/Engine.h` / `src/Engine.cpp`. Coverage is in
`tests/oversampler_tests.cpp` and `tests/engine_tests.cpp` (null,
transparency, aliasing, interleaving exactness, base-rate wrapping).

## The filter

Each doubling is one polyphase half-band stage built from two cascades of
six first-order allpass sections each (the classic HIIR structure). Design
properties, and why they matter here:

- **Minimum-phase, zero reported latency.** Group delay is a fraction of a
  sample across the audible band, so toggling oversampling never changes the
  latency a host reports for delay compensation
  (`Oversampler::getLatencySamples()` is always 0; see
  `Oversampler.ReportsStateAndZeroLatency` and
  `Oversampler.PassbandTransparentWithTinyGroupDelay`). Linear-phase FIR
  half-bands would add latency and cost more.
- **Cheap.** 24 multiply-adds per branch pair per sample, with >90 dB
  stopband rejection.
- **Fixed, not user-selectable.** The phase warp near the base Nyquist is
  inaudible; offering filter choices would be a knob without a sound.

Coefficients are adapted from
[DLC86/NAM-Oversampler](https://github.com/DLC86/NAM-Oversampler)'s
AudioDSPTools fork (MIT), following the published HIIR half-band allpass
tables. That project pioneered oversampled NAM processing, and this design
follows its lead in several places (including the time-scaled handling of
LSTM models noted below).

## Phase-interleaved NAM

Running a WaveNet/ConvNet model at N times its native rate normally requires
scaling its convolution dilations by N. We don't patch NeuralAmpModelerCore;
instead we use an identity: an oversampled model with scaled dilations is
mathematically equal to N independent copies of the unscaled model, each
processing every Nth sample of the oversampled stream. Every scaled dilation
lands its taps exactly N samples apart (within one phase), and the 1x1
convolutions and activations are per-sample, so nothing crosses phases. See
[ADR 0002](adr/0002-phase-interleaved-nam-oversampling.md) for the decision.

`Engine` therefore holds N instances of the same model and interleaves
them. The receptive field stays constant in seconds, so the model sounds the
same; only the alias behavior improves.
`Oversampler.PhaseInterleavedDilatedConvIsExact` pins the identity
sample-exactly, and `EngineTest.PhaseInterleaveMatchesManualPhaseReference`
pins the engine's interleaving against a hand-built phase split.

The phase instances are fully independent (separate models, disjoint
per-phase buffers), which is also what makes them parallelizable: given a
`WorkerPool`, `Engine::process` forks the phase loop across the realtime
workers instead of running it sequentially (see [multicore.md](multicore.md)).
Scheduling only; `EngineTest.PhaseForkMatchesSerialBitExact` pins forked
output bit-identical to serial.

The same slot layout carries stereo: the engine holds `voices x phases`
instances (voice-major). A mono engine has one voice and a stereo one has two
(`Engine::kMaxVoices`), each voice owning a full phase set fed from its own
channel and sharing the model and settings. The voice count is fixed when the
engine is created (`RuntimeSpec::channels` for a `Runtime`), so changing it
means building a new engine; it never changes with power, mix or silence.
Phase and voice slots are equally independent, so the fork is one flat loop
over all of them; `EngineTest.TwoVoicesProcessStereoChannelsIndependently`
pins each voice's output against a single-voice engine on that channel, and
`EngineTest.TwoVoiceForkMatchesSerialBitExact` pins the forked two-voice
schedule against the serial one.

LSTM models update state on consecutive samples and can't be phase-split;
they get a single instance run time-scaled at the full oversampled rate
(`Model::phaseSafe` is false for them;
`Engine.CreateModelThatIsNotPhaseSafeRunsOneTimeScaledInstance`). In
practice this is a defensive path for A2 WaveNet captures, which are always
phase-safe.

## Linear processing at the base rate

Convolution is linear: it creates no harmonics, so oversampling it buys
nothing and costs roughly quadratically (kernel length and rate both scale
with the factor). The library does not load impulse responses, but
`Oversampler::processAtBaseRate()` is the tool for exactly this case, with IR
convolution as the motivating example. Called from the oversampled region, it
decimates the stream down to 48 kHz, runs the callback you give it there (a
convolver, say), and interpolates back. The cost and the sound of that
section are then identical at every factor
(`Oversampler.AtBaseRateFrameCountsAndSignalPath`; at factor 1 it is a
bit-exact direct call, `Oversampler.AtBaseRateFactor1IsBitExact`). Each
wrapped section owns its own `Oversampler`, because the filter state is shared
between directions: an instance serves one role.

## Frame-count contract

Every base-rate input frame becomes exactly `factor` oversampled frames, and
the decimator never carries a partial pair across blocks
(`Oversampler.FrameCountContract`). This is what lets the phase-interleaved
engine assume buffers divisible by the factor, and it is asserted rather than
handled: a violation is a bug upstream, not a runtime condition.
