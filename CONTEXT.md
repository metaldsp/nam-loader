<!--
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini

SPDX-License-Identifier: MIT
-->

# NamLoader

NamLoader loads Neural Amp Modeler (NAM) captures and runs them in real time,
with oversampling, dual-mono voices and optional multi-core processing, inside
a host's audio graph.

## Language

### Models

**Model**:
One NAM capture, decoded and validated once by `loadModel`. It is immutable and
shared; every `Engine` or `Runtime` builds its own instances from it.
_Avoid_: Tone, preset, patch

**NAM capture**:
A model of how captured audio gear responds to an input signal, including its
nonlinear behavior. The gear can be an amplifier, a pedal or a complete rig.
_Avoid_: Impulse response, amp-only model

**A2 model**:
A WaveNet capture in NAM's A2 shape, trained at 48 kHz, or a slimmable
container made only of such captures. Reported by `Model::a2`.
_Avoid_: Latest model, fast model

**Slimmable size**:
The tier a slimmable A2 container runs at, from 0.0 to 1.0. A two-tier
container maps `[0, 0.5)` to lite and `[0.5, 1.0]` to full, because NAM assigns
the boundary value to the tier above. The lite request is therefore 0.0, not
0.5, which would select full. It has no effect on a model that is not
slimmable.
_Avoid_: Quality, model size

### Rates

**Base rate**:
48 kHz (`kBaseSampleRate`), the rate NAM A2 models are trained at and the rate
the whole processing graph is converted to once.
_Avoid_: Native rate, model rate

**Rate boundary**:
The single pair of conversions, host rate to base rate and back, that wraps the
whole processing graph (`RateBoundary`). It is skipped at a 48 kHz host. Its
latency is constant for a given host rate. Anything inside it is sized from
`RateBoundary::getMaxBaseBlockSize()`.
_Avoid_: Resampler (per model), sample-rate converter

**Oversampling**:
Running models at 2, 4 or 8 times the base rate (the factor) so the harmonics
of their nonlinear behavior land above the base Nyquist, where the oversampler
removes them instead of letting them alias. The factor is fixed per
`Runtime`.
_Avoid_: Resampling, upsampling

**Phase** (phase-interleaving):
One of the N independent copies of a model that together process an
oversampled stream, each taking every Nth sample at the base rate. This is
mathematically the same as the model run at N times the rate with its
dilations scaled by N, but needs no change to the NAM core. The receptive field
stays constant in seconds.
_Avoid_: Channel, thread, band

**Phase-safe**:
A model whose architecture can be phase-interleaved: WaveNet, ConvNet and
Linear, or a container made only of those (`Model::phaseSafe`). Others, such as
LSTM, run as one instance time-scaled to the oversampled rate.
_Avoid_: Thread-safe, oversampling-safe

**Voice** (dual mono):
One independent signal path through a model. A two-channel `Runtime` has two
voices, one per channel, each with its own set of phase instances, so a stereo
input is never collapsed to mono.
_Avoid_: Channel, instance, stereo mode

**Prewarm** (receptive field):
The number of samples a model needs before its output reflects its input
rather than its empty history (`Model::prewarmSamples`, in model samples).
`Runtime::getPrewarmSamples()` reports it at the oversampled rate, which is how
long a host pre-rolls a new `Runtime` before using it.
_Avoid_: Warm-up, latency

### Calibration and listening level

**Input calibration**:
The host's interface-level reference, in dBu, that relates digital input to the
physical signal level calibrated captures expect. It drives each model at the
level of its capture rig.
_Avoid_: Rig calibration, model calibration, loudness normalization

**Model calibration**:
The input and output reference levels a NAM capture carries
(`Model::inputLevelDbu`, `outputLevelDbu`), describing the physical levels of
the captured gear. Either may be absent.
_Avoid_: Input calibration, loudness normalization

**Calibrated hand-off**:
The gain a model applies after itself when input calibration is on and another
enabled model follows it on the signal path, set by
`ProcessContext::calibratedHandOff`. It carries the capture's output level
into the later model, as if the two devices were chained. The last model on
every path never has one.
_Avoid_: Loudness normalization, automatic volume matching

**Loudness normalization**:
A model's optional output adjustment toward a common listening level
(`targetLoudnessDb`) using its loudness metadata, clamped to +/-12 dB. Where a
calibrated hand-off applies, the hand-off takes precedence.
_Avoid_: Input calibration, calibrated hand-off

### Concurrency

**Worker pool**:
The real-time fork/join helper threads (`WorkerPool`) that run a model's phases
and voices in parallel. The audio thread passes it per call; output is
bit-identical to the serial schedule.
_Avoid_: Thread pool, job system
