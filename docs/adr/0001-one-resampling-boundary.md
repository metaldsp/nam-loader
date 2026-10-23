<!--
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
SPDX-License-Identifier: MIT
-->

# The whole processing graph runs at one fixed rate behind a single resampling boundary

Everything between the input and output stages runs at `kBaseSampleRate` (48 kHz) times the oversampling factor, wrapped in one stereo `RateBoundary` that converts to and from the host rate, rather than resampling inside each model stage. NAM A2 models are trained at 48 kHz and other processing, such as an IR convolver, runs at whatever rate it is prepared with, so per-stage resampling would pay a round trip between every NAM stage at a non-48 kHz host.

## Consequences

- At a 48 kHz host with oversampling off, both the boundary and the `Oversampler` are skipped: zero latency, bit-identical to processing directly (`RateBoundary.InactiveAt48kIsADirectCall`, `Oversampler.Factor1IsBitExactPassthrough`).
- At any other host rate the boundary engages even for an empty graph, so reported latency is constant per host rate (`RateBoundary.ReportedLatencyMatchesMeasuredDelay`) and adding or removing stages never changes the delay a host compensates for mid-session. The oversampler is minimum-phase with zero reported latency, so toggling oversampling never changes it either.
- Signals mixed together inside the graph, such as a dry/wet blend, have identical alignment (no comb filtering from per-stage resampler latency), and parallel paths stay sample-aligned with each other. Ampira, the original host, relies on this for its parallel sections and their alignment stage, which run inside the same domain and add no boundary of their own.
- Linear processing such as IR convolution is the one exception inside the oversampled region. Convolution creates no harmonics, so oversampling it buys nothing and costs roughly quadratically; `Oversampler::processAtBaseRate` decimates to the base rate, runs the processing there, and interpolates back, which keeps its sound and CPU identical at every factor.

Details live in `include/nam_loader/SampleRate.h`, `include/nam_loader/RateBoundary.h` and [docs/oversampling.md](../oversampling.md). The decision to run NAM models phase-interleaved inside this domain is [ADR 0002](0002-phase-interleaved-nam-oversampling.md).
