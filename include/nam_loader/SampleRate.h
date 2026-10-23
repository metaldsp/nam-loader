// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

// Sample-rate architecture of the library.
//
// NAM A2 models are trained at 48 kHz, so instead of resampling per model
// (and paying the round trip between every model at a non-48k host), the
// whole processing graph is wrapped in ONE stereo resampling boundary
// (RateBoundary), with an optional power-of-two Oversampler nested inside it:
//
//   host rate ─▶ [RateBoundary in] ─▶ 48 kHz ─▶ [Oversampler ×N up]
//                  ─▶ models at 48·N kHz ─▶ [÷N down] ─▶ [RateBoundary out]
//
// Linear processing that gains nothing from a higher rate (convolution, for
// one) can stay at the base rate inside the oversampled region through
// Oversampler::processAtBaseRate: it decimates, runs the callback at 48 kHz
// and interpolates back, so its CPU cost and sound are identical at every
// oversampling factor (oversampling a convolution buys nothing and costs
// roughly quadratically).
//
// Consequences, all deliberate:
//  - At a 48 kHz host with oversampling off, both layers are skipped
//    entirely: zero latency, zero cost, bit-identical to processing directly.
//  - At any other host rate the boundary engages regardless of what runs
//    inside it (even nothing), so the reported latency is a constant per host
//    rate and adding or removing models never changes it mid-session. The
//    oversampler is minimum-phase (zero reported latency), so toggling it
//    never changes the latency either.
//  - Signals that are mixed together inside the graph (a dry/wet blend, say)
//    keep identical alignment, with no comb filtering from per-stage resampler
//    latency, and stereo channels stay sample-aligned with each other.
//
// The boundary always targets the *base* rate; the oversampler raises the rate
// from there, so its ratios stay integer powers of two regardless of the host
// rate.

namespace namloader {

// The rate models run at (A2 models are trained at 48 kHz); the host rate is
// converted to it once (RateBoundary), then optionally oversampled by a power
// of two (Oversampler). The effective model rate is this times the current
// oversampling factor.
inline constexpr double kBaseSampleRate = 48000.0;

}  // namespace namloader
