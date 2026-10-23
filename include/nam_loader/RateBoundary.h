// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>

#include "nam_loader/SampleRate.h"

namespace namloader {

// The host-rate <-> kBaseSampleRate resampling boundary described in SampleRate.h: a stereo
// Lanczos resampler (AudioDSPTools, the engine the official NAM plugin uses for this job) that
// wraps the whole processing graph. It engages only when the host rate differs from 48 kHz.
// Held behind a pimpl so this header does not pull in the resampler.
//
// Typical use: prepare() once per host configuration, report getLatencySamples() to the host,
// then wrap each host block in process(). Size anything that runs inside the callback from
// getMaxBaseBlockSize() (times the oversampling factor for RuntimeSpec::maxBlockSize).
class RateBoundary {
public:
  using ProcessFn = std::function<void(float** inputs, float** outputs, int numFrames)>;

  RateBoundary();
  ~RateBoundary();
  RateBoundary(const RateBoundary&) = delete;
  RateBoundary& operator=(const RateBoundary&) = delete;

  // Off the audio thread; allocates. Engages only when hostSampleRate differs from
  // kBaseSampleRate (by more than 0.1 Hz); otherwise it releases any previous resampler. Warms
  // the shared Lanczos tables and measures the latency. Always starts from a fresh resampler,
  // so calling it again is also a full reset.
  void prepare(double hostSampleRate, int maxHostBlockSize);

  bool isActive() const noexcept;

  // Host samples; 0 when inactive.
  int getLatencySamples() const noexcept;

  // The most frames one baseRateFn call can see:
  // ceil(maxHostBlockSize / (hostSampleRate / kBaseSampleRate)), or the host block size when
  // inactive. The resampler can hand over slightly more than the plain rate ratio suggests.
  int getMaxBaseBlockSize() const noexcept;

  // Stereo. Inactive: calls baseRateFn(inputs, outputs, numFrames) directly. Active: resamples
  // `inputs` to the base rate, calls baseRateFn there (possibly several times, with any frame
  // count up to the bound above), and resamples the result into `outputs`. `inputs` and
  // `outputs` may alias. Real-time safe after prepare(), provided the caller builds the
  // ProcessFn once, off the audio thread, and passes that same object each call: converting a
  // lambda to std::function at the call site can allocate. This call never copies it.
  //
  // A host that exceeds the block size it promised is fed through in prepared-size slices. If
  // the resampler's invariants break or baseRateFn throws, nothing escapes: the outputs are
  // cleared instead, so a failure is silence rather than an exception into the host.
  void process(float** inputs, float** outputs, int numFrames, const ProcessFn& baseRateFn);

  // Clears the resampler history, as if freshly prepared. Off the audio thread: it rebuilds the
  // resampler (Reset() on an unchanged configuration would skip the silent prefill that makes
  // the first block yield enough samples and fixes the delay).
  void reset();

  // The Lanczos kernel tables are function-level statics filled by the first resampler
  // constructed, with no synchronisation. Call this once, off the audio thread, before any
  // prepare() that may run concurrently with another (prepare() calls it too, which makes the
  // first call the racy one). Idempotent and thread-safe.
  static void warmTables();

  // Physical group delay of the boundary at this host rate, in host samples (the resampler's own
  // GetLatency() under-reports it). Allocates; call off the audio thread.
  static int measureLatency(double hostSampleRate, int blockSize);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  double hostRate = kBaseSampleRate;
  int hostMaxBlock = 0;
  int latency = 0;
  int maxBaseBlock = 0;
};

}  // namespace namloader
