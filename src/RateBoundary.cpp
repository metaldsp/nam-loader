// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "nam_loader/RateBoundary.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <cmath>
#include <mutex>

// AudioDSPTools was written against a plugin framework and expects two of its names: the
// default block size and the PI constant. Defined here so they stay out of the public headers.
#ifndef DEFAULT_BLOCK_SIZE
#define DEFAULT_BLOCK_SIZE 1024
#endif

namespace iplug {
constexpr double PI = 3.14159265358979323846;
}

#include "dsp/ResamplingContainer/ResamplingContainer.h"

namespace namloader {
namespace {

// Stereo Lanczos (filter size 12) between the host rate and the base rate. Only instantiated
// when the host rate differs from 48 kHz.
using Resampler = dsp::ResamplingContainer<float, 2, 12>;

}  // namespace

struct RateBoundary::Impl {
  Impl() : resampler(kBaseSampleRate) {}
  Resampler resampler;
};

RateBoundary::RateBoundary() = default;
RateBoundary::~RateBoundary() = default;

void RateBoundary::warmTables() {
  static std::once_flag once;
  std::call_once(once, [] {
    Resampler warm(kBaseSampleRate);
    warm.Reset(44100.0, 64);
  });
}

void RateBoundary::prepare(double hostSampleRate, int maxHostBlockSize) {
  warmTables();

  hostRate = hostSampleRate;
  hostMaxBlock = std::max(1, maxHostBlockSize);
  if (std::abs(hostSampleRate - kBaseSampleRate) > 0.1) {
    impl = std::make_unique<Impl>();
    impl->resampler.Reset(hostRate, hostMaxBlock);
    // Not GetLatency(): that under-reports by the Lanczos kernels' group delay, and hosts align
    // dry paths against this number.
    latency = measureLatency(hostRate, hostMaxBlock);
    maxBaseBlock = static_cast<int>(
        std::ceil(static_cast<double>(hostMaxBlock) / (hostRate / kBaseSampleRate)));
  } else {
    impl.reset();
    latency = 0;
    maxBaseBlock = hostMaxBlock;
  }
}

bool RateBoundary::isActive() const noexcept {
  return impl != nullptr;
}

int RateBoundary::getLatencySamples() const noexcept {
  return latency;
}

int RateBoundary::getMaxBaseBlockSize() const noexcept {
  return maxBaseBlock;
}

void RateBoundary::reset() {
  if (impl == nullptr)
    return;
  impl = std::make_unique<Impl>();
  impl->resampler.Reset(hostRate, hostMaxBlock);
}

void RateBoundary::process(float** inputs,
                           float** outputs,
                           int numFrames,
                           const ProcessFn& baseRateFn) {
  if (numFrames <= 0)
    return;

  try {
    if (impl == nullptr) {
      baseRateFn(inputs, outputs, numFrames);
      return;
    }

    // ProcessBlock takes its callback by value. A std::function holding only a reference to
    // ours copies without allocating; copying baseRateFn itself would allocate whenever its
    // capture outgrows the small-buffer.
    const auto viaReference = std::ref(baseRateFn);
    // The container's scratch is sized for hostMaxBlock, so an oversize host block is fed
    // through in slices.
    for (int offset = 0; offset < numFrames; offset += hostMaxBlock) {
      const int length = std::min(hostMaxBlock, numFrames - offset);
      float* in[2] = {inputs[0] + offset, inputs[1] + offset};
      float* out[2] = {outputs[0] + offset, outputs[1] + offset};
      impl->resampler.ProcessBlock(in, out, length, viaReference);
    }
  } catch (...) {
    for (int ch = 0; ch < 2; ++ch)
      juce::FloatVectorOperations::clear(outputs[ch], numFrames);
  }
}

// GetLatency() only counts the warm-up prefill and misses the residual group delay of the two
// Lanczos kernels (2-5 samples, growing with the rate ratio), so an impulse is run through a
// scratch boundary and the peak located: exact by construction, and cheap enough for a host's
// prepare call.
int RateBoundary::measureLatency(double hostSampleRate, int blockSize) {
  Resampler probe(kBaseSampleRate);
  probe.Reset(hostSampleRate, blockSize);

  const int total = ((probe.GetLatency() + 2 * blockSize) / blockSize + 1) * blockSize;
  juce::AudioBuffer<float> in(2, total), out(2, total);
  in.clear();
  out.clear();
  in.setSample(0, 0, 1.0f);
  in.setSample(1, 0, 1.0f);

  auto identity = [](float** inputs, float** outputs, int frames) {
    juce::FloatVectorOperations::copy(outputs[0], inputs[0], frames);
    juce::FloatVectorOperations::copy(outputs[1], inputs[1], frames);
  };
  for (int offset = 0; offset < total; offset += blockSize) {
    float* ins[2] = {in.getWritePointer(0, offset), in.getWritePointer(1, offset)};
    float* outs[2] = {out.getWritePointer(0, offset), out.getWritePointer(1, offset)};
    probe.ProcessBlock(ins, outs, blockSize, identity);
  }

  int peak = 0;
  float best = 0.0f;
  const float* y = out.getReadPointer(0);
  for (int i = 0; i < total; ++i)
    if (std::abs(y[i]) > best) {
      best = std::abs(y[i]);
      peak = i;
    }
  return peak;
}

}  // namespace namloader
