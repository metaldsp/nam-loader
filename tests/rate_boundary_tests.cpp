// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// RateBoundary: the host-rate <-> 48 kHz boundary, driven directly with a callback standing in
// for the processing graph. Covers the boundary latency against an impulse and against the
// measured delay, transparency, variable chunking, and the inactive 48 kHz path.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "nam_loader/RateBoundary.h"
#include "test_helpers.h"

using namespace namloader;

namespace {

// What a graph that does nothing does to a block: the boundary's own round trip is all that is
// left. Distinct buffers are copied with memmove, so aliased in-place use is fine too.
const auto kIdentity = [](float** inputs, float** outputs, int frames) {
  for (int ch = 0; ch < 2; ++ch)
    std::memmove(outputs[ch], inputs[ch], sizeof(float) * static_cast<size_t>(frames));
};

// `in` on both channels through a fresh boundary at `hostRate`, in blocks that cycle through
// `sizes` (each at most `maxBlock`), in place.
std::vector<float> throughBoundary(const std::vector<float>& in,
                                   double hostRate,
                                   int maxBlock,
                                   const std::vector<int>& sizes) {
  RateBoundary boundary;
  boundary.prepare(hostRate, maxBlock);
  std::vector<float> left = in, right = in;
  size_t pos = 0;
  for (size_t block = 0; pos < in.size(); ++block) {
    const size_t n =
        std::min<size_t>(static_cast<size_t>(sizes[block % sizes.size()]), in.size() - pos);
    float* ch[2] = {left.data() + pos, right.data() + pos};
    boundary.process(ch, ch, static_cast<int>(n), kIdentity);
    pos += n;
  }
  return left;
}

double settledGainDb(const std::vector<float>& out,
                     const std::vector<float>& in,
                     double freq,
                     double fs,
                     int start = 16384,
                     int window = 16384) {
  const double gOut = goertzelPower(out.data() + start, static_cast<size_t>(window), freq, fs);
  const double gIn = goertzelPower(in.data() + start, static_cast<size_t>(window), freq, fs);
  return db(gOut) - db(gIn);
}

}  // namespace

// The figure a host aligns its dry path against is where an impulse actually comes out.
TEST(RateBoundary, LatencyMatchesImpulse) {
  for (const double rate : {44100.0, 96000.0, 22050.0}) {
    SCOPED_TRACE(rate);
    const int latency = RateBoundary::measureLatency(rate, 256);
    std::vector<float> impulse(4096, 0.0f);
    impulse[0] = 1.0f;
    const std::vector<float> out = throughBoundary(impulse, rate, 256, {256});
    const auto peak = std::max_element(out.begin(), out.end(),
                                       [](float a, float b) { return std::abs(a) < std::abs(b); });
    EXPECT_EQ(static_cast<int>(peak - out.begin()), latency);
    EXPECT_GT(latency, 0);
  }
}

// The reported figure must match the physical group delay of the signal, and the round trip must
// be transparent.
TEST(RateBoundary, ReportedLatencyMatchesMeasuredDelay) {
  for (const double hostRate : {44100.0, 96000.0}) {
    const int reported = RateBoundary::measureLatency(hostRate, 512);
    EXPECT_GT(reported, 0) << hostRate << " Hz host must engage the boundary";

    const int total = 120 * 512;
    const auto noise = makeNoise(total, 777, 0.25f);
    const auto out = throughBoundary(noise, hostRate, 512, {512});
    const int measured = bestCorrelationLag(out, noise, 16384, 8192, reported + 64);
    EXPECT_NEAR(measured, reported, 2) << hostRate << " Hz: reported latency drifted from reality";

    // And the boundary itself must be sonically transparent.
    const auto sine = makeSine(total, 997.0, 0.5f, hostRate);
    const auto sineOut = throughBoundary(sine, hostRate, 512, {512});
    EXPECT_NEAR(settledGainDb(sineOut, sine, 997.0, hostRate), 0.0, 0.1)
        << hostRate << " Hz: boundary not transparent";
  }
}

// The figure is a property of the host rate, not of how the host happens to size its
// blocks, so the latency a host compensates for stays put when the buffer size changes.
TEST(RateBoundary, MeasuredLatencyIsStableAcrossBlockSizes) {
  for (const double rate : {44100.0, 96000.0}) {
    SCOPED_TRACE(rate);
    const int reference = RateBoundary::measureLatency(rate, 512);
    EXPECT_GT(reference, 0);
    for (const int block : {32, 64, 128, 256, 1024})
      EXPECT_EQ(RateBoundary::measureLatency(rate, block), reference) << "block " << block;
    // Measuring again gives the same answer (no state survives between calls).
    EXPECT_EQ(RateBoundary::measureLatency(rate, 512), reference);
  }
}

// The output of a prepared boundary depends only on the input and the chunk sequence.
TEST(RateBoundary, VariableChunkRoundTripIsDeterministic) {
  const std::vector<int> sizes = {61, 127, 509, 1, 333, 480};
  const std::vector<float> in = makeNoise(20000, 5);
  for (const double rate : {44100.0, 96000.0}) {
    SCOPED_TRACE(rate);
    const std::vector<float> first = throughBoundary(in, rate, 512, sizes);
    const std::vector<float> second = throughBoundary(in, rate, 512, sizes);
    EXPECT_EQ(std::memcmp(first.data(), second.data(), in.size() * sizeof(float)), 0);
  }
}

// The Lanczos kernel tables are unsynchronised function statics, so the caller can warm
// them once before any prepare() runs concurrently. It must be safe to call any number of times,
// from any thread, and must not change what the boundary does.
TEST(RateBoundary, WarmTablesIsIdempotentAndThreadSafe) {
  const int before = RateBoundary::measureLatency(44100.0, 256);
  RateBoundary::warmTables();
  RateBoundary::warmTables();
  EXPECT_EQ(RateBoundary::measureLatency(44100.0, 256), before);

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([] {
      for (int k = 0; k < 8; ++k)
        RateBoundary::warmTables();
    });
  for (auto& t : threads)
    t.join();

  EXPECT_EQ(RateBoundary::measureLatency(44100.0, 256), before);
  EXPECT_EQ(RateBoundary::measureLatency(96000.0, 256), RateBoundary::measureLatency(96000.0, 256));
}

// At 48 kHz both layers are skipped: no latency, and the callback is simply called with the
// caller's own buffers. A stateful, non-identity callback (gain plus a one-sample delay) proves
// the result is exactly what a direct call produces.
TEST(RateBoundary, InactiveAt48kIsADirectCall) {
  RateBoundary boundary;
  boundary.prepare(48000.0, 512);
  EXPECT_FALSE(boundary.isActive());
  EXPECT_EQ(boundary.getLatencySamples(), 0);
  EXPECT_EQ(boundary.getMaxBaseBlockSize(), 512);

  const auto makeFn = [](float& state) {
    return [&state](float** in, float** out, int frames) {
      for (int ch = 0; ch < 2; ++ch) {
        float previous = ch == 0 ? state : 0.0f;
        for (int i = 0; i < frames; ++i) {
          const float x = in[ch][i];
          out[ch][i] = 0.5f * previous + 0.25f * x;
          previous = x;
        }
        if (ch == 0)
          state = previous;
      }
    };
  };

  const std::vector<int> sizes = {512, 61, 300, 1, 512};
  const auto input = makeNoise(5000, 91);
  std::vector<float> viaBoundary(input.size()), direct(input.size());
  std::vector<float> viaRight(input.size()), directRight(input.size());
  float stateA = 0.0f, stateB = 0.0f;
  const auto fnA = makeFn(stateA);
  const auto fnB = makeFn(stateB);
  size_t pos = 0;
  for (size_t block = 0; pos < input.size(); ++block) {
    const size_t n =
        std::min<size_t>(static_cast<size_t>(sizes[block % sizes.size()]), input.size() - pos);
    std::vector<float> in0(input.begin() + pos, input.begin() + pos + n), in1 = in0;
    float* ins[2] = {in0.data(), in1.data()};
    float* outsA[2] = {viaBoundary.data() + pos, viaRight.data() + pos};
    float* outsB[2] = {direct.data() + pos, directRight.data() + pos};
    boundary.process(ins, outsA, static_cast<int>(n), fnA);
    fnB(ins, outsB, static_cast<int>(n));
    pos += n;
  }
  EXPECT_EQ(std::memcmp(viaBoundary.data(), direct.data(), input.size() * sizeof(float)), 0);
  EXPECT_EQ(std::memcmp(viaRight.data(), directRight.data(), input.size() * sizeof(float)), 0);
  EXPECT_GT(*std::max_element(direct.begin(), direct.end()), 0.01f) << "callback did something";
}

// The boundary engages at any other host rate and reports the latency it measured.
TEST(RateBoundary, ActiveAtOtherRatesReportsMeasuredLatency) {
  for (const double rate : {44100.0, 96000.0, 88200.0, 22050.0}) {
    SCOPED_TRACE(rate);
    RateBoundary boundary;
    boundary.prepare(rate, 256);
    EXPECT_TRUE(boundary.isActive());
    EXPECT_GT(boundary.getLatencySamples(), 0);
    EXPECT_EQ(boundary.getLatencySamples(), RateBoundary::measureLatency(rate, 256));
  }
}

// The callback runs at the base rate: it sees block sizes scaled by the rate ratio, within
// the documented bound, and the frames it is handed add up to what 48 kHz needs.
TEST(RateBoundary, CallbackRunsAtTheBaseRate) {
  for (const double rate : {44100.0, 96000.0}) {
    SCOPED_TRACE(rate);
    constexpr int kBlock = 512;
    const int bound = static_cast<int>(std::ceil(kBlock / (rate / kFs)));
    RateBoundary boundary;
    boundary.prepare(rate, kBlock);
    EXPECT_EQ(boundary.getMaxBaseBlockSize(), bound);

    long long callbackFrames = 0;
    int largest = 0;
    const auto count = [&](float** in, float** out, int frames) {
      callbackFrames += frames;
      largest = std::max(largest, frames);
      kIdentity(in, out, frames);
    };
    const int blocks = 200;
    std::vector<float> left(kBlock, 0.1f), right(kBlock, 0.1f);
    for (int b = 0; b < blocks; ++b) {
      float* ch[2] = {left.data(), right.data()};
      boundary.process(ch, ch, kBlock, count);
    }
    EXPECT_LE(largest, bound);
    const double expected = static_cast<double>(blocks) * kBlock * kFs / rate;
    EXPECT_NEAR(static_cast<double>(callbackFrames), expected, kBlock * 2.0);
  }
}

// A host that exceeds the block size it promised is fed through in prepared-size slices;
// the result equals feeding the same samples in slices of the prepared size.
TEST(RateBoundary, OversizeHostBlockIsSliced) {
  const auto input = makeNoise(4 * 256, 17);
  for (const double rate : {44100.0, 96000.0}) {
    SCOPED_TRACE(rate);
    const auto sliced = throughBoundary(input, rate, 256, {256});

    RateBoundary boundary;
    boundary.prepare(rate, 256);
    std::vector<float> left = input, right = input;
    float* ch[2] = {left.data(), right.data()};
    boundary.process(ch, ch, static_cast<int>(input.size()), kIdentity);
    EXPECT_EQ(std::memcmp(left.data(), sliced.data(), input.size() * sizeof(float)), 0);
  }
}

// Reset() returns the boundary to its freshly prepared state, delay included.
TEST(RateBoundary, ResetMatchesAFreshlyPreparedBoundary) {
  for (const double rate : {44100.0, 96000.0}) {
    SCOPED_TRACE(rate);
    const auto warmup = makeNoise(3000, 3);
    const auto probe = makeNoise(5000, 4);

    RateBoundary used;
    used.prepare(rate, 512);
    std::vector<float> a = warmup, b = warmup;
    for (size_t pos = 0; pos < warmup.size(); pos += 500) {
      float* ch[2] = {a.data() + pos, b.data() + pos};
      used.process(ch, ch, 500, kIdentity);
    }
    used.reset();
    EXPECT_TRUE(used.isActive());

    RateBoundary fresh;
    fresh.prepare(rate, 512);
    std::vector<float> fromUsed = probe, fromUsedRight = probe, fromFresh = probe,
                       fromFreshRight = probe;
    for (size_t pos = 0; pos < probe.size(); pos += 500) {
      float* chUsed[2] = {fromUsed.data() + pos, fromUsedRight.data() + pos};
      float* chFresh[2] = {fromFresh.data() + pos, fromFreshRight.data() + pos};
      used.process(chUsed, chUsed, 500, kIdentity);
      fresh.process(chFresh, chFresh, 500, kIdentity);
    }
    EXPECT_EQ(std::memcmp(fromUsed.data(), fromFresh.data(), probe.size() * sizeof(float)), 0);
  }
}

// Preparing again at 48 kHz releases the resampler, and again at another rate engages it.
TEST(RateBoundary, PrepareSwitchesBetweenActiveAndInactive) {
  RateBoundary boundary;
  boundary.prepare(44100.0, 256);
  EXPECT_TRUE(boundary.isActive());
  boundary.prepare(48000.0, 256);
  EXPECT_FALSE(boundary.isActive());
  EXPECT_EQ(boundary.getLatencySamples(), 0);
  boundary.prepare(96000.0, 256);
  EXPECT_TRUE(boundary.isActive());
  EXPECT_EQ(boundary.getLatencySamples(), RateBoundary::measureLatency(96000.0, 256));
}

// Nothing escapes process(): a throwing callback leaves silence, active or not.
TEST(RateBoundary, ThrowingCallbackLeavesSilence) {
  for (const double rate : {48000.0, 44100.0}) {
    SCOPED_TRACE(rate);
    RateBoundary boundary;
    boundary.prepare(rate, 256);
    std::vector<float> left(256, 0.5f), right(256, 0.5f);
    float* ch[2] = {left.data(), right.data()};
    const auto thrower = [](float**, float**, int) { throw std::runtime_error("boom"); };
    EXPECT_NO_THROW(boundary.process(ch, ch, 256, thrower));
    for (const float s : left)
      ASSERT_EQ(s, 0.0f);
    for (const float s : right)
      ASSERT_EQ(s, 0.0f);
  }
}
