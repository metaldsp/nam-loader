// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// Oversampler tests
//
// The x2/4/8 wrapper and the base-rate section: null tests, frame-count
// contracts, transparency, group delay, alias reduction, and the
// phase-interleave math that Engine's oversampled mode rests on.
#include "nam_loader/Oversampler.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// Streams `in` through an Oversampler in `blockSize` chunks;
// `processFn(channels, frames)` processes the oversampled stereo block in
// place. Returns the base-rate output (channel 0).
template <typename Fn>
std::vector<float> runOversampled(const std::vector<float>& in,
                                  int factor,
                                  int blockSize,
                                  Fn&& processFn) {
  namloader::Oversampler os;
  os.prepare(factor, blockSize);
  std::vector<float> out(in.size(), 0.0f);
  std::vector<float> silentIn(in.size(), 0.0f), silentOut(in.size(), 0.0f);
  for (size_t off = 0; off < in.size(); off += static_cast<size_t>(blockSize)) {
    const int frames = static_cast<int>(std::min<size_t>(blockSize, in.size() - off));
    float* ins[2] = {const_cast<float*>(in.data()) + off, silentIn.data() + off};
    float* outs[2] = {out.data() + off, silentOut.data() + off};
    os.process(ins, outs, frames, [&](float** ci, float** co, int n) {
      for (int ch = 0; ch < 2; ++ch)
        if (co[ch] != ci[ch])
          std::copy(ci[ch], ci[ch] + n, co[ch]);
      processFn(co, n);
    });
  }
  return out;
}

// Oversampler

TEST(Oversampler, Factor1IsBitExactPassthrough) {
  const int blockSize = 512, total = 93 * 512;
  const auto in = makeNoise(total, 42);
  const auto out = runOversampled(in, 1, blockSize, [](float**, int) {});
  EXPECT_EQ(out, in);  // the OS-off null test: not one bit may move
}

TEST(Oversampler, FrameCountContract) {
  // Every base frame must become exactly `factor` oversampled frames per call;
  // Engine's phase interleave and the decimator's pairing depend on it.
  const int blockSize = 512, total = 24 * 512;
  const std::vector<float> in(static_cast<size_t>(total), 0.0f);
  for (int factor : {2, 4, 8}) {
    int calls = 0;
    bool framesOk = true;
    runOversampled(in, factor, blockSize, [&](float**, int n) {
      framesOk = framesOk && n == blockSize * factor;
      ++calls;
    });
    EXPECT_TRUE(framesOk) << "factor " << factor;
    EXPECT_EQ(calls, total / blockSize) << "factor " << factor;
  }
}

TEST(Oversampler, PassbandTransparentWithTinyGroupDelay) {
  const int blockSize = 512, total = 93 * 512;
  const double f0 = 997.0;
  const auto in = makeSine(total, f0);
  for (int factor : {2, 4, 8}) {
    const auto out = runOversampled(in, factor, blockSize, [](float**, int) {});

    std::vector<float> tail(out.begin() + 8192, out.begin() + 8192 + 16384);
    const double gainDb = db(goertzelPower(tail, f0));
    EXPECT_NEAR(gainDb, 0.0, 0.1) << "factor " << factor << ": passband gain moved";

    // Min-phase halfbands report zero PDC latency but carry a few samples of
    // physical group delay; it must stay tiny and not blow up with factor.
    int bestLag = -1;
    double bestCorr = -1e30;
    for (int lag = 0; lag < 32; ++lag) {
      double corr = 0.0;
      for (int i = 8192; i < 8192 + 4096; ++i)
        corr += static_cast<double>(out[static_cast<size_t>(i)]) *
                static_cast<double>(in[static_cast<size_t>(i - lag)]);
      if (corr > bestCorr) {
        bestCorr = corr;
        bestLag = lag;
      }
    }
    EXPECT_LE(bestLag, 8) << "factor " << factor << ": group delay grew";
  }
}

TEST(Oversampler, HardClipperAliasingDropsWithFactor) {
  // A clipped 4998 Hz sine has harmonics far above the base Nyquist; at
  // factor 1 they fold back to known inharmonic frequencies. Oversampling
  // must push those folded components down (until the float noise floor).
  const int blockSize = 512, total = 93 * 512;
  const double f0 = 4998.046875;
  const auto in = makeSine(total, f0, 2.5f);

  auto clipper = [](float** ch, int n) {
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < n; ++i)
        ch[c][i] = std::max(-1.0f, std::min(1.0f, ch[c][i]));
  };

  std::vector<double> aliasFreqs;
  for (int h = 5; h <= 15; h += 2)  // clipped sine: odd harmonics only
    aliasFreqs.push_back(foldFrequency(h * f0));

  double aliasDb1 = 0.0, prevAlias = 0.0;
  for (int factor : {1, 2, 4, 8}) {
    const auto out = runOversampled(in, factor, blockSize, clipper);
    std::vector<float> tail(out.begin() + 8192, out.begin() + 8192 + 16384);

    double aliasPower = 0.0;
    for (double f : aliasFreqs)
      aliasPower += goertzelPower(tail, f);
    const double aliasRatioDb = db(aliasPower) - db(goertzelPower(tail, f0));
    std::printf("  clipper factor %d: folded harmonics vs fundamental = %+.1f dB\n", factor,
                aliasRatioDb);

    if (factor == 1)
      aliasDb1 = aliasRatioDb;
    else
      EXPECT_TRUE(aliasRatioDb < prevAlias - 3.0 || aliasRatioDb < -110.0)
          << "factor " << factor << " did not improve on factor " << factor / 2;
    prevAlias = aliasRatioDb;
  }
  EXPECT_LT(prevAlias, aliasDb1 - 20.0) << "factor 8 should cut aliasing by >20 dB";
}

TEST(Oversampler, PhaseInterleavedDilatedConvIsExact) {
  // The math Engine's oversampled mode rests on: N phase copies of a
  // dilated convolution at the native rate == one dilation-scaled
  // convolution at the oversampled rate. Exact, not approximate.
  const int factorN = 4, dilation = 3, taps = 4, len = 4096;
  std::vector<double> x(len), kernel = {0.6, -0.3, 0.2, 0.15};
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& s : x)
    s = dist(rng);

  std::vector<double> highRate(len, 0.0), phased(len, 0.0);
  for (int i = 0; i < len; ++i)
    for (int t = 0; t < taps; ++t) {
      const int j = i - t * dilation * factorN;
      if (j >= 0)
        highRate[static_cast<size_t>(i)] +=
            kernel[static_cast<size_t>(t)] * x[static_cast<size_t>(j)];
    }
  for (int p = 0; p < factorN; ++p)
    for (int i = p; i < len; i += factorN)
      for (int t = 0; t < taps; ++t) {
        const int j = i - t * dilation * factorN;
        if (j >= 0)
          phased[static_cast<size_t>(i)] +=
              kernel[static_cast<size_t>(t)] * x[static_cast<size_t>(j)];
      }

  for (int i = 0; i < len; ++i)
    ASSERT_EQ(highRate[static_cast<size_t>(i)], phased[static_cast<size_t>(i)]) << "at " << i;
}

TEST(Oversampler, ReportsStateAndZeroLatency) {
  namloader::Oversampler os;
  os.prepare(1, 512);
  EXPECT_FALSE(os.isActive());
  EXPECT_EQ(os.getFactor(), 1);
  for (int factor : {2, 4, 8}) {
    os.prepare(factor, 512);
    EXPECT_TRUE(os.isActive());
    EXPECT_EQ(os.getFactor(), factor);
    EXPECT_EQ(os.getLatencySamples(), 0) << "factor " << factor;
  }
}

// reset() must return the filters to their post-prepare state: replaying the
// same input after a reset reproduces the first run bit for bit.
TEST(Oversampler, ResetClearsFilterState) {
  const int blockSize = 512;
  const auto in = makeNoise(blockSize, 5);
  for (int factor : {2, 4, 8}) {
    namloader::Oversampler os;
    os.prepare(factor, blockSize);
    auto run = [&] {
      std::vector<float> a = in, silence(in.size(), 0.0f), out(in.size()), outRight(in.size());
      float* ins[2] = {a.data(), silence.data()};
      float* outs[2] = {out.data(), outRight.data()};
      os.process(ins, outs, blockSize, [](float**, float**, int) {});
      return out;
    };
    const auto first = run();
    run();  // leave non-zero filter state behind
    os.reset();
    EXPECT_EQ(run(), first) << "factor " << factor;
  }
}

TEST(Oversampler, AtBaseRateFactor1IsBitExact) {
  const int baseBlock = 512, total = 93 * 512;
  auto in = makeNoise(total, 99);
  const auto ref = in;
  std::vector<float> right(in.size(), 0.0f);

  namloader::Oversampler baseSection;
  baseSection.prepare(1, baseBlock);
  for (size_t off = 0; off < in.size(); off += static_cast<size_t>(baseBlock)) {
    float* chans[2] = {in.data() + off, right.data() + off};
    baseSection.processAtBaseRate(chans, 2, baseBlock, [](float* const*, int) {});
  }
  EXPECT_EQ(in, ref);
}

TEST(Oversampler, AtBaseRateFrameCountsAndSignalPath) {
  // The inner function must see exactly osFrames/factor base frames, and
  // whatever it does must reach the output (marker gain ×2 → +6.02 dB).
  const int baseBlock = 512;
  const double f0 = 997.0;
  for (int factor : {2, 4, 8}) {
    const int osBlock = baseBlock * factor;
    const int osTotal = 93 * osBlock;
    auto sig = makeSine(osTotal, f0);
    std::vector<float> right(sig.size(), 0.0f);

    namloader::Oversampler baseSection;
    baseSection.prepare(factor, baseBlock);
    bool framesOk = true;
    int calls = 0;
    for (int off = 0; off < osTotal; off += osBlock) {
      float* chans[2] = {sig.data() + off, right.data() + off};
      baseSection.processAtBaseRate(chans, 2, osBlock, [&](float* const* ch, int n) {
        framesOk = framesOk && n == osBlock / factor;
        ++calls;
        for (int i = 0; i < n; ++i)
          ch[0][i] *= 2.0f;
      });
    }
    EXPECT_TRUE(framesOk) << "factor " << factor;
    EXPECT_EQ(calls, osTotal / osBlock) << "factor " << factor;

    std::vector<float> tail(sig.begin() + 8192, sig.begin() + 8192 + 16384);
    const double gainDb = db(goertzelPower(tail, f0));
    EXPECT_NEAR(gainDb, 20.0 * std::log10(2.0), 0.1) << "factor " << factor;
  }
}

TEST(Oversampler, AtBaseRateHandlesMonoBuffers) {
  const int baseBlock = 512;
  namloader::Oversampler baseSection;
  baseSection.prepare(8, baseBlock);
  std::vector<float> mono(static_cast<size_t>(baseBlock * 8), 0.5f);
  float* chans[1] = {mono.data()};
  baseSection.processAtBaseRate(chans, 1, baseBlock * 8, [](float* const*, int) {});
  for (float s : mono)
    ASSERT_TRUE(std::isfinite(s));
}

}  // namespace
