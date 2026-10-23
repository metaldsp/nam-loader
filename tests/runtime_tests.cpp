// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// Runtime: one Model run through an Engine with the input calibration and post-model gain
// around it: voices per declared channel, worker-pool bit-identity, prewarm samples, calibrated
// hand-off and normalization gain math, engine exception.
//
// The real captures in tests/files carry the metadata the hand-off tests need
// (a2-amp-cab-test.nam: input 15 dBu, output 14 dBu, loudness -15.927; a2-amp-test.nam: loudness
// only). The gain-math tests edit a copy of a Model running a pass-through architecture, so the
// applied gain can be read off the output exactly.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/model_config.h"
#include "nam_loader/Model.h"
#include "nam_loader/Runtime.h"
#include "nam_loader/WorkerPool.h"
#include "test_helpers.h"

using namespace namloader;

namespace {

constexpr int kBlock = 512;  // base-rate frames per block

// ---------------------------------------------------------------------------
// A NAM architecture that passes audio through and throws on demand, to prove
// the engine-exception path. Registered once with NAM's own registry, so it
// loads through the same loadModel/Runtime path as a real capture.
// ---------------------------------------------------------------------------

std::atomic<bool> gThrowingDspThrows{false};
std::atomic<int> gThrowingDspThrowCount{0};

class ThrowingDsp : public nam::DSP {
public:
  ThrowingDsp() : nam::DSP(1, 1, 48000.0) {}
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int numFrames) override {
    if (gThrowingDspThrows.load()) {
      ++gThrowingDspThrowCount;
      throw std::runtime_error("test engine failure");
    }
    for (int i = 0; i < numFrames; ++i)
      output[0][i] = input[0][i];
  }
};

class ThrowingConfig : public nam::ModelConfig {
public:
  std::unique_ptr<nam::DSP> create(std::vector<float>, double) override {
    return std::make_unique<ThrowingDsp>();
  }
};

const nam::ConfigParserHelper kRegisterThrowingDsp("RuntimeTestThrowing",
                                                   [](const nlohmann::json&,
                                                      double) -> std::unique_ptr<nam::ModelConfig> {
                                                     return std::make_unique<ThrowingConfig>();
                                                   });

std::vector<std::uint8_t> throwingModelBytes() {
  const std::string json =
      R"({"version":"0.7.0","architecture":"RuntimeTestThrowing","config":{},"weights":[],)"
      R"("metadata":{},"sample_rate":48000})";
  return std::vector<std::uint8_t>(json.begin(), json.end());
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

std::shared_ptr<const Model> loadFile(const juce::File& file) {
  juce::String error;
  auto model = loadModel(file, error);
  EXPECT_NE(model, nullptr) << file.getFileName() << ": " << error;
  return model;
}

// One of the real captures in tests/files.
std::shared_ptr<const Model> loadCapture(const char* fixture) {
  return loadFile(testFile(fixture));
}

// A copy of `base` with `edit` applied: the metadata fields are plain values,
// so a test sets exactly the ones it is about without a new fixture.
std::shared_ptr<const Model> edited(const std::shared_ptr<const Model>& base,
                                    const std::function<void(Model&)>& edit) {
  auto copy = std::make_shared<Model>(*base);
  edit(*copy);
  return copy;
}

// The capture the calibration tests are written around: input 15 dBu, output 14 dBu, loudness
// -15.927 LUFS. At the default calibration level of 12 dBu and target of -18 LUFS:
//   hand-off gain           = 14 - 12            = +2.000 dB
//   normalization gain      = -18 - (-15.927188) = -2.073 dB
//   hand-off over normalized                     = +4.073 dB
const char* const kMeteredCapture = "a2-amp-cab-test.nam";

constexpr double kHandOffDb = 14.0 - 12.0;
constexpr double kNormalizationDb = -18.0 - (-15.927188453331137);
constexpr double kHandOffOverNormalizedDb = kHandOffDb - kNormalizationDb;

// The pass-through architecture with no metadata: the engine returns its input
// bit-exactly (float -> double -> float), so what the runtime applies around it
// is the whole gain.
std::shared_ptr<const Model> loadUnity() {
  juce::String error;
  auto model = loadModel(throwingModelBytes(), error);
  EXPECT_NE(model, nullptr) << error;
  return model;
}

RuntimeSpec specFor(int oversampleFactor, int channels, double slimmableSize) {
  RuntimeSpec spec;
  spec.oversampleFactor = oversampleFactor;
  spec.maxBlockSize = kBlock * oversampleFactor;
  spec.channels = channels;
  spec.slimmableSize = slimmableSize;
  return spec;
}

// Builds runtimes and drives them in place.
class Rig {
public:
  explicit Rig(int oversampleFactor = 1) : factor(oversampleFactor) {}

  int maxBlockSize() const { return kBlock * factor; }

  // `slimSize` defaults to 0.0, as the original block runtime tests ran their models.
  std::unique_ptr<Runtime> build(std::shared_ptr<const Model> model,
                                 int channels,
                                 double slimSize = 0.0) {
    return Runtime::create(std::move(model), specFor(factor, channels, slimSize));
  }

  // Runs one block in place.
  void process(Runtime& rt,
               std::vector<std::vector<float>>& channels,
               int offset,
               int count,
               const ProcessContext& ctx,
               bool normalize = true,
               bool calibratedHandOff = false) {
    std::array<float*, 2> ptrs{};
    for (size_t c = 0; c < channels.size(); ++c)
      ptrs[c] = channels[c].data() + offset;
    ProcessContext call = ctx;
    call.normalize = normalize;
    call.calibratedHandOff = calibratedHandOff;
    rt.process(ptrs.data(), count, call);
  }

  // The whole signal in maxBlockSize blocks.
  void run(Runtime& rt,
           std::vector<std::vector<float>>& channels,
           const ProcessContext& ctx,
           bool normalize = true,
           bool calibratedHandOff = false) {
    const int total = static_cast<int>(channels[0].size());
    for (int offset = 0; offset < total; offset += maxBlockSize())
      process(rt, channels, offset, std::min(maxBlockSize(), total - offset), ctx, normalize,
              calibratedHandOff);
  }

  int factor;
};

std::vector<float> noise(int n, unsigned seed, float amp = 0.2f) {
  return makeNoise(n, seed, amp);
}

// Test material: noise plus a mid-range tone, so the amp's nonlinearity has
// something to act on.
std::vector<float> material(int n, unsigned seed) {
  auto x = noise(n, seed, 0.15f);
  const auto sine = makeSine(n, 220.0, 0.25f);
  for (int i = 0; i < n; ++i)
    x[static_cast<size_t>(i)] += sine[static_cast<size_t>(i)];
  return x;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t from = 0) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (size_t i = from; i < std::min(a.size(), b.size()); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return worst;
}

double rms(const std::vector<float>& x, size_t from, size_t count) {
  double sum = 0.0;
  for (size_t i = from; i < from + count; ++i)
    sum += static_cast<double>(x[i]) * x[i];
  return std::sqrt(sum / static_cast<double>(count));
}

double toDb(double amplitude) {
  return 20.0 * std::log10(std::max(amplitude, 1.0e-12));
}

// The gain, in dB, the runtime applies around a pass-through model: one block
// through a fresh runtime, then the least-squares scale between input and
// output. The first block holds the primed gain from its first sample, so the
// block is a pure scale. `model` must be built on the pass-through
// architecture (loadUnity()).
double appliedGainDb(const std::shared_ptr<const Model>& model,
                     const ProcessContext& ctx,
                     bool normalize,
                     bool calibratedHandOff) {
  Rig rig;
  auto rt = rig.build(model, 1);
  EXPECT_NE(rt, nullptr);
  if (rt == nullptr)
    return 0.0;
  const std::vector<float> input = makeSine(kBlock, 440.0, 0.1f);
  std::vector<std::vector<float>> io = {input};
  rig.process(*rt, io, 0, kBlock, ctx, normalize, calibratedHandOff);
  double cross = 0.0, energy = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    cross += static_cast<double>(io[0][i]) * input[i];
    energy += static_cast<double>(input[i]) * input[i];
  }
  return toDb(cross / energy);
}

constexpr double kGainToleranceDb = 1.0e-3;

}  // namespace

// ---------------------------------------------------------------------------
// Declared formats
// ---------------------------------------------------------------------------

TEST(Runtime, StereoNamTwoIndependentVoices) {
  // A stereo NAM runs one voice per channel (spec 4.2): each output channel
  // equals what a mono runtime produces for that channel alone, whatever the
  // other channel carries, with and without phase interleaving.
  for (const int factor : {1, 4}) {
    Rig h(factor);
    const int n = h.maxBlockSize() * 8;
    const auto left = material(n, 31);
    const auto right = noise(n, 32, 0.3f);
    ProcessContext ctx;

    const auto model = loadCapture("a2-amp-test.nam");
    auto stereo = h.build(model, 2);
    ASSERT_NE(stereo, nullptr);
    std::vector<std::vector<float>> both = {left, right};
    h.run(*stereo, both, ctx);

    for (int ch = 0; ch < 2; ++ch) {
      auto mono = h.build(model, 1);
      ASSERT_NE(mono, nullptr);
      std::vector<std::vector<float>> single = {ch == 0 ? left : right};
      h.run(*mono, single, ctx);
      EXPECT_EQ(maxAbsDiff(both[static_cast<size_t>(ch)], single[0]), 0.0)
          << "x" << factor << " channel " << ch;
    }
    EXPECT_GT(maxAbsDiff(both[0], both[1]), 0.01) << "channels must stay independent";
  }
}

TEST(Runtime, StereoNamForksAcrossThePoolBitIdentically) {
  WorkerPool pool;
  pool.start(48000.0, kBlock * 4, 2);
  Rig h(4);
  const int n = h.maxBlockSize() * 6;
  const auto left = material(n, 41);
  const auto right = noise(n, 42, 0.3f);

  const auto model = loadCapture("a2-amp-test.nam");
  ProcessContext serial;
  ProcessContext pooled;
  pooled.pool = &pool;

  auto a = h.build(model, 2);
  auto b = h.build(model, 2);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  std::vector<std::vector<float>> viaSerial = {left, right};
  std::vector<std::vector<float>> viaPool = {left, right};
  h.run(*a, viaSerial, serial);
  h.run(*b, viaPool, pooled);
  pool.stop();
  EXPECT_EQ(maxAbsDiff(viaSerial[0], viaPool[0]), 0.0);
  EXPECT_EQ(maxAbsDiff(viaSerial[1], viaPool[1]), 0.0);
}

// The build contract in the Runtime header.
TEST(Runtime, CreateRejectsAnUnusableBuild) {
  Rig h;
  const auto model = loadCapture("a2-amp-test.nam");
  EXPECT_NE(h.build(model, 1), nullptr);
  EXPECT_NE(h.build(model, 2), nullptr);
  EXPECT_EQ(h.build(model, 0), nullptr) << "channel count must be 1 or 2";
  EXPECT_EQ(h.build(model, 3), nullptr) << "channel count must be 1 or 2";
  EXPECT_EQ(h.build(nullptr, 1), nullptr) << "no model";
  EXPECT_EQ(h.build(edited(model, [](Model& m) { m.data = nullptr; }), 1), nullptr)
      << "a model without NAM data";

  RuntimeSpec noBlock = specFor(1, 1, 0.0);
  noBlock.maxBlockSize = 0;
  EXPECT_EQ(Runtime::create(model, noBlock), nullptr);
  RuntimeSpec noFactor = specFor(1, 1, 0.0);
  noFactor.oversampleFactor = 0;
  EXPECT_EQ(Runtime::create(model, noFactor), nullptr);
}

// The accessors report what the runtime was built from.
TEST(Runtime, ExposesItsModelAndEngine) {
  const auto model = loadCapture("a2-amp-test.nam");
  for (const int factor : {1, 2, 8}) {
    SCOPED_TRACE(factor);
    Rig h(factor);
    auto rt = h.build(model, 2);
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(&rt->getModel(), model.get());
    EXPECT_EQ(rt->getEngine().getOversampleFactor(), factor);
    EXPECT_EQ(rt->getEngine().getVoiceCount(), 2);
    EXPECT_EQ(rt->getEngine().getSlimmableSize(), 0.0);
  }
}

TEST(Runtime, ReportsPrewarmSamples) {
  // The model's receptive field is what a caller pre-rolls before crossfading a runtime in.
  const auto model = loadCapture("a2-amp-test.nam");
  {
    Rig h(1);
    auto amp = h.build(model, 1);
    ASSERT_NE(amp, nullptr);
    EXPECT_EQ(amp->getPrewarmSamples(), model->prewarmSamples);
    EXPECT_GT(amp->getPrewarmSamples(), 0);
    EXPECT_FALSE(amp->hasFailed());
  }
  {
    // A phase instance sees every 4th oversampled sample, so its receptive field is four times
    // longer measured at the oversampled rate.
    Rig h(4);
    auto amp = h.build(model, 1);
    ASSERT_NE(amp, nullptr);
    EXPECT_EQ(amp->getEngine().getPhaseCount(), 4);
    EXPECT_EQ(amp->getPrewarmSamples(), 4 * model->prewarmSamples);
  }
  {
    // An architecture that is not phase-safe runs one instance per voice
    // at any oversampling factor, so there is no phase multiplier.
    const auto lstm = loadFile(exampleModel("lstm.nam"));
    ASSERT_FALSE(lstm->phaseSafe);
    Rig h(4);
    auto amp = h.build(lstm, 1);
    ASSERT_NE(amp, nullptr);
    EXPECT_EQ(amp->getEngine().getPhaseCount(), 1);
    EXPECT_EQ(amp->getPrewarmSamples(), lstm->prewarmSamples);
  }
}

// ---------------------------------------------------------------------------
// Calibrated hand-off and normalization with a real model
// ---------------------------------------------------------------------------

TEST(Runtime, HandoffFollowsTheCalibratedHandOffFlag) {
  // Calibrated hand-off needs calibration on, the model's output level, and a following model
  // that is enabled right now. The caller passes that per block, so toggling it moves the gain
  // with smoothing and never rebuilds anything.
  Rig h;
  const int n = kBlock * 40;
  const auto tone = makeSine(n, 220.0, 0.3f);
  // input 15 dBu, output 14 dBu, loudness -15.93
  auto runtime = h.build(loadCapture(kMeteredCapture), 1);
  ASSERT_NE(runtime, nullptr);

  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  // Steady-state level for a context/hand-off setting, same runtime instance.
  const auto levelAfter = [&](bool handOffOn, bool calibrate) {
    ProcessContext c = ctx;
    c.calibrateInput = calibrate;
    std::vector<std::vector<float>> io = {tone};
    for (int offset = 0; offset < n; offset += kBlock)
      h.process(*runtime, io, offset, kBlock, c, /*normalize=*/true, handOffOn);
    return toDb(rms(io[0], static_cast<size_t>(n) - 4096, 4096));
  };

  const double handoff = levelAfter(true, true);
  const double normalized = levelAfter(false, true);
  const double handoffAgain = levelAfter(true, true);
  const double calibrationOff = levelAfter(true, false);

  // The same capture; only the post-model gain differs. Hand-off applies
  // output level - calibration = 14 - 12 = +2 dB; normalization pulls the
  // -15.93 dB loudness to the -18 dB target: -2.07 dB. The calibration also
  // changes the model's drive (input gain 12 - 15 = -3 dB), identically in
  // both states, so the difference is purely the post-model gain.
  EXPECT_NEAR(handoff - normalized, 2.0 - (-18.0 - -15.927188), 0.15);
  EXPECT_NEAR(handoffAgain, handoff, 0.02);
  // With calibration off the model is driven at unity and normalizes: no
  // hand-off even though a following model is on.
  EXPECT_NE(calibrationOff, handoff);

  // The switch itself is smoothed. Going from normalization up to hand-off
  // (a 4 dB step up) must not step harder than the settled hand-off signal
  // already does; a hard gain switch would jump by ~0.4 of the peak.
  {
    std::vector<std::vector<float>> warm = {tone};
    for (int offset = 0; offset < n; offset += kBlock)
      h.process(*runtime, warm, offset, kBlock, ctx, true, /*calibratedHandOff=*/false);
    std::vector<std::vector<float>> io = {tone};
    for (int offset = 0; offset < kBlock * 8; offset += kBlock)
      h.process(*runtime, io, offset, kBlock, ctx, true, /*calibratedHandOff=*/true);
    std::vector<std::vector<float>> settled = {tone};
    for (int offset = 0; offset < n; offset += kBlock)
      h.process(*runtime, settled, offset, kBlock, ctx, true, true);

    const auto worstStep = [](const std::vector<float>& x, size_t from, size_t to) {
      double worst = 0.0;
      for (size_t i = from + 1; i < to; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(x[i]) - x[i - 1]));
      return worst;
    };
    const double steadyStep =
        worstStep(settled[0], static_cast<size_t>(n) - 4096, settled[0].size());
    const double switchStep = worstStep(io[0], 0, static_cast<size_t>(kBlock) * 8);
    EXPECT_LT(switchStep, steadyStep * 1.5);
  }
}

TEST(Runtime, HandoffNeedsTheFlagAndAnOutputLevel) {
  // The topology is a flag per call: none or disabled following model, or one enabled.
  Rig h;
  const int n = kBlock * 40;
  const auto tone = makeSine(n, 220.0, 0.3f);
  ProcessContext ctx;
  ctx.calibrateInput = true;

  const auto settledDb = [&](const std::shared_ptr<const Model>& model, bool calibratedHandOff) {
    auto runtime = h.build(model, 1);
    EXPECT_NE(runtime, nullptr);
    std::vector<std::vector<float>> io = {tone};
    h.run(*runtime, io, ctx, /*normalize=*/true, calibratedHandOff);
    return toDb(rms(io[0], static_cast<size_t>(n) - 4096, 4096));
  };

  const auto withLevels = loadCapture(kMeteredCapture);
  const double none = settledDb(withLevels, false);
  const double oneOn = settledDb(withLevels, true);
  EXPECT_NEAR(oneOn - none, 2.0 - (-18.0 - -15.927188), 0.15);

  // A model without an output level can not hand off: it normalizes whatever follows it,
  // which is the same as having nothing follow.
  const auto noLevels = loadCapture("a2-amp-test.nam");
  ASSERT_FALSE(noLevels->outputLevelDbu.has_value());
  EXPECT_EQ(settledDb(noLevels, false), settledDb(noLevels, true));
}

// The hand-off flag switches between hand-off and normalization on the same instance.
TEST(Runtime, HandOffFlagToggleSwitchesWithoutRebuild) {
  Rig h;
  const int n = kBlock * 150;  // 1.6 s: smoothers settled
  const auto tone = makeSine(n, 220.0, 0.1f);
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  auto runtime = h.build(loadCapture(kMeteredCapture), 1);
  ASSERT_NE(runtime, nullptr);
  const Runtime* original = runtime.get();

  const auto outDb = [&](bool handOffOn) {
    std::vector<std::vector<float>> io = {tone};
    h.run(*runtime, io, ctx, true, handOffOn);
    return toDb(rms(io[0], static_cast<size_t>(n) - 4096, 4096));
  };

  const double handedOff = outDb(true);
  const double normalized = outDb(false);
  EXPECT_NEAR(handedOff - normalized, kHandOffOverNormalizedDb, 0.3);
  EXPECT_NEAR(outDb(true), handedOff, 0.1);
  EXPECT_EQ(runtime.get(), original);
  EXPECT_FALSE(runtime->hasFailed());
}

// The last model stays on normalization; switching it off removes exactly the normalization
// gain.
TEST(Runtime, NormalizeOffRemovesTheNormalizationGain) {
  Rig h;
  const int n = kBlock * 150;
  const auto tone = makeSine(n, 220.0, 0.1f);
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  const auto terminalDb = [&](bool normalize) {
    auto runtime = h.build(loadCapture(kMeteredCapture), 1);
    EXPECT_NE(runtime, nullptr);
    std::vector<std::vector<float>> io = {tone};
    h.run(*runtime, io, ctx, normalize, /*calibratedHandOff=*/false);
    return toDb(rms(io[0], static_cast<size_t>(n) - 4096, 4096));
  };
  EXPECT_NEAR(terminalDb(false) - terminalDb(true), -kNormalizationDb, 0.3);
}

// ---------------------------------------------------------------------------
// Gain math (exact, on the pass-through architecture)
// ---------------------------------------------------------------------------

// The pass-through model reads the applied gains off one runtime exactly.
TEST(Runtime, InputCalibrationDrivesTheModelAtItsCaptureLevel) {
  const auto model = edited(loadUnity(), [](Model& m) { m.inputLevelDbu = 15.0; });
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;

  // The user's calibration dBu minus the model's: 12 - 15.
  EXPECT_NEAR(appliedGainDb(model, ctx, false, false), -3.0, kGainToleranceDb);
  ctx.inputCalibrationDbu = 18.0f;
  EXPECT_NEAR(appliedGainDb(model, ctx, false, false), 3.0, kGainToleranceDb);

  // Calibration off: the model is driven at unity.
  ctx.calibrateInput = false;
  EXPECT_NEAR(appliedGainDb(model, ctx, false, false), 0.0, kGainToleranceDb);

  // A model without an input level has nothing to calibrate against.
  ctx.calibrateInput = true;
  EXPECT_NEAR(appliedGainDb(loadUnity(), ctx, false, false), 0.0, kGainToleranceDb);
}

TEST(Runtime, HandoffIsOutputLevelMinusCalibration) {
  const auto model = edited(loadUnity(), [](Model& m) {
    m.outputLevelDbu = 14.0;
    m.loudnessDb = -15.927188453331137;
  });
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  // Another NAM follows: 14 - 12, whatever the loudness metadata says.
  EXPECT_NEAR(appliedGainDb(model, ctx, true, true), kHandOffDb, kGainToleranceDb);
  // Never both: the hand-off is not stacked with normalization, and it applies
  // even when this NAM's normalize switch is off.
  EXPECT_NEAR(appliedGainDb(model, ctx, false, true), kHandOffDb, kGainToleranceDb);
  ctx.targetLoudnessDb = -30.0f;
  EXPECT_NEAR(appliedGainDb(model, ctx, true, true), kHandOffDb, kGainToleranceDb);
  ctx.targetLoudnessDb = -18.0f;

  // The user's setting cancels against the next NAM's own input calibration:
  // the gain follows the calibration level one for one.
  ctx.inputCalibrationDbu = 6.0f;
  EXPECT_NEAR(appliedGainDb(model, ctx, true, true), 14.0 - 6.0, kGainToleranceDb);
  ctx.inputCalibrationDbu = 12.0f;

  // No clamp: it is a physical level difference, not a guess.
  const auto hot = edited(model, [](Model& m) { m.outputLevelDbu = 40.0; });
  EXPECT_NEAR(appliedGainDb(hot, ctx, true, true), 40.0 - 12.0, kGainToleranceDb);

  // The metadata sanity range is inclusive.
  const auto atMax = edited(model, [](Model& m) { m.outputLevelDbu = 60.0; });
  EXPECT_NEAR(appliedGainDb(atMax, ctx, true, true), 60.0 - 12.0, kGainToleranceDb);
  const auto atMin = edited(model, [](Model& m) { m.outputLevelDbu = -60.0; });
  EXPECT_NEAR(appliedGainDb(atMin, ctx, true, true), -60.0 - 12.0, kGainToleranceDb);
}

TEST(Runtime, HandoffFallsBackToNormalizationWhenItsConditionsFail) {
  const auto model = edited(loadUnity(), [](Model& m) {
    m.outputLevelDbu = 14.0;
    m.loudnessDb = -15.927188453331137;
  });
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  // The last model stays on normalization: calibrated output at the end of a series would swing
  // the overall volume with each capture's metadata.
  EXPECT_NEAR(appliedGainDb(model, ctx, true, /*calibratedHandOff=*/false), kNormalizationDb,
              kGainToleranceDb);
  EXPECT_NEAR(appliedGainDb(model, ctx, false, false), 0.0, kGainToleranceDb);

  // Calibration off: no hand-off even with a following model.
  ProcessContext off = ctx;
  off.calibrateInput = false;
  EXPECT_NEAR(appliedGainDb(model, off, true, true), kNormalizationDb, kGainToleranceDb);

  // No output level in the metadata.
  const auto noLevel = edited(model, [](Model& m) { m.outputLevelDbu.reset(); });
  EXPECT_NEAR(appliedGainDb(noLevel, ctx, true, true), kNormalizationDb, kGainToleranceDb);
}

TEST(Runtime, OutOfRangeOutputLevelFallsBackToNormalization) {
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  for (const double bad : {61.0, -61.0, 1.0e6, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    SCOPED_TRACE(bad);
    const auto model = edited(loadUnity(), [bad](Model& m) {
      m.outputLevelDbu = bad;
      m.loudnessDb = -15.927188453331137;
    });
    EXPECT_NEAR(appliedGainDb(model, ctx, true, true), kNormalizationDb, kGainToleranceDb);
    EXPECT_NEAR(appliedGainDb(model, ctx, false, true), 0.0, kGainToleranceDb);
  }
}

TEST(Runtime, NormalizationMovesLoudnessToTheTarget) {
  ProcessContext ctx;
  ctx.targetLoudnessDb = -18.0f;
  const auto withLoudness = [](double loudness) {
    return edited(loadUnity(), [loudness](Model& m) { m.loudnessDb = loudness; });
  };

  EXPECT_NEAR(appliedGainDb(withLoudness(-15.927188453331137), ctx, true, false), kNormalizationDb,
              kGainToleranceDb);
  EXPECT_NEAR(appliedGainDb(withLoudness(-26.4), ctx, true, false), 8.4, kGainToleranceDb);
  // The switch is the caller's per-call value.
  EXPECT_NEAR(appliedGainDb(withLoudness(-26.4), ctx, false, false), 0.0, kGainToleranceDb);

  // Calibration does not change it: only the hand-off replaces normalization.
  ctx.calibrateInput = true;
  EXPECT_NEAR(appliedGainDb(withLoudness(-26.4), ctx, true, false), 8.4, kGainToleranceDb);
  ctx.calibrateInput = false;

  // No loudness metadata: the model is taken to sit at the target.
  EXPECT_NEAR(appliedGainDb(loadUnity(), ctx, true, false), 0.0, kGainToleranceDb);
}

TEST(Runtime, NormalizationIsClampedToPlusMinusTwelveDb) {
  const auto withLoudness = [](double loudness) {
    return edited(loadUnity(), [loudness](Model& m) { m.loudnessDb = loudness; });
  };
  ProcessContext ctx;
  ctx.targetLoudnessDb = -18.0f;

  // A very quiet capture would want +22 dB.
  EXPECT_NEAR(appliedGainDb(withLoudness(-40.0), ctx, true, false), 12.0, kGainToleranceDb);
  // A very hot target against a loud capture would want -27 dB.
  ctx.targetLoudnessDb = -30.0f;
  EXPECT_NEAR(appliedGainDb(withLoudness(-3.0), ctx, true, false), -12.0, kGainToleranceDb);
  // The loudness sanity range is inclusive: 0 and -100 are legal and clamp.
  EXPECT_NEAR(appliedGainDb(withLoudness(0.0), ctx, true, false), -12.0, kGainToleranceDb);
  ctx.targetLoudnessDb = -18.0f;
  EXPECT_NEAR(appliedGainDb(withLoudness(-100.0), ctx, true, false), 12.0, kGainToleranceDb);
}

TEST(Runtime, OutOfRangeLoudnessFallsBackToTheTarget) {
  ProcessContext ctx;
  ctx.targetLoudnessDb = -18.0f;
  for (const double bad :
       {0.5, 5.0, -100.5, -120.0, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    SCOPED_TRACE(bad);
    const auto model = edited(loadUnity(), [bad](Model& m) { m.loudnessDb = bad; });
    EXPECT_NEAR(appliedGainDb(model, ctx, true, false), 0.0, kGainToleranceDb);
  }
}

TEST(Runtime, FirstBlockStartsAtTheGainItWillHold) {
  // A new runtime starts at the gain it will hold, so a replacement crossfaded in next to a
  // running one does not glide in from unity.
  const auto model = edited(loadUnity(), [](Model& m) {
    m.loudnessDb = -26.0;  // +8 dB toward the -18 target
  });
  ProcessContext ctx;
  ctx.targetLoudnessDb = -18.0f;

  Rig h;
  auto rt = h.build(model, 1);
  ASSERT_NE(rt, nullptr);
  const auto input = makeSine(kBlock, 440.0, 0.1f);
  std::vector<std::vector<float>> io = {input};
  h.process(*rt, io, 0, kBlock, ctx);
  const float gain = juce::Decibels::decibelsToGain(8.0f);
  for (size_t i = 0; i < input.size(); ++i)
    ASSERT_NEAR(io[0][i], input[i] * gain, 1.0e-6) << "sample " << i;
}

TEST(Runtime, GainChangesGlideOverFiftyMilliseconds) {
  const auto model = edited(loadUnity(), [](Model& m) { m.loudnessDb = -26.0; });
  ProcessContext ctx;
  ctx.targetLoudnessDb = -18.0f;

  Rig h;
  auto rt = h.build(model, 1);
  ASSERT_NE(rt, nullptr);
  const std::vector<float> ones(static_cast<size_t>(kBlock) * 8, 1.0f);
  std::vector<std::vector<float>> settled = {ones};
  h.process(*rt, settled, 0, kBlock, ctx);  // primes at +8 dB
  const float from = juce::Decibels::decibelsToGain(8.0f);
  EXPECT_NEAR(settled[0][kBlock - 1], from, 1.0e-6);

  // Switching normalization off moves the target to unity: the gain ramps
  // linearly down over 50 ms (2400 samples at 48 kHz) instead of stepping.
  std::vector<std::vector<float>> ramp = {ones};
  for (int offset = 0; offset < static_cast<int>(ones.size()); offset += kBlock)
    h.process(*rt, ramp, offset, kBlock, ctx, /*normalize=*/false);
  EXPECT_LT(ramp[0][0], from);
  EXPECT_GT(ramp[0][0], from - 0.01f) << "the first sample moves by one ramp step only";
  EXPECT_GT(ramp[0][1200], 1.0f);
  EXPECT_LT(ramp[0][1200], from);
  EXPECT_NEAR(ramp[0][2600], 1.0f, 1.0e-6) << "settled after the 50 ms ramp";
  for (size_t i = 1; i < ramp[0].size(); ++i)
    ASSERT_LE(ramp[0][i], ramp[0][i - 1] + 1.0e-7f) << "monotonic glide at " << i;
}

// ---------------------------------------------------------------------------
// Engine failure
// ---------------------------------------------------------------------------

TEST(Runtime, EngineExceptionPassesDry) {
  const auto model = loadUnity();

  Rig h;
  auto rt = h.build(model, 1);
  ASSERT_NE(rt, nullptr);
  ProcessContext ctx;

  const int n = kBlock * 4;
  const auto input = material(n, 111);

  // Healthy: the model is a pass-through, so the runtime is audibly transparent.
  std::vector<std::vector<float>> io = {input};
  h.process(*rt, io, 0, kBlock, ctx, /*normalize=*/false);
  EXPECT_FALSE(rt->hasFailed());

  // The engine throws: the failure is flagged and the runtime passes the dry
  // input from then on.
  gThrowingDspThrows = true;
  h.process(*rt, io, kBlock, kBlock, ctx, false);
  gThrowingDspThrows = false;
  EXPECT_TRUE(rt->hasFailed());
  for (int i = kBlock; i < 2 * kBlock; ++i)
    ASSERT_EQ(io[0][static_cast<size_t>(i)], input[static_cast<size_t>(i)]) << i;
  // Even with the engine healthy again, a failed runtime stays dry until it is
  // rebuilt.
  std::vector<std::vector<float>> later = {input};
  h.process(*rt, later, 2 * kBlock, kBlock, ctx, false);
  for (int i = 2 * kBlock; i < 3 * kBlock; ++i)
    ASSERT_EQ(later[0][static_cast<size_t>(i)], input[static_cast<size_t>(i)]) << i;
}

// With calibration and normalization in play: the input calibration gain is applied to the buffer
// before the engine runs, so a mid-block throw must hand back the untouched input, on both voices,
// and the later blocks must skip the post-model gain too.
TEST(Runtime, EngineExceptionRestoresTheDryInputAfterCalibration) {
  const auto model = edited(loadUnity(), [](Model& m) {
    m.inputLevelDbu = 0.0;  // calibration 12 dBu would drive it +12 dB
    m.loudnessDb = -30.0;
  });
  ProcessContext ctx;
  ctx.calibrateInput = true;
  ctx.inputCalibrationDbu = 12.0f;
  ctx.targetLoudnessDb = -18.0f;

  Rig h;
  auto rt = h.build(model, 2);
  ASSERT_NE(rt, nullptr);
  const auto left = material(kBlock * 2, 121);
  const auto right = noise(kBlock * 2, 122, 0.3f);
  std::vector<std::vector<float>> io = {left, right};

  gThrowingDspThrows = true;
  h.process(*rt, io, 0, kBlock, ctx);
  gThrowingDspThrows = false;
  ASSERT_TRUE(rt->hasFailed());
  for (int i = 0; i < kBlock; ++i) {
    ASSERT_EQ(io[0][static_cast<size_t>(i)], left[static_cast<size_t>(i)]) << i;
    ASSERT_EQ(io[1][static_cast<size_t>(i)], right[static_cast<size_t>(i)]) << i;
  }
  h.process(*rt, io, kBlock, kBlock, ctx);
  for (int i = kBlock; i < 2 * kBlock; ++i) {
    ASSERT_EQ(io[0][static_cast<size_t>(i)], left[static_cast<size_t>(i)]) << i;
    ASSERT_EQ(io[1][static_cast<size_t>(i)], right[static_cast<size_t>(i)]) << i;
  }
}
