// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// Engine: the phase-interleaved, multi-voice NAM host.
//
//   Engine (parameterised)  real A2 captures plus a plain WaveNet: load/run sanity, slicing
//                           invariance, exactness of the phase-interleaved oversampled mode
//                           against a hand-built reference, voices, and bit-exact pool forks.
//   Engine.Constructor*     the constructor's voice x phase validation.
//   Engine.Create*          Engine::create: phases, voices, bad arguments, slimmable size.
//   Engine.Oversampling*    the headline claim: oversampling reduces a real amp's aliasing.

#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "NAM/get_dsp.h"
#include "json.hpp"
#include "nam_loader/Engine.h"
#include "nam_loader/Model.h"
#include "nam_loader/Oversampler.h"
#include "nam_loader/WorkerPool.h"
#include "test_helpers.h"

using namloader::Engine;
using namloader::Model;
using namloader::WorkerPool;

namespace {

// Streams `in` through an Oversampler in `blockSize` chunks; `modelFn(channels, frames)` processes
// the oversampled stereo block in place. Returns the base-rate output (channel 0).
template <typename Fn>
std::vector<float> runOversampled(const std::vector<float>& in,
                                  int factor,
                                  int blockSize,
                                  Fn&& modelFn) {
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
      modelFn(co, n);
    });
  }
  return out;
}

// Builds model instances from a .nam file the way Engine::create does (nam::get_dsp per instance,
// all from the same parsed config): `instanceCount` phases per voice, `voices` voices
// (voice-major layout). Going through the constructor directly keeps the tests free to build
// layouts create() would never pick.
std::unique_ptr<Engine> makeEngine(const juce::File& modelFile,
                                   int oversampleFactor,
                                   int instanceCount,
                                   int voices = 1) {
  juce::MemoryBlock bytes;
  EXPECT_TRUE(modelFile.loadFileAsData(bytes));
  const auto* begin = static_cast<const char*>(bytes.getData());
  const nlohmann::json config = nlohmann::json::parse(begin, begin + bytes.getSize());

  std::vector<std::unique_ptr<nam::DSP>> instances;
  for (int i = 0; i < instanceCount * voices; ++i) {
    auto dsp = nam::get_dsp(config);
    EXPECT_NE(dsp, nullptr);
    instances.push_back(std::move(dsp));
  }
  return std::make_unique<Engine>(std::move(instances), oversampleFactor, voices);
}

struct ModelParam {
  const char* file;
  bool example;  // false: a capture in tests/files; true: one of NAM core's example models
  const char* label;
};

juce::File modelFile(const ModelParam& param) {
  return param.example ? exampleModel(param.file) : testFile(param.file);
}

class EngineTest : public ::testing::TestWithParam<ModelParam> {};

INSTANTIATE_TEST_SUITE_P(Engine,
                         EngineTest,
                         ::testing::Values(ModelParam{"a2-amp-cab-test.nam", false, "AmpCab"},
                                           ModelParam{"a2-amp-test.nam", false, "AmpHead"},
                                           ModelParam{"wavenet.nam", true, "WaveNet"}),
                         [](const auto& info) { return info.param.label; });

}  // namespace

TEST_P(EngineTest, LoadsAndProducesAmpedAudio) {
  auto engine = makeEngine(modelFile(GetParam()), 1, 1);
  EXPECT_NEAR(engine->getInstanceSampleRate(), kFs, 0.1) << "one phase at factor 1 runs at 48 kHz";

  engine->prepare(512);
  juce::AudioBuffer<float> buffer(2, 512);
  const auto in = makeSine(512, 220.0, 0.25f);
  buffer.copyFrom(0, 0, in.data(), 512);
  buffer.copyFrom(1, 0, in.data(), 512);
  engine->process(buffer);

  float peak = 0.0f, diff = 0.0f;
  for (int i = 0; i < 512; ++i) {
    const float s = buffer.getSample(0, i);
    ASSERT_TRUE(std::isfinite(s)) << "at " << i;
    peak = std::max(peak, std::abs(s));
    diff = std::max(diff, std::abs(s - in[static_cast<size_t>(i)]));
    // Mono model fans out: both channels identical.
    ASSERT_EQ(s, buffer.getSample(1, i)) << "at " << i;
  }
  EXPECT_GT(peak, 1e-4f) << "model produced silence";
  EXPECT_GT(diff, 1e-4f) << "model passed audio through untouched";
}

TEST_P(EngineTest, SlicedProcessingMatchesOneShot) {
  // NAM models stream statefully, so process() slicing (engine prepared
  // smaller than the buffer it receives) must be inaudible; this is the
  // defensive path that keeps the engine alive when a startup prepare raced the
  // host's real block size.
  const int total = 4096;
  const auto in = makeNoise(total, 1234, 0.5f);

  auto oneShot = makeEngine(modelFile(GetParam()), 1, 1);
  oneShot->prepare(total);
  juce::AudioBuffer<float> bufferA(1, total);
  bufferA.copyFrom(0, 0, in.data(), total);
  oneShot->process(bufferA);

  auto sliced = makeEngine(modelFile(GetParam()), 1, 1);
  sliced->prepare(512);  // forces internal 512-frame slicing of the same 4096
  juce::AudioBuffer<float> bufferB(1, total);
  bufferB.copyFrom(0, 0, in.data(), total);
  sliced->process(bufferB);

  float maxDiff = 0.0f;
  for (int i = 0; i < total; ++i)
    maxDiff = std::max(maxDiff, std::abs(bufferA.getSample(0, i) - bufferB.getSample(0, i)));
  EXPECT_LE(maxDiff, 2e-6f) << "slicing changed the output";
}

TEST_P(EngineTest, PhaseInterleaveMatchesManualPhaseReference) {
  // The oversampled engine (N instances fed every Nth sample) must compute
  // exactly what N hand-driven copies of the model compute on hand-split
  // phase streams; this pins the de-interleave/re-interleave bookkeeping.
  const int factor = 4, total = 4096;
  const auto in = makeNoise(total, 4321, 0.5f);

  auto engine = makeEngine(modelFile(GetParam()), factor, factor);
  engine->prepare(total);
  juce::AudioBuffer<float> buffer(1, total);
  buffer.copyFrom(0, 0, in.data(), total);
  engine->process(buffer);

  // Reference: fresh instances, same prewarm geometry as Engine::prepare
  // (per-phase capacity total/factor + 1 at the base rate).
  juce::MemoryBlock bytes;
  ASSERT_TRUE(modelFile(GetParam()).loadFileAsData(bytes));
  const auto* begin = static_cast<const char*>(bytes.getData());
  const nlohmann::json config = nlohmann::json::parse(begin, begin + bytes.getSize());

  const int perPhase = total / factor;
  std::vector<float> reference(static_cast<size_t>(total), 0.0f);
  for (int p = 0; p < factor; ++p) {
    auto dsp = nam::get_dsp(config);
    ASSERT_NE(dsp, nullptr);
    dsp->ResetAndPrewarm(kFs, perPhase + 1);

    std::vector<double> phaseIn(static_cast<size_t>(perPhase)),
        phaseOut(static_cast<size_t>(perPhase));
    for (int i = 0; i < perPhase; ++i)
      phaseIn[static_cast<size_t>(i)] =
          static_cast<double>(in[static_cast<size_t>(i * factor + p)]);
    NAM_SAMPLE* ins[] = {phaseIn.data()};
    NAM_SAMPLE* outs[] = {phaseOut.data()};
    dsp->process(ins, outs, perPhase);
    for (int i = 0; i < perPhase; ++i)
      reference[static_cast<size_t>(i * factor + p)] =
          static_cast<float>(phaseOut[static_cast<size_t>(i)]);
  }

  for (int i = 0; i < total; ++i)
    ASSERT_EQ(buffer.getSample(0, i), reference[static_cast<size_t>(i)]) << "at " << i;
}

TEST_P(EngineTest, PhaseForkMatchesSerialBitExact) {
  // Pool-forked phases (see Engine::process) against the serial phase
  // loop, deliberately on the ugliest geometry: chunk sizes that aren't
  // divisible by the factor (the phaseOffset carry) and an engine prepared
  // smaller than its buffers (internal defensive slicing). Any scheduling
  // dependence (a shared buffer, a phase touching a sibling's stream) shows
  // up as a bit mismatch.
  WorkerPool pool;
  pool.start(kFs, 512);

  for (const int factor : {2, 4, 8}) {
    SCOPED_TRACE("factor " + juce::String(factor));
    // 313 is odd and coprime to every factor, so the phase offset lands
    // differently on every chunk; 256 < 313 forces the internal slice loop.
    const int chunk = 313, chunks = 24, total = chunk * chunks;
    const auto in = makeNoise(total, 55501 + factor, 0.5f);

    auto run = [&](WorkerPool* forkPool) {
      auto engine = makeEngine(modelFile(GetParam()), factor, factor);
      engine->prepare(256);
      std::vector<float> out(in.begin(), in.end());
      for (int off = 0; off < total; off += chunk) {
        float* ptr = out.data() + off;
        juce::AudioBuffer<float> view(&ptr, 1, chunk);
        engine->process(view, forkPool);
      }
      return out;
    };

    const auto serial = run(nullptr);
    const auto forked = run(&pool);
    for (int i = 0; i < total; ++i)
      ASSERT_EQ(serial[static_cast<size_t>(i)], forked[static_cast<size_t>(i)]) << "at " << i;
  }
}

// Two voices for a stereo input (see the Engine.h class comment): a two-voice
// engine fed a stereo buffer with the second-voice flag set (`dualMono` in
// the engine API) must compute, on each channel, exactly what a fresh
// single-voice engine computes on that channel alone. A caller builds such
// an engine for a stereo input; this is the whole definition of that at the
// engine level.
TEST_P(EngineTest, TwoVoicesProcessStereoChannelsIndependently) {
  const int total = 4096;
  const auto inL = makeNoise(total, 7001, 0.5f);
  const auto inR = makeNoise(total, 7002, 0.5f);

  auto dual = makeEngine(modelFile(GetParam()), 1, 1, /*voices=*/2);
  EXPECT_EQ(dual->getVoiceCount(), 2);
  EXPECT_EQ(dual->getPhaseCount(), 1);
  dual->prepare(512);
  juce::AudioBuffer<float> stereo(2, total);
  stereo.copyFrom(0, 0, inL.data(), total);
  stereo.copyFrom(1, 0, inR.data(), total);
  dual->process(stereo, nullptr, /*dualMono=*/true);

  auto reference = [&](const std::vector<float>& in) {
    auto engine = makeEngine(modelFile(GetParam()), 1, 1);
    engine->prepare(512);
    juce::AudioBuffer<float> mono(1, total);
    mono.copyFrom(0, 0, in.data(), total);
    engine->process(mono);
    return std::vector<float>(mono.getReadPointer(0), mono.getReadPointer(0) + total);
  };
  const auto refL = reference(inL);
  const auto refR = reference(inR);

  float lrDiff = 0.0f;
  for (int i = 0; i < total; ++i) {
    ASSERT_EQ(stereo.getSample(0, i), refL[static_cast<size_t>(i)]) << "left at " << i;
    ASSERT_EQ(stereo.getSample(1, i), refR[static_cast<size_t>(i)]) << "right at " << i;
    lrDiff = std::max(lrDiff, std::abs(stereo.getSample(0, i) - stereo.getSample(1, i)));
  }
  EXPECT_GT(lrDiff, 1e-3f) << "the two channels should carry different signals";
}

// Without the flag, or on a mono buffer, a two-voice engine is
// indistinguishable from a single-voice one: voice 0 processes channel 0
// and fans out. Pins that the per-callback gating (a mono buffer reaching an
// engine built with two voices) never pays for the idle voice and never
// changes the sound.
TEST_P(EngineTest, TwoVoiceEngineFallsBackToFanOut) {
  const int total = 2048;
  const auto inL = makeNoise(total, 7003, 0.5f);
  const auto inR = makeNoise(total, 7004, 0.5f);

  auto single = makeEngine(modelFile(GetParam()), 1, 1);
  single->prepare(512);
  juce::AudioBuffer<float> ref(1, total);
  ref.copyFrom(0, 0, inL.data(), total);
  single->process(ref);

  {
    SCOPED_TRACE("stereo buffer, second-voice flag off");
    auto dual = makeEngine(modelFile(GetParam()), 1, 1, 2);
    dual->prepare(512);
    juce::AudioBuffer<float> stereo(2, total);
    stereo.copyFrom(0, 0, inL.data(), total);
    stereo.copyFrom(1, 0, inR.data(), total);
    dual->process(stereo, nullptr, false);
    for (int i = 0; i < total; ++i) {
      ASSERT_EQ(stereo.getSample(0, i), ref.getSample(0, i)) << "at " << i;
      ASSERT_EQ(stereo.getSample(1, i), ref.getSample(0, i)) << "fan-out at " << i;
    }
  }
  {
    SCOPED_TRACE("mono buffer, second-voice flag on");
    auto dual = makeEngine(modelFile(GetParam()), 1, 1, 2);
    dual->prepare(512);
    juce::AudioBuffer<float> mono(1, total);
    mono.copyFrom(0, 0, inL.data(), total);
    dual->process(mono, nullptr, true);
    for (int i = 0; i < total; ++i)
      ASSERT_EQ(mono.getSample(0, i), ref.getSample(0, i)) << "at " << i;
  }
  {
    SCOPED_TRACE("single-voice engine ignores the second-voice flag");
    auto one = makeEngine(modelFile(GetParam()), 1, 1);
    one->prepare(512);
    juce::AudioBuffer<float> stereo(2, total);
    stereo.copyFrom(0, 0, inL.data(), total);
    stereo.copyFrom(1, 0, inR.data(), total);
    one->process(stereo, nullptr, true);
    for (int i = 0; i < total; ++i) {
      ASSERT_EQ(stereo.getSample(0, i), ref.getSample(0, i)) << "at " << i;
      ASSERT_EQ(stereo.getSample(1, i), ref.getSample(0, i)) << "fan-out at " << i;
    }
  }
}

// The flat voice × phase fork against the serial loop, on the same ugly
// chunk geometry as the phase-only test, at every factor up to the pool's
// 16-job limit (2 voices × 8 phases). Any job touching a sibling's stream
// (the one hazard a voice-major slot layout could introduce) shows up as a
// bit mismatch on one channel.
TEST_P(EngineTest, TwoVoiceForkMatchesSerialBitExact) {
  WorkerPool pool;
  pool.start(kFs, 512);

  for (const int factor : {1, 2, 4, 8}) {
    SCOPED_TRACE("factor " + juce::String(factor));
    const int chunk = 313, chunks = 24, total = chunk * chunks;
    const auto inL = makeNoise(total, 60001 + factor, 0.5f);
    const auto inR = makeNoise(total, 61001 + factor, 0.5f);

    auto run = [&](WorkerPool* forkPool) {
      auto engine = makeEngine(modelFile(GetParam()), factor, factor, 2);
      engine->prepare(256);
      std::vector<float> l(inL.begin(), inL.end()), r(inR.begin(), inR.end());
      for (int off = 0; off < total; off += chunk) {
        float* ptrs[] = {l.data() + off, r.data() + off};
        juce::AudioBuffer<float> view(ptrs, 2, chunk);
        engine->process(view, forkPool, true);
      }
      return std::make_pair(std::move(l), std::move(r));
    };

    const auto [sl, sr] = run(nullptr);
    const auto [pl, pr] = run(&pool);
    for (int i = 0; i < total; ++i) {
      ASSERT_EQ(sl[static_cast<size_t>(i)], pl[static_cast<size_t>(i)]) << "left at " << i;
      ASSERT_EQ(sr[static_cast<size_t>(i)], pr[static_cast<size_t>(i)]) << "right at " << i;
    }
  }
}

// Layout validation: the constructor is the only place the voice × phase
// geometry is checked, so every malformed shape must be rejected there
// rather than indexing out of range on the audio thread.
TEST(Engine, ConstructorRejectsMalformedVoiceLayouts) {
  juce::MemoryBlock bytes;
  ASSERT_TRUE(testFile("a2-amp-test.nam").loadFileAsData(bytes));
  const auto* begin = static_cast<const char*>(bytes.getData());
  const nlohmann::json config = nlohmann::json::parse(begin, begin + bytes.getSize());
  auto make = [&](int count) {
    std::vector<std::unique_ptr<nam::DSP>> instances;
    for (int i = 0; i < count; ++i)
      instances.push_back(nam::get_dsp(config));
    return instances;
  };

  EXPECT_THROW(Engine(make(1), 1, 0), std::invalid_argument) << "zero voices";
  EXPECT_THROW(Engine(make(3), 1, 3), std::invalid_argument) << "three voices";
  EXPECT_THROW(Engine(make(3), 2, 2), std::invalid_argument) << "count not a voice multiple";
  EXPECT_THROW(Engine(make(6), 2, 2), std::invalid_argument) << "phases neither 1 nor factor";
  EXPECT_THROW(Engine(make(1), 0, 1), std::invalid_argument) << "factor below 1";
  EXPECT_NO_THROW(Engine(make(2), 4, 2)) << "one phase per voice is valid at any factor";
  EXPECT_NO_THROW(Engine(make(8), 4, 2)) << "factor phases per voice";
  EXPECT_NO_THROW(Engine(make(16), 8, 2)) << "the 16-job maximum";
}

// Engine::create (the supported way to build an engine from a Model)

namespace {

std::shared_ptr<const Model> loadFor(const juce::File& file) {
  juce::String error;
  auto model = namloader::loadModel(file, error);
  EXPECT_NE(model, nullptr) << file.getFileName() << ": " << error;
  return model;
}

// `in` run through `engine` as one mono block.
std::vector<float> runMono(Engine& engine, const std::vector<float>& in) {
  juce::AudioBuffer<float> buffer(1, static_cast<int>(in.size()));
  buffer.copyFrom(0, 0, in.data(), static_cast<int>(in.size()));
  engine.process(buffer);
  return std::vector<float>(buffer.getReadPointer(0), buffer.getReadPointer(0) + in.size());
}

}  // namespace

TEST(Engine, CreatePhaseSafeModelGetsOnePhasePerFactor) {
  const auto model = loadFor(testFile("a2-amp-test.nam"));
  ASSERT_NE(model, nullptr);
  ASSERT_TRUE(model->phaseSafe);

  for (const int factor : {1, 2, 4, 8}) {
    SCOPED_TRACE("factor " + juce::String(factor));
    const auto engine = Engine::create(*model, factor, 1, 512 * factor);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getOversampleFactor(), factor);
    EXPECT_EQ(engine->getPhaseCount(), factor);
    EXPECT_EQ(engine->getVoiceCount(), 1);
    EXPECT_DOUBLE_EQ(engine->getInstanceSampleRate(), kFs) << "each phase runs at the base rate";
  }
}

TEST(Engine, CreateModelThatIsNotPhaseSafeRunsOneTimeScaledInstance) {
  const auto model = loadFor(exampleModel("lstm.nam"));
  ASSERT_NE(model, nullptr);
  ASSERT_FALSE(model->phaseSafe);

  const auto engine = Engine::create(*model, 4, 1, 2048);
  ASSERT_NE(engine, nullptr);
  EXPECT_EQ(engine->getOversampleFactor(), 4);
  EXPECT_EQ(engine->getPhaseCount(), 1);
  EXPECT_DOUBLE_EQ(engine->getInstanceSampleRate(), kFs * 4) << "the one instance runs at 4x";

  // It runs: prepared, finite, and the model changed the signal.
  const auto in = makeSine(2048, 220.0, 0.25f);
  const auto out = runMono(*engine, in);
  float diff = 0.0f;
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(std::isfinite(out[i])) << "at " << i;
    diff = std::max(diff, std::abs(out[i] - in[i]));
  }
  EXPECT_GT(diff, 1e-4f);
}

TEST(Engine, CreateVoicesSetTheInstanceLayout) {
  const auto model = loadFor(testFile("a2-amp-test.nam"));
  ASSERT_NE(model, nullptr);

  const auto mono = Engine::create(*model, 2, 1, 1024);
  ASSERT_NE(mono, nullptr);
  EXPECT_EQ(mono->getVoiceCount(), 1);

  const auto dual = Engine::create(*model, 8, 2, 4096);  // 16 instances: the pool's job limit
  ASSERT_NE(dual, nullptr);
  EXPECT_EQ(dual->getVoiceCount(), 2);
  EXPECT_EQ(dual->getPhaseCount(), 8);
}

TEST(Engine, CreateReturnsNullForBadArguments) {
  const auto model = loadFor(testFile("a2-amp-test.nam"));
  ASSERT_NE(model, nullptr);

  EXPECT_EQ(Engine::create(*model, 0, 1, 512), nullptr) << "factor below 1";
  EXPECT_EQ(Engine::create(*model, -2, 1, 512), nullptr) << "negative factor";
  EXPECT_EQ(Engine::create(*model, 1, 0, 512), nullptr) << "zero voices";
  EXPECT_EQ(Engine::create(*model, 1, 3, 512), nullptr) << "three voices";
  EXPECT_EQ(Engine::create(*model, 1, 1, 0), nullptr) << "zero block size";
  EXPECT_EQ(Engine::create(*model, 1, 1, -1), nullptr) << "negative block size";
  EXPECT_EQ(Engine::create(Model{}, 1, 1, 512), nullptr) << "a model with no data";
  EXPECT_NE(Engine::create(*model, 1, 1, 512), nullptr) << "the same arguments are fine";
}

TEST(Engine, CreateMatchesAHandBuiltEngineBitExact) {
  // create() is the constructor plus per-instance copies of the weights; it must produce what
  // the hand-built layout produces, and the Model must stay whole so it can build another.
  const auto model = loadFor(testFile("a2-amp-cab-test.nam"));
  ASSERT_NE(model, nullptr);
  const int factor = 4, total = 4096;
  const auto in = makeNoise(total, 9001, 0.5f);

  auto manual = makeEngine(testFile("a2-amp-cab-test.nam"), factor, factor);
  manual->prepare(total);
  const auto expected = runMono(*manual, in);

  for (int round = 0; round < 2; ++round) {
    SCOPED_TRACE("round " + juce::String(round));
    const auto created = Engine::create(*model, factor, 1, total);
    ASSERT_NE(created, nullptr);
    const auto out = runMono(*created, in);
    for (int i = 0; i < total; ++i)
      ASSERT_EQ(out[static_cast<size_t>(i)], expected[static_cast<size_t>(i)]) << "at " << i;
  }
}

TEST(Engine, CreateSlimmableSizeIsApplied) {
  const auto model = loadFor(testFile("a2-amp-test.nam"));
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->architecture, "SlimmableContainer");
  const int total = 4096;
  const auto in = makeNoise(total, 9002, 0.5f);

  const auto full = Engine::create(*model, 1, 1, total);  // default: full
  const auto lite = Engine::create(*model, 1, 1, total, 0.0);
  ASSERT_NE(full, nullptr);
  ASSERT_NE(lite, nullptr);
  EXPECT_DOUBLE_EQ(full->getSlimmableSize(), 1.0);
  EXPECT_DOUBLE_EQ(lite->getSlimmableSize(), 0.0);

  const auto fullOut = runMono(*full, in);
  const auto liteOut = runMono(*lite, in);
  float diff = 0.0f;
  for (int i = 0; i < total; ++i)
    diff =
        std::max(diff, std::abs(fullOut[static_cast<size_t>(i)] - liteOut[static_cast<size_t>(i)]));
  EXPECT_GT(diff, 1e-4f) << "the lite and full tiers should sound different";

  // The request is clamped to [0, 1].
  EXPECT_DOUBLE_EQ(Engine::create(*model, 1, 1, total, 7.0)->getSlimmableSize(), 1.0);
  EXPECT_DOUBLE_EQ(Engine::create(*model, 1, 1, total, -1.0)->getSlimmableSize(), 0.0);
}

TEST(Engine, CreateSlimmableSizeChangesAfterPrepareAndFansOutToEveryPhase) {
  const auto model = loadFor(testFile("a2-amp-test.nam"));
  ASSERT_NE(model, nullptr);
  const int factor = 4, total = 4096;
  const auto in = makeNoise(total, 9003, 0.5f);

  // Created lite vs created full and switched to lite afterwards: every phase must follow, so
  // the output matches a lite engine's, up to the reset the tier switch performs.
  const auto lite = Engine::create(*model, factor, 1, total, 0.0);
  const auto switched = Engine::create(*model, factor, 1, total, 1.0);
  ASSERT_NE(lite, nullptr);
  ASSERT_NE(switched, nullptr);
  switched->setSlimmableSize(0.0);
  EXPECT_DOUBLE_EQ(switched->getSlimmableSize(), 0.0);

  const auto liteOut = runMono(*lite, in);
  const auto switchedOut = runMono(*switched, in);
  float maxDiff = 0.0f;
  for (int i = 0; i < total; ++i)
    maxDiff = std::max(
        maxDiff, std::abs(liteOut[static_cast<size_t>(i)] - switchedOut[static_cast<size_t>(i)]));
  EXPECT_LE(maxDiff, 1e-4f);
}

TEST(Engine, OversamplingReducesRealAmpAliasing) {
  // The headline claim, end to end with a real amp capture (the head-only
  // model; a cab's rolloff would bury the measurement): drive it with a hot
  // ~5 kHz tone and compare the energy at the fold-back frequencies of
  // harmonics 5..15 between factor 1 and factor 8.
  const juce::File model = testFile("a2-amp-test.nam");
  const int blockSize = 512, total = 93 * 512;
  const double f0 = 4998.046875;
  const auto in = makeSine(total, f0, 0.8f);

  std::vector<double> aliasFreqs;
  for (int h = 5; h <= 15; ++h)  // an amp makes even + odd harmonics
    aliasFreqs.push_back(foldFrequency(h * f0));

  auto aliasRatioAtFactor = [&](int factor) {
    auto engine = makeEngine(model, factor, factor);
    engine->prepare(blockSize * factor);
    const auto out = runOversampled(in, factor, blockSize, [&](float** ch, int n) {
      juce::AudioBuffer<float> block(ch, 2, n);
      engine->process(block);
    });
    std::vector<float> tail(out.begin() + 8192, out.begin() + 8192 + 16384);
    double aliasPower = 0.0;
    for (double f : aliasFreqs)
      aliasPower += goertzelPower(tail, f);
    return db(aliasPower) - db(goertzelPower(tail, f0));
  };

  const double alias1 = aliasRatioAtFactor(1);
  const double alias8 = aliasRatioAtFactor(8);
  std::printf(
      "  real amp: folded harmonics vs fundamental at factor 1: %+.1f dB, factor 8: "
      "%+.1f dB (%.1f dB reduction)\n",
      alias1, alias8, alias1 - alias8);
  EXPECT_LT(alias8, alias1 - 10.0) << "8x oversampling should cut real-amp aliasing by >10 dB";
}
