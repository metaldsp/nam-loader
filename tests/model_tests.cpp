// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// Model: models are decoded and validated once off the audio thread into immutable facts that
// every engine shares. These tests pin what the loader reads from the A2 captures and NAM core's
// example models, that bad input is rejected with a reason, what the a2 and phaseSafe flags say
// for each architecture, and that loading from a juce::File survives a non-ASCII path.

#include <gtest/gtest.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "json.hpp"
#include "nam_loader/Model.h"
#include "test_helpers.h"

using namloader::loadModel;
using namloader::Model;

namespace {

std::vector<std::uint8_t> bytesOfFile(const juce::File& file) {
  juce::MemoryBlock block;
  EXPECT_TRUE(file.loadFileAsData(block));
  const auto* data = static_cast<const std::uint8_t*>(block.getData());
  return std::vector<std::uint8_t>(data, data + block.getSize());
}

std::vector<std::uint8_t> bytesOfCapture(const char* fixture) {
  return bytesOfFile(testFile(fixture));
}

std::vector<std::uint8_t> bytesOfJson(const nlohmann::json& json) {
  const std::string dumped = json.dump();
  return std::vector<std::uint8_t>(dumped.begin(), dumped.end());
}

nlohmann::json parseFile(const juce::File& file) {
  const auto bytes = bytesOfFile(file);
  return nlohmann::json::parse(bytes.begin(), bytes.end());
}

// The captures are the only fixtures with a known metadata shape, so metadata a test needs that a
// model lacks (or must lack) is patched into the parsed JSON instead of adding binary fixtures.
template <typename Edit>
std::vector<std::uint8_t> bytesWithMetadata(const char* fixture, Edit&& edit) {
  nlohmann::json json = parseFile(testFile(fixture));
  edit(json["metadata"]);
  return bytesOfJson(json);
}

std::shared_ptr<const Model> loadBytes(const std::vector<std::uint8_t>& bytes, const char* what) {
  juce::String error;
  auto model = loadModel(bytes, error);
  EXPECT_NE(model, nullptr) << what << ": " << error;
  return model;
}

std::shared_ptr<const Model> loadCapture(const char* fixture) {
  return loadBytes(bytesOfCapture(fixture), fixture);
}

std::shared_ptr<const Model> loadExample(const char* name) {
  return loadBytes(bytesOfFile(exampleModel(name)), name);
}

// A WAV in memory, so a non-model file can be fed to the loader without committing audio
// fixtures.
std::vector<std::uint8_t> makeWav(int channels, int samples, double rate, unsigned seed = 1) {
  juce::MemoryBlock block;
  {
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream =
        std::make_unique<juce::MemoryOutputStream>(block, false);
    const auto options = juce::AudioFormatWriterOptions{}
                             .withSampleRate(rate)
                             .withNumChannels(channels)
                             .withBitsPerSample(16);
    std::unique_ptr<juce::AudioFormatWriter> writer = wav.createWriterFor(stream, options);
    EXPECT_NE(writer, nullptr);
    juce::AudioBuffer<float> audio(channels, samples);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int ch = 0; ch < channels; ++ch)
      for (int i = 0; i < samples; ++i)
        audio.setSample(ch, i, dist(rng));
    writer->writeFromAudioSampleBuffer(audio, 0, samples);
  }
  const auto* data = static_cast<const std::uint8_t*>(block.getData());
  return std::vector<std::uint8_t>(data, data + block.getSize());
}

// A scratch directory removed when the test ends.
struct TempDir {
  TempDir()
      : dir(juce::File::getSpecialLocation(juce::File::tempDirectory)
                .getChildFile("namloader-" + juce::Uuid().toDashedString())) {
    EXPECT_TRUE(dir.createDirectory());
  }
  ~TempDir() { dir.deleteRecursively(); }
  juce::File dir;
};

}  // namespace

TEST(Model, CaptureWithLevelsReportsThem) {
  const auto model = loadCapture("a2-amp-cab-test.nam");
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->data, nullptr);
  EXPECT_EQ(model->architecture, "SlimmableContainer");
  ASSERT_TRUE(model->inputLevelDbu.has_value());
  EXPECT_DOUBLE_EQ(*model->inputLevelDbu, 15.0);
  ASSERT_TRUE(model->outputLevelDbu.has_value());
  EXPECT_DOUBLE_EQ(*model->outputLevelDbu, 14.0);
  ASSERT_TRUE(model->loudnessDb.has_value());
  EXPECT_NEAR(*model->loudnessDb, -15.927188, 1.0e-5);
  EXPECT_DOUBLE_EQ(model->sampleRate, 48000.0);
  EXPECT_GT(model->prewarmSamples, 0);
  // A SlimmableContainer of WaveNets is phase-interleavable.
  EXPECT_TRUE(model->phaseSafe);
  EXPECT_TRUE(model->a2);
}

TEST(Model, CaptureWithoutLevelsHasNone) {
  const auto model = loadCapture("a2-amp-test.nam");
  ASSERT_NE(model, nullptr);
  EXPECT_FALSE(model->inputLevelDbu.has_value());
  EXPECT_FALSE(model->outputLevelDbu.has_value());
  ASSERT_TRUE(model->loudnessDb.has_value());
  EXPECT_NEAR(*model->loudnessDb, -14.129041, 1.0e-5);
}

TEST(Model, EachLevelIsReportedIndependently) {
  // NAM stores levels as floats, so the patched values are exactly representable.
  const auto inputOnly = loadBytes(
      bytesWithMetadata("a2-amp-test.nam",
                        [](nlohmann::json& metadata) { metadata["input_level_dbu"] = 15.0; }),
      "input level only");
  ASSERT_NE(inputOnly, nullptr);
  ASSERT_TRUE(inputOnly->inputLevelDbu.has_value());
  EXPECT_DOUBLE_EQ(*inputOnly->inputLevelDbu, 15.0);
  EXPECT_FALSE(inputOnly->outputLevelDbu.has_value());

  const auto outputOnly = loadBytes(
      bytesWithMetadata("a2-amp-cab-test.nam",
                        [](nlohmann::json& metadata) { metadata.erase("input_level_dbu"); }),
      "output level only");
  ASSERT_NE(outputOnly, nullptr);
  EXPECT_FALSE(outputOnly->inputLevelDbu.has_value());
  ASSERT_TRUE(outputOnly->outputLevelDbu.has_value());
  EXPECT_DOUBLE_EQ(*outputOnly->outputLevelDbu, 14.0);
  EXPECT_TRUE(outputOnly->loudnessDb.has_value());
}

TEST(Model, ExampleWaveNetReportsItsLevels) {
  const auto model = loadExample("wavenet.nam");
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->architecture, "WaveNet");
  ASSERT_TRUE(model->inputLevelDbu.has_value());
  ASSERT_TRUE(model->outputLevelDbu.has_value());
  EXPECT_NEAR(*model->inputLevelDbu, 18.3, 1.0e-4);
  EXPECT_NEAR(*model->outputLevelDbu, 12.3, 1.0e-4);
  EXPECT_GT(model->prewarmSamples, 0);
}

TEST(Model, DataBuildsIndependentInstancesFromCopies) {
  // get_dsp(dspData&) consumes the weights it is given, so each engine instance is created from
  // a copy; the cached data must stay whole. (A SlimmableContainer keeps its weights inside the
  // submodel configs.)
  const auto model = loadCapture("a2-amp-test.nam");
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->data, nullptr);
  EXPECT_EQ(model->data->architecture, "SlimmableContainer");
  EXPECT_FALSE(model->data->config.empty());

  const auto before = model->data->config.dump().size();
  for (int i = 0; i < 3; ++i) {
    nam::dspData copy = *model->data;
    const auto dsp = nam::get_dsp(copy);
    ASSERT_NE(dsp, nullptr);
    EXPECT_EQ(dsp->NumInputChannels(), 1);
    EXPECT_EQ(dsp->NumOutputChannels(), 1);
  }
  EXPECT_EQ(model->data->config.dump().size(), before);
}

TEST(Model, InvalidBytesAreRejectedWithAReason) {
  const auto reject = [](const std::vector<std::uint8_t>& bytes) {
    juce::String error;
    EXPECT_EQ(loadModel(bytes, error), nullptr);
    EXPECT_TRUE(error.isNotEmpty());
  };
  reject({});
  reject({'n', 'o', 't', ' ', 'j', 's', 'o', 'n'});
  reject({'{', '}'});
  reject({'{', '"', 'v', 'e', 'r', 's', 'i', 'o', 'n', '"', ':', '1', '}'});
  reject(makeWav(1, 4800, 48000.0));  // audio is not a model
}

TEST(Model, A2FlagIsTrueForA2Models) {
  EXPECT_TRUE(loadCapture("a2-amp-test.nam")->a2);
  EXPECT_TRUE(loadCapture("a2-amp-cab-test.nam")->a2);
  EXPECT_TRUE(loadCapture("a2-am-test-2.nam")->a2);
  EXPECT_TRUE(loadExample("A2.nam")->a2);
}

TEST(Model, A2FlagIsFalseForOtherModels) {
  // Valid NAM files that are not A2.
  EXPECT_FALSE(loadExample("lstm.nam")->a2);
  EXPECT_FALSE(loadExample("wavenet.nam")->a2) << "a WaveNet that is not the A2 shape";
  EXPECT_FALSE(loadExample("slimmable_container.nam")->a2)
      << "a container whose submodels are not A2";
}

TEST(Model, A2BareWaveNetIsA2) {
  // The A2 WaveNet inside a container passes as a model on its own too.
  nlohmann::json bare = parseFile(exampleModel("A2.nam"))["config"]["submodels"][0]["model"];
  ASSERT_EQ(bare["architecture"], "WaveNet");
  const auto model = loadBytes(bytesOfJson(bare), "bare A2 WaveNet");
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->architecture, "WaveNet");
  EXPECT_TRUE(model->a2);
  EXPECT_TRUE(model->phaseSafe);
}

TEST(Model, ContainerIsA2OnlyWhenEverySubmodelIs) {
  // One A2 submodel swapped for a non-A2 WaveNet spoils the whole container (the engine
  // switches tiers, so any submodel can end up running). The container is still phase-safe: both
  // submodels are WaveNets.
  nlohmann::json mixed = parseFile(exampleModel("A2.nam"));
  ASSERT_GE(mixed["config"]["submodels"].size(), 2u);
  mixed["config"]["submodels"][1]["model"] = parseFile(exampleModel("wavenet.nam"));
  const auto model = loadBytes(bytesOfJson(mixed), "mixed container");
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->architecture, "SlimmableContainer");
  EXPECT_FALSE(model->a2);
  EXPECT_TRUE(model->phaseSafe);
}

TEST(Model, PhaseSafeFollowsTheArchitecture) {
  EXPECT_TRUE(loadExample("wavenet.nam")->phaseSafe) << "WaveNet";
  EXPECT_TRUE(loadExample("A2.nam")->phaseSafe) << "container of WaveNets";
  EXPECT_FALSE(loadExample("slimmable_container.nam")->phaseSafe)
      << "container with an LSTM submodel";
  EXPECT_FALSE(loadExample("lstm.nam")->phaseSafe) << "LSTM is recurrent";
}

TEST(Model, ContainerIsPhaseSafeOnlyWhenEverySubmodelIs) {
  // Swap one WaveNet submodel for an LSTM: the container can switch to it at runtime, so it can
  // no longer be phase-interleaved.
  nlohmann::json mixed = parseFile(exampleModel("A2.nam"));
  ASSERT_GE(mixed["config"]["submodels"].size(), 2u);
  mixed["config"]["submodels"][1]["model"] = parseFile(exampleModel("lstm.nam"));
  const auto model = loadBytes(bytesOfJson(mixed), "container with an LSTM");
  ASSERT_NE(model, nullptr);
  EXPECT_FALSE(model->phaseSafe);
  EXPECT_FALSE(model->a2);
}

TEST(Model, LoadsFromAFile) {
  juce::String error;
  const auto model = loadModel(testFile("a2-amp-cab-test.nam"), error);
  ASSERT_NE(model, nullptr) << error;
  EXPECT_EQ(model->architecture, "SlimmableContainer");
  ASSERT_TRUE(model->loudnessDb.has_value());
  EXPECT_NEAR(*model->loudnessDb, -15.927188, 1.0e-5);
}

TEST(Model, LoadsFromAFileWithANonAsciiPath) {
  // The file is read into bytes rather than handed to NAM by path, which is what keeps a UTF-8
  // path from being mis-decoded on Windows.
  TempDir temp;
  const auto copy = temp.dir.getChildFile(juce::String::fromUTF8("amp\xC3\xA8re \xC3\xBC.nam"));
  ASSERT_TRUE(testFile("a2-amp-cab-test.nam").copyFileTo(copy));
  ASSERT_TRUE(copy.existsAsFile());

  juce::String error;
  const auto model = loadModel(copy, error);
  ASSERT_NE(model, nullptr) << error;
  EXPECT_TRUE(model->a2);
  ASSERT_TRUE(model->inputLevelDbu.has_value());
  EXPECT_DOUBLE_EQ(*model->inputLevelDbu, 15.0);
}

TEST(Model, FileFailuresComeWithAReason) {
  TempDir temp;
  {
    juce::String error;
    EXPECT_EQ(loadModel(temp.dir.getChildFile("missing.nam"), error), nullptr);
    EXPECT_TRUE(error.isNotEmpty());
  }
  {
    juce::String error;
    EXPECT_EQ(loadModel(temp.dir, error), nullptr) << "a directory is not a model";
    EXPECT_TRUE(error.isNotEmpty());
  }
  {
    const auto notModel = temp.dir.getChildFile("not-a-model.nam");
    ASSERT_TRUE(notModel.replaceWithText("not json"));
    juce::String error;
    EXPECT_EQ(loadModel(notModel, error), nullptr);
    EXPECT_TRUE(error.isNotEmpty());
  }
}
