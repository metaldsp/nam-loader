// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "nam_loader/Model.h"

#include <algorithm>
#include <exception>
#include <string>

#include "NAM/get_dsp.h"
#include "NAM/wavenet/a2_fast.h"
#include "json.hpp"

namespace namloader {
namespace {

// Phase-interleaved oversampling eligibility (see Engine.h): exact only for architectures whose
// temporal structure is pure (dilated) convolution. LSTM and anything unknown run as one
// time-scaled instance instead. A container is safe only when every submodel is (it switches
// tiers at runtime).
bool configIsPhaseSafe(const std::string& architecture, const nlohmann::json& config) {
  if (architecture == "WaveNet" || architecture == "ConvNet" || architecture == "Linear")
    return true;
  if (architecture == "SlimmableContainer") {
    if (!config.is_object() || !config.contains("submodels"))
      return false;
    const auto& submodels = config["submodels"];
    if (!submodels.is_array() || submodels.empty())
      return false;
    for (const auto& entry : submodels) {
      if (!entry.contains("model") || !entry["model"].is_object())
        return false;
      const auto& model = entry["model"];
      if (!configIsPhaseSafe(model.value("architecture", std::string()),
                             model.contains("config") ? model["config"] : nlohmann::json()))
        return false;
    }
    return true;
  }
  return false;
}

// Architecture-2 models are the shape NAM's fast path and the slimmable tiers are tuned around
// (48 kHz training rate). Same container shape as configIsPhaseSafe: a bare A2 WaveNet, or a
// SlimmableContainer whose every submodel is one. `modelJson` is the whole .nam document
// (architecture + config).
bool configIsA2(const nlohmann::json& modelJson) {
  const std::string architecture = modelJson.value("architecture", "");
  if (architecture == "WaveNet") {
    int channels = 0;
    return modelJson.contains("config") &&
           nam::wavenet::a2_fast::is_a2_shape(modelJson["config"], &channels);
  }
  if (architecture == "SlimmableContainer") {
    if (!modelJson.contains("config") || !modelJson["config"].contains("submodels"))
      return false;
    const auto& submodels = modelJson["config"]["submodels"];
    if (!submodels.is_array() || submodels.empty())
      return false;
    for (const auto& entry : submodels) {
      if (!entry.contains("model") || !configIsA2(entry["model"]))
        return false;
    }
    return true;
  }
  return false;
}

}  // namespace

std::shared_ptr<const Model> loadModel(const std::vector<std::uint8_t>& bytes,
                                       juce::String& error) {
  if (bytes.empty()) {
    error = "The model file is empty";
    return nullptr;
  }

  try {
    // Parse the JSON straight from the bytes. Never round-trip through a temp file:
    // nam::get_dsp(std::filesystem::path) built from a JUCE UTF-8 string mis-decodes non-ASCII
    // characters on Windows.
    const nlohmann::json json = nlohmann::json::parse(bytes.begin(), bytes.end());

    // The weights are read once, here. get_dsp(dspData&) consumes the weights it is given, so
    // every engine instance is created from a copy of `data` (Engine::create), and this probe
    // instance only supplies the metadata and receptive field. Skipping the construction-time
    // prewarm keeps the load cheap; the probe is never run.
    auto data = std::make_shared<nam::dspData>();
    nam::DspLoadOptions options;
    options.prewarm = false;
    std::unique_ptr<nam::DSP> probe = nam::get_dsp(json, *data, options);
    if (probe == nullptr) {
      error = "The model could not be loaded";
      return nullptr;
    }
    if (probe->NumInputChannels() != 1) {
      error =
          "NAM model must have 1 input channel, but has " + juce::String(probe->NumInputChannels());
      return nullptr;
    }
    if (probe->NumOutputChannels() != 1) {
      error = "NAM model must have 1 output channel, but has " +
              juce::String(probe->NumOutputChannels());
      return nullptr;
    }

    auto model = std::make_shared<Model>();
    model->architecture = data->architecture;
    model->phaseSafe = configIsPhaseSafe(data->architecture, data->config);
    model->a2 = configIsA2(json);
    const double rate = probe->GetExpectedSampleRate();
    // -1 means "unknown" in old files; the engine feeds the model its own rate regardless, so
    // this is informational.
    model->sampleRate = rate > 0.0 ? rate : kBaseSampleRate;
    model->prewarmSamples = std::max(0, probe->GetPrewarmSamples());
    if (probe->HasInputLevel())
      model->inputLevelDbu = probe->GetInputLevel();
    if (probe->HasOutputLevel())
      model->outputLevelDbu = probe->GetOutputLevel();
    if (probe->HasLoudness())
      model->loudnessDb = probe->GetLoudness();
    model->data = std::move(data);
    return model;
  } catch (const std::exception& e) {
    error = juce::String("Not a valid NAM model: ") + e.what();
    return nullptr;
  } catch (...) {
    error = "Not a valid NAM model";
    return nullptr;
  }
}

std::shared_ptr<const Model> loadModel(const juce::File& file, juce::String& error) {
  juce::MemoryBlock block;
  if (!file.existsAsFile() || !file.loadFileAsData(block)) {
    error = "The model file could not be read";
    return nullptr;
  }
  const auto* begin = static_cast<const std::uint8_t*>(block.getData());
  return loadModel(std::vector<std::uint8_t>(begin, begin + block.getSize()), error);
}

}  // namespace namloader
