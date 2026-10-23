// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "nam_loader/Runtime.h"

#include <cmath>
#include <utility>

#include "nam_loader/SampleRate.h"

namespace namloader {
namespace {

// The post-model gain ramps over 50 ms.
constexpr double kGainRampSeconds = 0.05;

}  // namespace

Runtime::Runtime(std::shared_ptr<const Model> m, const RuntimeSpec& s)
    : model(std::move(m)), spec(s) {}

Runtime::~Runtime() = default;

std::unique_ptr<Runtime> Runtime::create(std::shared_ptr<const Model> model,
                                         const RuntimeSpec& spec) {
  try {
    if ((spec.channels != 1 && spec.channels != 2) || spec.maxBlockSize < 1 ||
        spec.oversampleFactor < 1 || model == nullptr || model->data == nullptr)
      return nullptr;

    auto runtime = std::unique_ptr<Runtime>(new Runtime(std::move(model), spec));
    // One voice per declared channel (never collapses stereo); the engine picks the phase count
    // from the oversampling factor for phase-safe architectures.
    runtime->engine = Engine::create(*runtime->model, spec.oversampleFactor, spec.channels,
                                     spec.maxBlockSize, spec.slimmableSize);
    if (runtime->engine == nullptr)
      return nullptr;

    runtime->gainSmoother.reset(kBaseSampleRate * spec.oversampleFactor, kGainRampSeconds);
    runtime->gainSmoother.setCurrentAndTargetValue(1.0f);  // snapped on the first block
    // Sized here so process() never allocates.
    runtime->dry.setSize(spec.channels, spec.maxBlockSize);
    return runtime;
  } catch (...) {
    return nullptr;
  }
}

void Runtime::process(float* const* channels,
                      int numSamples,
                      const ProcessContext& context) noexcept {
  if (numSamples <= 0)
    return;
  jassert(numSamples <= spec.maxBlockSize);

  // Failed: the input already is the dry signal, leave it untouched.
  if (failed)
    return;

  for (int ch = 0; ch < spec.channels; ++ch)
    juce::FloatVectorOperations::copy(dry.getWritePointer(ch), channels[ch], numSamples);

  try {
    processModel(channels, numSamples, context);
  } catch (...) {
    // Real-time failure path: nothing is logged or allocated here. The flag lets the caller see
    // the failure and rebuild without the model; until then, and from now on for this runtime,
    // the signal passes dry.
    failed = true;
    for (int ch = 0; ch < spec.channels; ++ch)
      juce::FloatVectorOperations::copy(channels[ch], dry.getReadPointer(ch), numSamples);
  }
}

int Runtime::getPrewarmSamples() const noexcept {
  // The model's receptive field is counted at its own rate in the stream each instance sees; a
  // phase instance sees every Nth oversampled sample.
  return engine != nullptr ? model->prewarmSamples * engine->getPhaseCount() : 0;
}

void Runtime::processModel(float* const* channels, int numSamples, const ProcessContext& context) {
  const int numChannels = spec.channels;
  juce::AudioBuffer<float> buffer(channels, numChannels, numSamples);  // a view, no allocation

  // Calibration: drive the model at the level of its capture rig.
  float calibrationGain = 1.0f;
  if (context.calibrateInput && model->inputLevelDbu.has_value()) {
    const double adjustmentDb = context.inputCalibrationDbu - *model->inputLevelDbu;
    calibrationGain = juce::Decibels::decibelsToGain(static_cast<float>(adjustmentDb));
  }
  if (calibrationGain != 1.0f)
    for (int ch = 0; ch < numChannels; ++ch)
      buffer.applyGain(ch, 0, numSamples, calibrationGain);

  // One voice per input channel: with two voices, each channel runs through its own set of phase
  // instances; with one, channel 0 is the whole signal.
  engine->process(buffer, context.pool, /*dualMono=*/true);

  // Post-model gain: see Runtime::process for why hand-off and normalization are exclusive.
  const float targetLoudnessDb = context.targetLoudnessDb;
  float blockGain = 1.0f;
  bool handedOff = false;
  if (context.calibrateInput && model->outputLevelDbu.has_value()) {
    const float outputLevel = static_cast<float>(*model->outputLevelDbu);
    if (std::isfinite(outputLevel) && outputLevel >= -60.0f && outputLevel <= 60.0f &&
        context.calibratedHandOff) {
      blockGain = juce::Decibels::decibelsToGain(outputLevel - context.inputCalibrationDbu);
      handedOff = true;
    }
  }
  if (!handedOff && context.normalize) {
    float modelLoudnessDb = targetLoudnessDb;
    if (model->loudnessDb.has_value())
      modelLoudnessDb = static_cast<float>(*model->loudnessDb);
    if (!std::isfinite(modelLoudnessDb) || modelLoudnessDb < -100.0f || modelLoudnessDb > 0.0f)
      modelLoudnessDb = targetLoudnessDb;
    const float gainAdjustmentDb = juce::jlimit(-12.0f, 12.0f, targetLoudnessDb - modelLoudnessDb);
    blockGain = juce::Decibels::decibelsToGain(gainAdjustmentDb);
  }
  // A new runtime starts at the gain it will hold (the first block after a build), so a
  // replacement runtime crossfaded in next to a running one does not glide in from unity.
  if (!gainPrimed) {
    gainSmoother.setCurrentAndTargetValue(blockGain);
    gainPrimed = true;
  }
  gainSmoother.setTargetValue(blockGain);
  for (int i = 0; i < numSamples; ++i) {
    const float g = gainSmoother.getNextValue();
    for (int ch = 0; ch < numChannels; ++ch)
      channels[ch][i] *= g;
  }
}

}  // namespace namloader
