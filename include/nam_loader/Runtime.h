// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <memory>

#include "nam_loader/Engine.h"
#include "nam_loader/Model.h"

namespace namloader {

class WorkerPool;

// Fixed facts a Runtime is built for. A change of any of them means a new Runtime.
struct RuntimeSpec {
  // 1, 2, 4 or 8. The Runtime works at kBaseSampleRate times this (see SampleRate.h).
  int oversampleFactor = 1;
  // Largest frame count per process() call, at the oversampled rate.
  int maxBlockSize = 0;
  // 1, or 2 for two independent voices (dual mono): each channel runs through its own set of
  // model instances, so a stereo input is never collapsed to mono.
  int channels = 1;
  // 0.0 = lite, 1.0 = full; see Engine::setSlimmableSize.
  double slimmableSize = 1.0;
};

// Per-call values the audio thread passes down.
struct ProcessContext {
  // Drive the model at the level of its capture rig: gain of inputCalibrationDbu minus the
  // model's input level. A model without an input level is left at unity.
  bool calibrateInput = false;
  float inputCalibrationDbu = 12.0f;
  // Loudness normalization toward targetLoudnessDb, from the model's loudness metadata.
  bool normalize = true;
  float targetLoudnessDb = -18.0f;
  // Another (enabled) model follows this one: with calibrateInput, hand the calibrated level on
  // instead of normalizing (see Runtime::process).
  bool calibratedHandOff = false;
  // Worker pool for forking the model's phases and voices; null runs them serially. Serial and
  // parallel schedules produce bit-identical output.
  WorkerPool* pool = nullptr;
};

// One model, ready to run: an Engine built from a Model (one voice per input channel, phases per
// the oversampling factor) plus the input calibration and post-model gain (calibrated hand-off
// or loudness normalization) around it. It runs at the oversampled rate, in place.
//
// Anything beyond that (metering, dry/wet mixing, widening to the output format, deciding which
// model follows which, telemetry) is the caller's job; what it knows about its neighbours
// arrives as flags in ProcessContext.
class Runtime {
public:
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  // Off the audio thread: instantiates and prewarms every model instance. Null when the model
  // is null or has no data, the spec is unusable (channels not 1 or 2, no block size, factor
  // below 1), or the build throws.
  static std::unique_ptr<Runtime> create(std::shared_ptr<const Model> model,
                                         const RuntimeSpec& spec);

  // Audio thread. Processes spec.channels channels of `numSamples` frames in place
  // (numSamples <= spec.maxBlockSize). An engine exception never escapes: this call and every
  // later one pass the dry input through, and hasFailed() turns true.
  //
  // After the model, the gain is either a calibrated hand-off or loudness normalization, never
  // both, because their goals contradict (reproduce the capture rig's true level vs. make every
  // capture equally loud). Hand-off applies when calibrateInput and calibratedHandOff are set
  // and the model carries a sane output level: the model's output dBu minus the caller's
  // calibration dBu converts the output back into the caller's analog reference frame, and the
  // next model's own input calibration converts from that frame into its own, so the caller's
  // setting cancels and the hand-off carries exactly the level of plugging device A into
  // device B. The last model in a series stays on normalization: calibrated output at the end
  // would swing the overall volume with each capture's metadata. The hand-off is not clamped (it
  // is a physical level difference, not a guess); only a sanity check on the metadata falls
  // back to normalization.
  void process(float* const* channels, int numSamples, const ProcessContext& context) noexcept;

  // The model's receptive field in samples at the oversampled rate (the model's prewarm count
  // times the phase count): what a caller pre-rolls before crossfading this runtime in.
  int getPrewarmSamples() const noexcept;

  bool hasFailed() const noexcept { return failed; }

  const Engine& getEngine() const noexcept { return *engine; }
  const Model& getModel() const noexcept { return *model; }

private:
  Runtime(std::shared_ptr<const Model> model, const RuntimeSpec& spec);

  void processModel(float* const* channels, int numSamples, const ProcessContext& context);

  std::shared_ptr<const Model> model;
  RuntimeSpec spec;

  bool failed = false;  // an engine threw: pass dry from now on
  bool gainPrimed = false;

  std::unique_ptr<Engine> engine;
  juce::LinearSmoothedValue<float> gainSmoother;
  // This call's dry input, restored when the engine throws mid-block.
  juce::AudioBuffer<float> dry;
};

}  // namespace namloader
