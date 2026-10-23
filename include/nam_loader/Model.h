// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nam_loader/SampleRate.h"

namespace nam {
struct dspData;
}

namespace namloader {

// One NAM model, decoded and validated once off the audio thread. It is immutable afterwards, so
// it can be shared (shared_ptr<const>) by every engine that runs it. Engines build their own
// nam::DSP instances from copies of `data` (mutable DSP state is never shared), so creating
// another engine costs instantiation, not a re-parse of the model file.
struct Model {
  // Parsed config and weights (nam::get_dsp(json, dspData&)); each engine instance is created
  // from a copy.
  std::shared_ptr<const nam::dspData> data;
  // e.g. "WaveNet", "SlimmableContainer", "LSTM".
  std::string architecture;
  // WaveNet, ConvNet and Linear (and containers made only of those) can be phase-interleaved
  // when oversampled; LSTM and anything unknown runs as one time-scaled instance. See Engine.h.
  bool phaseSafe = false;
  // A2 WaveNet, or a SlimmableContainer made only of A2 WaveNets: the shape NAM's fast path is
  // tuned around (48 kHz training rate, slimmable tiers).
  bool a2 = false;
  // The rate the model was trained at. Informational: the engine always feeds the model the rate
  // it was prepared for.
  double sampleRate = kBaseSampleRate;
  // Receptive field at the model rate (nam::DSP::GetPrewarmSamples). Callers that pre-roll an
  // engine before using it (for example ahead of a crossfade) cover at least this many samples.
  int prewarmSamples = 0;
  // Calibration and loudness metadata; absent when the file does not carry it.
  std::optional<double> inputLevelDbu;
  std::optional<double> outputLevelDbu;
  std::optional<double> loudnessDb;
};

// Off the audio thread. Validates and decodes; returns null with a user-facing reason in `error`
// on failure (bad JSON, not a 1-in/1-out NAM, ...). A pure function of its input.
std::shared_ptr<const Model> loadModel(const std::vector<std::uint8_t>& bytes, juce::String& error);

// Reads the file and decodes its bytes. The path is never handed to NAM: nam::get_dsp(path)
// built from a JUCE UTF-8 string mis-decodes non-ASCII characters on Windows.
std::shared_ptr<const Model> loadModel(const juce::File& file, juce::String& error);

}  // namespace namloader
