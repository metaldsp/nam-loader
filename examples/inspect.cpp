// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

// nam_loader_inspect <model.nam> [oversampleFactor] [hostSampleRate]
//
// Loads a NAM model, prints its metadata, then runs one second of a 440 Hz
// stereo sine through RateBoundary -> Oversampler -> Runtime at the host rate
// and reports the output peak.

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "nam_loader/Model.h"
#include "nam_loader/Oversampler.h"
#include "nam_loader/RateBoundary.h"
#include "nam_loader/Runtime.h"
#include "nam_loader/SampleRate.h"

namespace {

constexpr int kHostBlock = 512;
constexpr int kChannels = 2;

const char* yesNo(bool value) {
  return value ? "yes" : "no";
}

void printOptional(const char* label, const std::optional<double>& value, const char* unit) {
  if (value.has_value())
    std::printf("%-16s %.2f %s\n", label, *value, unit);
  else
    std::printf("%-16s absent\n", label);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s <model.nam> [oversampleFactor: 1|2|4|8] [hostSampleRate]\n",
                 argv[0]);
    return 2;
  }

  int factor = 1;
  if (argc >= 3) {
    factor = std::atoi(argv[2]);
    if (factor != 1 && factor != 2 && factor != 4 && factor != 8) {
      std::fprintf(stderr, "oversampleFactor must be 1, 2, 4 or 8\n");
      return 2;
    }
  }

  double hostRate = 44100.0;
  if (argc == 4) {
    hostRate = std::atof(argv[3]);
    if (hostRate < 8000.0 || hostRate > 384000.0) {
      std::fprintf(stderr, "hostSampleRate must be between 8000 and 384000\n");
      return 2;
    }
  }

  juce::String error;
  const auto model =
      namloader::loadModel(juce::File::getCurrentWorkingDirectory().getChildFile(argv[1]), error);
  if (model == nullptr) {
    std::fprintf(stderr, "load failed: %s\n", error.toRawUTF8());
    return 1;
  }

  std::printf("file:            %s\n", argv[1]);
  std::printf("architecture:    %s\n", model->architecture.c_str());
  std::printf("a2:              %s\n", yesNo(model->a2));
  std::printf("phase-safe:      %s\n", yesNo(model->phaseSafe));
  std::printf("sample rate:     %.1f Hz\n", model->sampleRate);
  std::printf("prewarm:         %d samples (model rate)\n", model->prewarmSamples);
  printOptional("input level:", model->inputLevelDbu, "dBu");
  printOptional("output level:", model->outputLevelDbu, "dBu");
  printOptional("loudness:", model->loudnessDb, "dB");

  namloader::RateBoundary boundary;
  boundary.prepare(hostRate, kHostBlock);
  const int maxBaseFrames = boundary.getMaxBaseBlockSize();

  namloader::RuntimeSpec spec;
  spec.oversampleFactor = factor;
  spec.maxBlockSize = maxBaseFrames * factor;
  spec.channels = kChannels;
  auto runtime = namloader::Runtime::create(model, spec);
  if (runtime == nullptr) {
    std::fprintf(stderr, "could not build a Runtime for this model\n");
    return 1;
  }

  namloader::Oversampler oversampler;
  oversampler.prepare(factor, maxBaseFrames);

  const int totalFrames = static_cast<int>(hostRate);  // one second
  std::vector<float> left(static_cast<size_t>(totalFrames));
  for (int i = 0; i < totalFrames; ++i)
    left[static_cast<size_t>(i)] =
        0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 440.0 * i / hostRate));
  std::vector<float> right = left;

  const namloader::ProcessContext context;
  // Built once, before processing: converting a lambda to std::function on the audio thread
  // can allocate.
  const namloader::RateBoundary::ProcessFn processAtBaseRate = [&](float** base, float** baseOut,
                                                                   int baseFrames) {
    // The runtime works in place, and a x1 oversampler passes its buffers straight through,
    // so the result only reaches the boundary's output if it is processed there.
    for (int c = 0; c < kChannels; ++c)
      if (base[c] != baseOut[c])
        std::copy(base[c], base[c] + baseFrames, baseOut[c]);
    oversampler.process(baseOut, baseOut, baseFrames,
                        [&](float** oversampled, float** /*out*/, int oversampledFrames) {
                          runtime->process(oversampled, oversampledFrames, context);
                        });
  };

  float peak = 0.0f;
  for (int offset = 0; offset < totalFrames; offset += kHostBlock) {
    const int frames = std::min(kHostBlock, totalFrames - offset);
    float* channels[kChannels] = {left.data() + offset, right.data() + offset};
    boundary.process(channels, channels, frames, processAtBaseRate);
    for (int i = 0; i < frames; ++i)
      peak = std::max(peak, std::abs(channels[0][i]));
  }

  std::printf("host rate:       %.1f Hz\n", hostRate);
  std::printf("oversampling:    x%d\n", factor);
  std::printf("output peak:     %.6f\n", static_cast<double>(peak));
  std::printf("prewarm:         %d samples (oversampled rate)\n", runtime->getPrewarmSamples());
  std::printf("latency:         %d samples (host rate)\n", boundary.getLatencySamples());
  std::printf("failed:          %s\n", yesNo(runtime->hasFailed()));
  return runtime->hasFailed() ? 1 : 0;
}
