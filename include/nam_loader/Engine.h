// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <memory>
#include <vector>

#include "NAM/dsp.h"
#include "nam_loader/Model.h"
#include "nam_loader/SampleRate.h"

namespace namloader {

class WorkerPool;

// A JUCE-compatible host for nam::DSP models. It runs at the oversampled rate
// (kBaseSampleRate x oversampling factor, see SampleRate.h). Sample-rate conversion is NOT this
// class's job (the caller puts the whole processing graph behind one rate boundary and
// oversampler), so all this does is:
//  - float <-> double conversion (NAM models process NAM_SAMPLE == double),
//  - mono processing with fan-out to stereo buffers,
//  - slimmable (A2 container) tier selection,
//  - phase-interleaved oversampled processing (below).
//
// Phase-interleaved oversampling:
// A NAM model oversampled by N with its convolution dilations scaled by N is mathematically
// identical to N independent copies of the *unscaled* model, each processing every Nth sample of
// the oversampled stream at the native rate: every scaled dilation lands taps exactly N samples
// apart (within one phase), and 1x1 convolutions/activations are per-sample. So instead of
// patching dilation scaling into NeuralAmpModelerCore, the engine holds N instances of the same
// model and interleaves them. The receptive field stays constant in seconds (the model sounds the
// same) while its nonlinear harmonics land in the widened band where the oversampler's
// decimation filter removes them instead of letting them alias.
//
// Recurrent architectures (LSTM) update state on consecutive samples and can't be phase-split;
// they get a single instance that runs time-scaled at the full oversampled rate (matching
// NAM-Oversampler's behavior for such models). Model::phaseSafe says which kind a model is.
//
// Voices:
// A NAM model is mono, so by default channel 0 goes through the model and the result is fanned
// out to channel 1. With two voices the engine holds a second, independent set of phase instances
// for channel 1, so a stereo buffer is processed as two separate mono signals through the same
// model. The voice count is fixed at construction like the phase count: a dormant voice would
// resume with stale model history, and nam::DSP has no RT-safe reset, so a caller whose input
// format changes builds a new engine. process() only *uses* voice 1 when asked (`dualMono`) and
// the buffer is stereo; otherwise voice 0 fans out exactly as a single-voice engine does, so the
// two schedules never mix channels.
//
// Multi-core:
// Every (voice, phase) instance is fully independent (separate model, disjoint I/O buffers), so
// process() forks them as one flat job set across a WorkerPool. If the caller is itself running
// inside a pool job (say, one branch of a parallel graph), forking its phases is the pool's
// supported one-deep nesting. Scheduling is the only thing that changes: deinterleave and
// reinterleave stay on the calling thread and every job writes only its own buffers, so parallel
// output is bit-identical to serial. Passing no pool runs the sequential loop.
class Engine {
public:
  // Largest voice count an engine can carry (channel 0 and channel 1).
  static constexpr int kMaxVoices = 2;

  // Takes ownership of the model instances, all built from the same model config, laid out
  // voice-major: `voices` groups of `phases` instances, where `phases` is 1 or `oversampleFactor`.
  // One phase runs at the full oversampled rate (kBaseSampleRate x `oversampleFactor`); N > 1
  // phases run phase-interleaved, each at 1/Nth of it. create() picks both counts:
  // `oversampleFactor` phases for phase-safe architectures, one otherwise; one voice per input
  // channel.
  // Throws std::invalid_argument on empty/null instances, an invalid factor or voice count, or a
  // size that isn't voices x {1, factor}.
  Engine(std::vector<std::unique_ptr<nam::DSP>> instances, int oversampleFactor, int voices = 1);
  ~Engine() = default;

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = default;
  Engine& operator=(Engine&&) = default;

  // Off the audio thread. Instantiates the engine for `model` (a copy of model.data per instance,
  // since get_dsp consumes the weights it is given), applies `slimmableSize` and prepares it for
  // blocks of up to `maxBlockSize` frames at the oversampled rate (which also prewarms every
  // instance). Returns null on any failure: a null model or data, an unusable argument, or an
  // exception from NAM.
  static std::unique_ptr<Engine> create(const Model& model,
                                        int oversampleFactor,
                                        int voices,
                                        int maxBlockSize,
                                        double slimmableSize = 1.0);

  // Prepare for processing at the oversampled rate. `maxBlockSize` is the largest per-call frame
  // count (at that rate).
  void prepare(int maxBlockSize);

  // Process an oversampled-rate buffer in place. Default: channel 0 through voice 0, fanned out
  // to channel 1 if present. With `dualMono` set on a two-voice engine and a stereo buffer,
  // channel 1 goes through voice 1 instead (two independent mono signals through the same model);
  // any other combination falls back to the fan-out, so a single-voice engine or a mono buffer
  // behaves identically whatever `dualMono` says. Must be prepared first. With a pool and more
  // than one (voice, phase) job, the jobs fork across the pool's workers (bit-identical to the
  // serial loop; see the class comment); a job failure rethrows as std::runtime_error on this
  // thread once every job has completed. `pool` may be null.
  void process(juce::AudioBuffer<float>& buffer, WorkerPool* pool = nullptr, bool dualMono = false);

  // The oversampling factor this engine was built for. A caller that tracks a live factor
  // compares it: a mismatch means the phase count is wrong and the engine must be rebuilt.
  int getOversampleFactor() const { return oversampleFactor; }

  // The voice count this engine was built for (1, or 2 for a stereo input). Compared against the
  // live requirement exactly like the factor above.
  int getVoiceCount() const { return voices; }

  // Phase instances per voice (1, or the oversampling factor).
  int getPhaseCount() const { return phases; }

  // The rate each instance runs at: the oversampled rate for a single phase, the base rate for
  // interleaved phases.
  double getInstanceSampleRate() const {
    return kBaseSampleRate * oversampleFactor / static_cast<double>(phases);
  }

  // Requested slimmable size (0.0 = lite, 1.0 = full). Clamped to [0.0, 1.0]. A no-op for models
  // that aren't SlimmableModel (non-container A2 files). NAM tier mappers assign the boundary
  // value to the tier above (a two-tier container selects lite for [0, 0.5) and full for
  // [0.5, 1.0]), so the lite request must be 0.0; 0.5 would select full. Applied in prepare() and
  // immediately if already prepared. Fans out to every phase instance so all phases always run
  // the same tier.
  void setSlimmableSize(double val);

  double getSlimmableSize() const noexcept { return requestedSlimmableSize; }

private:
  // Flat index of (voice, phase) in `instances` and the I/O slot arrays.
  int slotIndex(int voice, int phase) const noexcept { return voice * phases + phase; }

  std::vector<std::unique_ptr<nam::DSP>> instances;
  int oversampleFactor = 1;
  int voices = 1;
  int phases = 1;

  // Per-(voice, phase) double-precision I/O (NAM expects double); the deinterleave IS the
  // float -> double conversion. A single instance is just the one-voice, one-phase case of the
  // same path. Indexed by slotIndex; sized in prepare(). phaseFrames is per phase only: both
  // voices split the same chunk with the same phase offset, so their counts are equal.
  std::vector<std::vector<double>> slotInputs;
  std::vector<std::vector<double>> slotOutputs;
  std::vector<int> phaseFrames;

  // Which phase the next incoming sample belongs to. Oversampled buffers are normally divisible
  // by the phase count (the oversampler guarantees it), but tracking the offset keeps every
  // instance's sample stream continuous even if a caller hands over a partial block.
  int phaseOffset = 0;

  bool isPrepared = false;
  int maxBlockSize = 0;

  double requestedSlimmableSize{1.0};
};

}  // namespace namloader
