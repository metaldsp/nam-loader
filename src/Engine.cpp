// SPDX-FileCopyrightText: 2026 TONE3000
// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "nam_loader/Engine.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>

#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "nam_loader/WorkerPool.h"

namespace namloader {

Engine::Engine(std::vector<std::unique_ptr<nam::DSP>> modelInstances, int factor, int voiceCount)
    : instances(std::move(modelInstances)), oversampleFactor(factor), voices(voiceCount) {
  if (instances.empty())
    throw std::invalid_argument("Engine needs at least one model instance");
  for (const auto& instance : instances) {
    if (instance == nullptr)
      throw std::invalid_argument("NAM model cannot be null");
  }
  if (factor < 1)
    throw std::invalid_argument("Engine oversampling factor must be >= 1");
  if (voices < 1 || voices > kMaxVoices)
    throw std::invalid_argument("Engine voice count must be 1 or 2");
  const auto count = static_cast<int>(instances.size());
  if (count % voices != 0)
    throw std::invalid_argument("Engine instance count must be a multiple of the voice count");
  phases = count / voices;
  if (phases != 1 && phases != factor)
    throw std::invalid_argument("Engine phases per voice must be 1 or the oversampling factor");
  // The flat voice x phase fork publishes one job per instance.
  if (count > WorkerPool::kMaxJobs)
    throw std::invalid_argument("Engine instance count exceeds the worker pool's job limit");
}

std::unique_ptr<Engine> Engine::create(const Model& model,
                                       int oversampleFactor,
                                       int voices,
                                       int maxBlockSize,
                                       double slimmableSize) {
  try {
    if (model.data == nullptr || oversampleFactor < 1 || voices < 1 || voices > kMaxVoices ||
        maxBlockSize < 1)
      return nullptr;

    // Phases follow the oversampling factor for phase-safe architectures (see Engine.h);
    // everything else runs as one time-scaled instance.
    const bool phaseSafe = oversampleFactor > 1 && model.phaseSafe;
    const int phases = phaseSafe ? oversampleFactor : 1;

    std::vector<std::unique_ptr<nam::DSP>> instances;
    instances.reserve(static_cast<size_t>(voices * phases));
    for (int i = 0; i < voices * phases; ++i) {
      // get_dsp consumes the weights it is given: every instance gets a copy.
      nam::dspData copy = *model.data;
      std::unique_ptr<nam::DSP> dsp = nam::get_dsp(copy);
      if (dsp == nullptr)
        return nullptr;
      instances.push_back(std::move(dsp));
    }

    auto engine = std::make_unique<Engine>(std::move(instances), oversampleFactor, voices);
    engine->setSlimmableSize(slimmableSize);
    engine->prepare(maxBlockSize);  // also prewarms every instance
    return engine;
  } catch (...) {
    return nullptr;
  }
}

void Engine::prepare(int newMaxBlockSize) {
  maxBlockSize = newMaxBlockSize;

  const auto slotCount = instances.size();
  // +1: a carried phase offset can hand one phase an extra frame when a partial slice isn't
  // divisible by the phase count.
  const size_t perPhaseCapacity =
      static_cast<size_t>(maxBlockSize) / static_cast<size_t>(phases) + 1;
  slotInputs.assign(slotCount, std::vector<double>(perPhaseCapacity));
  slotOutputs.assign(slotCount, std::vector<double>(perPhaseCapacity));
  phaseFrames.assign(static_cast<size_t>(phases), 0);
  phaseOffset = 0;

  for (auto& instance : instances) {
    instance->ResetAndPrewarm(getInstanceSampleRate(), static_cast<int>(perPhaseCapacity));

    // SlimmableContainer / SlimmableWavenet: after max buffer size is set on the root DSP,
    // SetSlimmableSize re-selects the active tier and ResetAndPrewarms it (see NAM render tools
    // and NeuralAmpModelerCore tests). Without this, container models can stay on a stale or
    // uninitialized sub-path.
    if (auto* slimmable = dynamic_cast<nam::SlimmableModel*>(instance.get()))
      slimmable->SetSlimmableSize(requestedSlimmableSize);
  }

  isPrepared = true;
}

void Engine::setSlimmableSize(double val) {
  requestedSlimmableSize = juce::jlimit(0.0, 1.0, val);
  if (!isPrepared)
    return;
  for (auto& instance : instances) {
    if (auto* slimmable = dynamic_cast<nam::SlimmableModel*>(instance.get()))
      slimmable->SetSlimmableSize(requestedSlimmableSize);
  }
}

void Engine::process(juce::AudioBuffer<float>& buffer, WorkerPool* pool, bool dualMono) {
  if (!isPrepared || maxBlockSize < 1) {
    throw std::runtime_error("Engine must be prepared before processing");
  }

  const int numSamples = buffer.getNumSamples();
  const int numChannels = buffer.getNumChannels();

  // Which channels get their own voice this call: channel 0 always; channel 1 only on a
  // two-voice engine with a stereo buffer and the dual-mono flag. Every other combination is the
  // classic mono path (voice 0, then fan out), so a single-voice engine is unaffected by the flag
  // and a two-voice engine never processes a mirrored channel twice for nothing.
  const int activeVoices = (dualMono && voices > 1 && numChannels > 1) ? 2 : 1;
  const int jobCount = activeVoices * phases;

  // NAM models are mono: each active voice processes its own channel. Buffers larger than the
  // prepared size are run in prepared-size slices (models stream statefully, so slicing is
  // exact); throwing here would permanently disable the engine when a startup prepare races the
  // host's actual block size.
  for (int offset = 0; offset < numSamples; offset += maxBlockSize) {
    const int chunk = juce::jmin(maxBlockSize, numSamples - offset);

    // Deinterleave each active channel into its voice's per-phase streams (single phase = one
    // stream = a straight copy). This doubles as the float -> double conversion. Both voices see
    // the same phase walk, so phaseFrames is computed once from voice 0.
    std::fill(phaseFrames.begin(), phaseFrames.end(), 0);
    for (int v = 0; v < activeVoices; ++v) {
      const float* channel = buffer.getReadPointer(v) + offset;
      int phase = phaseOffset;
      int frames[WorkerPool::kMaxJobs] = {};
      for (int i = 0; i < chunk; ++i) {
        slotInputs[static_cast<size_t>(slotIndex(v, phase))][static_cast<size_t>(frames[phase]++)] =
            static_cast<double>(channel[i]);
        if (++phase == phases)
          phase = 0;
      }
      if (v == 0)
        for (int p = 0; p < phases; ++p)
          phaseFrames[static_cast<size_t>(p)] = frames[p];
    }

    if (pool != nullptr && jobCount > 1) {
      // Fork every (voice, phase) job across the pool (see the class comment): each job touches
      // only its own instance and buffers, so any schedule produces the same bits as the serial
      // loop below. Exceptions can't cross threads, so a job traps its own and the joiner
      // rethrows once all jobs are done; the caller's real-time failure handling (stop using the
      // engine, log off-thread) works exactly as for a serial throw.
      struct SlotJob {
        nam::DSP* model;
        NAM_SAMPLE* input;
        NAM_SAMPLE* output;
        int frames;
        std::atomic<bool>* failed;
      };
      std::atomic<bool> jobFailed{false};
      jassert(jobCount <= WorkerPool::kMaxJobs);  // enforced by the constructor
      SlotJob jobs[WorkerPool::kMaxJobs];
      void* ctxs[WorkerPool::kMaxJobs];
      for (int v = 0; v < activeVoices; ++v) {
        for (int p = 0; p < phases; ++p) {
          const int job = slotIndex(v, p);
          const auto slot = static_cast<size_t>(job);
          jobs[job] = {instances[slot].get(), slotInputs[slot].data(), slotOutputs[slot].data(),
                       phaseFrames[static_cast<size_t>(p)], &jobFailed};
          ctxs[job] = &jobs[job];
        }
      }
      pool->forkJoin(
          [](void* ctx) {
            auto& j = *static_cast<SlotJob*>(ctx);
            if (j.frames <= 0)
              return;
            try {
              NAM_SAMPLE* inputPtrs[] = {j.input};
              NAM_SAMPLE* outputPtrs[] = {j.output};
              j.model->process(inputPtrs, outputPtrs, j.frames);
            } catch (...) {
              j.failed->store(true, std::memory_order_release);
            }
          },
          ctxs, jobCount);
      if (jobFailed.load(std::memory_order_acquire))
        throw std::runtime_error("NAM phase processing failed");
    } else {
      for (int v = 0; v < activeVoices; ++v) {
        for (int p = 0; p < phases; ++p) {
          const int frames = phaseFrames[static_cast<size_t>(p)];
          if (frames <= 0)
            continue;
          const auto slot = static_cast<size_t>(slotIndex(v, p));
          NAM_SAMPLE* inputPtrs[] = {slotInputs[slot].data()};
          NAM_SAMPLE* outputPtrs[] = {slotOutputs[slot].data()};
          instances[slot]->process(inputPtrs, outputPtrs, frames);
        }
      }
    }

    // Reinterleave each voice back onto its channel (and convert to float).
    for (int v = 0; v < activeVoices; ++v) {
      float* channel = buffer.getWritePointer(v) + offset;
      int phase = phaseOffset;
      int frames[WorkerPool::kMaxJobs] = {};
      for (int i = 0; i < chunk; ++i) {
        channel[i] = static_cast<float>(slotOutputs[static_cast<size_t>(slotIndex(v, phase))]
                                                   [static_cast<size_t>(frames[phase]++)]);
        if (++phase == phases)
          phase = 0;
      }
    }

    phaseOffset = (phaseOffset + chunk) % phases;
  }

  // Mono path: fan voice 0's result out to the other channels. With two voices channel 1 already
  // carries voice 1's output and is left alone.
  for (int ch = activeVoices; ch < numChannels; ++ch) {
    buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
  }
}

}  // namespace namloader
