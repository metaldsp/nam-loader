<!--
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
SPDX-License-Identifier: MIT
-->

# NAM oversampling runs phase-interleaved copies of the model instead of patching the core

To oversample a WaveNet or ConvNet NAM capture by N, the `Engine` holds N independent instances of the unscaled model and feeds each every Nth sample, rather than scaling the model's convolution dilations by N inside NeuralAmpModelerCore. The two are mathematically identical (`Oversampler.PhaseInterleavedDilatedConvIsExact` pins this sample-exactly), the receptive field stays constant in seconds so the capture sounds the same with less aliasing, and NeuralAmpModelerCore stays unpatched.

## Consequences

- An engine's slots are `voices x phases` model instances. The voice count follows the channel count the engine is created for: a mono engine has one voice, a stereo engine two, each with a full phase set fed from its own channel.
- Slots are fully independent, so given a `WorkerPool` they fork as one flat loop. Forked output is bit-identical to serial (`EngineTest.PhaseForkMatchesSerialBitExact`), so multi-core is pure scheduling.
- LSTM models update state on consecutive samples and cannot be phase-split. They run as a single time-scaled instance at the full oversampled rate, a defensive path in practice since A2 WaveNets are always phase-safe.
- This sits inside the single resampling boundary of [ADR 0001](0001-one-resampling-boundary.md): the phases run at the oversampled rate, and the rest of the graph never sees them.

Details live in [docs/oversampling.md](../oversampling.md) and [docs/multicore.md](../multicore.md).
