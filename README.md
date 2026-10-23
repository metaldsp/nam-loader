<!-- SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini -->
<!-- SPDX-License-Identifier: MIT -->

# NamLoader

A JUCE 9-based C++20 library that loads Neural Amp Modeler (NAM) captures and
runs them in real time, with phase-interleaved oversampling, dual-mono voices,
optional multi-core processing and a host-rate to 48 kHz boundary. It is
designed around A2 WaveNet captures but runs any architecture the NAM core
supports. Everything lives in the `namloader` namespace.

The library is MIT licensed; see [LICENSE](LICENSE). Files adapted from
TONE3000 code carry TONE3000 and Pier Luigi Fiorini copyrights.
NeuralAmpModelerCore and nlohmann/json are MIT; Eigen and its components keep
their upstream licenses. CPM fetches dependencies with their notices; NamLoader
does not relicense them. The NAM core's MIT notice is in
[LICENSES/MIT.txt](LICENSES/MIT.txt).

## Origin

The code was extracted from Ampira's audio engine and its tests, then adapted
into a standalone library with its own API and vocabulary. Ampira's
publication, crossfade and retirement logic, IR loading and model fetching are
not part of it.

## Components

Public headers are in `include/nam_loader/` (include them as
`#include "nam_loader/Runtime.h"`). They do not include AudioDSPTools or
nlohmann/json.

| Component | Header | Role |
| --- | --- | --- |
| `kBaseSampleRate` | `SampleRate.h` | The 48 kHz rate models run at; the architecture notes live here. |
| `Model` / `loadModel` | `Model.h` | Decode and validate a `.nam` once; immutable, shared metadata (architecture, `a2`, `phaseSafe`, sample rate, prewarm, levels, loudness). |
| `Engine` | `Engine.h` | Phase-interleaved, multi-voice NAM host with optional multi-core dispatch. |
| `Runtime` | `Runtime.h` | An `Engine` plus gain staging: input calibration, loudness normalization, 50 ms smoothing, failure passthrough. |
| `RateBoundary` | `RateBoundary.h` | Lanczos resampling between the host rate and 48 kHz (skipped at 48 kHz). |
| `Oversampler` | `Oversampler.h` | Minimum-phase, zero-latency x2/x4/x8 oversampler; `processAtBaseRate` for linear work such as convolution. |
| `WorkerPool` | `WorkerPool.h` | Real-time worker pool behind multi-core `Engine::process`. |

### Signal flow

```
host audio (any rate)
   |
   v
RateBoundary          host rate -> 48 kHz (skipped at 48 kHz)
   |
   v  base rate: 48 kHz
Oversampler x N       N = 1, 2, 4 or 8 (skipped when N = 1)
   |
   v  oversampled rate: 48 kHz * N
Runtime / Engine      models run here
   |
   v
Oversampler x N       back down
   |
   v
RateBoundary          48 kHz -> host rate
   |
   v
host audio
```

Use one boundary for the whole processing graph. Its latency is constant for a
given host rate and is reported in host samples. The oversampler adds no
latency. Linear processing that does not need the oversampled rate can run at
48 kHz through `Oversampler::processAtBaseRate`.

The boundary can hand the base-rate function slightly more frames than
`hostFrames * 48000 / hostRate`. Size anything inside it from
`RateBoundary::getMaxBaseBlockSize()`, and set `RuntimeSpec::maxBlockSize` to
that times the oversampling factor.

### Threading and real-time rules

- Off the audio thread (they allocate or block): `loadModel`,
  `Runtime::create`, `Engine::create` and `prepare`,
  `RateBoundary::prepare`, `warmTables` and `measureLatency`,
  `Oversampler::prepare`, `WorkerPool::start` and `stop`, and destruction of
  any of these.
- RT-safe on the audio thread (no allocation, locks or I/O): `Runtime::process`,
  `Engine::process`, `Oversampler::process`, `RateBoundary::process` and
  `WorkerPool::forkJoin`. `RateBoundary::process` takes a `std::function`: build
  it once, off the audio thread, and pass the same object every call, because
  converting a lambda at the call site can allocate.
- `Model` is immutable and shared. Each `Engine` instance gets its own copy of
  the weights, because the NAM core consumes them.
- An engine exception never escapes `Runtime::process`: that call and every
  later one pass the dry input through, and `hasFailed()` turns true. Bypass,
  reporting and retirement are the host's job.
- The host owns publication, crossfades and retirement of runtimes. Build on a
  loader thread, hand over at a callback boundary, and destroy the old runtime
  off the audio thread.
- Models must be trusted, valid NAM exports; this is not a hardened parser.

### Usage

This mirrors `examples/inspect.cpp`, which sizes the buffers as described above
and runs a second of audio through the chain.

```cpp
#include "nam_loader/Model.h"
#include "nam_loader/Oversampler.h"
#include "nam_loader/RateBoundary.h"
#include "nam_loader/Runtime.h"

// Off the audio thread.
juce::String error;
auto model = namloader::loadModel(juce::File("amp.nam"), error);
if (model == nullptr)
  return;  // `error` holds the reason

namloader::RateBoundary boundary;
boundary.prepare(hostRate, maxHostFrames);
const int maxBaseFrames = boundary.getMaxBaseBlockSize();

namloader::RuntimeSpec spec;
spec.oversampleFactor = 4;
spec.maxBlockSize = maxBaseFrames * spec.oversampleFactor;  // oversampled rate
spec.channels = 2;
auto runtime = namloader::Runtime::create(model, spec);

namloader::Oversampler oversampler;
oversampler.prepare(spec.oversampleFactor, maxBaseFrames);

namloader::ProcessContext context;
const namloader::RateBoundary::ProcessFn processAtBaseRate = [&](float** base, float** out,
                                                                 int baseFrames) {
  // Runtime works in place; make `out` hold the signal first (see examples/inspect.cpp).
  for (int c = 0; c < 2; ++c)
    if (base[c] != out[c])
      std::copy(base[c], base[c] + baseFrames, out[c]);
  oversampler.process(out, out, baseFrames, [&](float** os, float**, int osFrames) {
    runtime->process(os, osFrames, context);
  });
};

// Audio thread, per host block (in place).
boundary.process(channels, channels, frames, processAtBaseRate);
```

## Further reading

- [CONTEXT.md](CONTEXT.md): the domain glossary (base rate, phase, voice,
  calibrated hand-off, ...).
- [docs/oversampling.md](docs/oversampling.md): the half-band filter design,
  phase-interleaved NAM oversampling and the frame-count contract.
- [docs/multicore.md](docs/multicore.md): the realtime worker pool's design
  rules, job protocol, wakeups and failure containment.
- [docs/calibration.md](docs/calibration.md): input calibration, the
  calibrated hand-off and loudness normalization, mapped onto `ProcessContext`.
- [docs/hosting.md](docs/hosting.md): how a host builds, pre-rolls, crossfades
  and retires runtimes (the pattern Ampira uses).
- [docs/adr/](docs/adr/): the decisions behind the single resampling boundary
  and phase-interleaved oversampling.

## Build, consume and test

Requirements: CMake 3.24+, a C++20 compiler, Git, and network access on the
first configure (github.com, plus gitlab.com for the Eigen submodule that both
NeuralAmpModelerCore and AudioDSPTools pin). CPM 0.40.2 is bootstrapped with a SHA-256 check unless the
parent already provides CPM. Pinned dependencies (`cmake/Dependencies.cmake`):

- NeuralAmpModelerCore `1f42f88` (sources compiled into `nam_loader_core`; Eigen
  from its submodule only, the core's other submodules are not cloned).
- nlohmann/json 3.12.0 (not a NAM submodule, fetched separately).
- AudioDSPTools `0827c6c` (header-only `ResamplingContainer`, used only by
  `src/RateBoundary.cpp`).
- JUCE 9.0.3, fetched only if the parent has not already added JUCE
  (`if(NOT COMMAND juce_add_module)`). The library needs only
  `juce_audio_basics`.
- GoogleTest 1.15.2, tests only.

Targets:

- `nam_loader_core` (STATIC): the NAM core sources. Linked whole-archive because
  architectures register themselves in otherwise unreferenced objects. Hidden
  visibility; `-fno-gnu-unique` on GCC so registries stay private to each plugin
  binary loaded into one host.
- `nam_loader` / `NamLoader::NamLoader` (INTERFACE): its `.cpp` files compile
  into the consumer target together with the consumer's JUCE modules. A STATIC
  wrapper would compile JUCE twice (duplicate symbols, ODR violations).

Consume it by adding JUCE first, then the library:

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyPlugin LANGUAGES CXX)
add_subdirectory(JUCE)            # or your existing JUCE setup
add_subdirectory(nam-loader)      # or CPMAddPackage(...)
juce_add_plugin(MyPlugin ...)
target_link_libraries(MyPlugin PRIVATE NamLoader::NamLoader)
```

Link the CMake target rather than archives: it carries the core, the include
paths and the linker handling. This is a source-subdirectory package; there is
no `find_package()` export.

Options (off by default): `NAM_LOADER_BUILD_TESTS` and
`NAM_LOADER_BUILD_EXAMPLE` (`examples/inspect.cpp`, built as
`nam_loader_inspect <model.nam> [oversampleFactor] [hostSampleRate]`).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNAM_LOADER_BUILD_TESTS=ON -DNAM_LOADER_BUILD_EXAMPLE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Tests

One GoogleTest app, `NamLoaderTests`, with a suite per component (`WorkerPool`,
`Oversampler`, `Model`, `Engine`, `RateBoundary`, `Runtime`). Fixtures:

- `tests/files`: real A2 captures that TONE3000's open-source plugin shipped
  as test assets (by way of Ampira). Two credit community creators in their
  metadata, and whether TONE3000 could redistribute them is not known, so they
  are deliberately not covered in `REUSE.toml` and `reuse lint` lists them.
- The NAM core's `example_models`, found through the fetched dependency, for
  other architectures.

### JUCE patch

Ampira, the original host, applies a patch to JUCE's `juce_Threads_mac.mm`
(marker `AMPIRA_RT_START_STALE_HANDLE`, in Ampira's root `CMakeLists.txt`). When a realtime upgrade fails on macOS, the
pthread exits before `juce_threadEntryPoint`, so `closeThreadHandle()` never
runs; the patch clears the stale handle (`threadHandle = nullptr; threadId = {}`)
when `Thread::startThread` fails to start, so the Thread can be restarted or
destroyed without `stopThread()` stalling. `WorkerPool` depends on this. A
standalone NamLoader build uses stock JUCE 9.0.3 and does not apply it;
consumers that care on macOS should apply the same patch to their JUCE.

## Licensing hooks

The shared checker is `.agents/hooks/check-reuse.py`. Claude's configuration is
in `.claude/settings.json`; Codex's is in `.codex/hooks.json`. Start the agent
from the NamLoader folder to use its project configuration. These files travel
with the library; they do not change Ampira's own hook configuration.

Install Python 3.9+, Git and the REUSE CLI (`pipx install reuse`), with `python3`,
`git` and `reuse` on the agent's PATH. Review and trust the new Codex hooks using
`/hooks`; untrusted hook definitions are skipped. See the official
[Codex hook documentation](https://learn.chatgpt.com/docs/hooks) and
[Claude hook reference](https://code.claude.com/docs/en/hooks) for loading and
trust behavior.

After tool calls and before the agent finishes, the checker runs `reuse
lint-file` on staged, unstaged and untracked files within NamLoader. Git-ignored
outputs, deleted ordinary files, submodules and symlinks are skipped. Unchanged
licensing gaps elsewhere do not affect this check. Editing licensing metadata
or deleting a license triggers a full library lint because those changes can
invalidate unchanged files. Outside a Git worktree, it runs a full lint.

Both configurations report failures back to the agent without undoing edits or
adding licensing claims. Claude also checks failed tool calls; Codex includes
failed shell commands in its post-tool event. The Stop hook requests a repair
pass once, then reports unresolved problems without forcing an endless loop.
This is agent feedback, not a Git commit hook or a replacement for CI.

From the NamLoader folder, run the same check manually or test the integration:

```sh
python3 .agents/hooks/check-reuse.py --check
python3 -m unittest discover -s tests -p 'test_reuse_hook.py' -v
```

The tests exercise both configured commands with hook payloads, including nested
working directories, using temporary Git repositories and the real REUSE CLI.
JSON configuration files receive their SPDX metadata from `REUSE.toml`.
