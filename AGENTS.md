<!-- SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini -->
<!-- SPDX-License-Identifier: MIT -->

# NamLoader agent instructions

NamLoader is a MIT-licensed JUCE 9 / C++20 library (namespace `namloader`) that
loads and runs NAM captures. Read [README.md](README.md) first for the
components, signal flow and threading rules, and use the terms in
[CONTEXT.md](CONTEXT.md). Before changing oversampling, the worker pool, gain
staging or runtime ownership, read the matching file in [docs/](docs/), and keep
those docs (and the test names they cite) in step with the code. The library was
extracted from Ampira's audio engine but is now standalone; never modify the
Ampira repository from here.

## Commands

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNAM_LOADER_BUILD_TESTS=ON -DNAM_LOADER_BUILD_EXAMPLE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
clang-format -i <changed files>
reuse lint
```

Inspect a capture:
`build/nam_loader_inspect /abs/path/model.nam [oversampleFactor] [hostSampleRate]`.

## Public API

- Public headers are in `include/nam_loader`; sources in `src/`.
- Keep AudioDSPTools and `json.hpp` types out of public headers (they belong to
  `src/RateBoundary.cpp` and `src/Model.cpp`). `NAM/dsp.h` is allowed in `Engine.h`.
- Everything is in namespace `namloader`. Use library terms (host, base rate,
  oversampled rate), not Ampira vocabulary.
- Update the README for any API change.

## Dependency and linker constraints

- Fetch with `CPMAddPackage` at pinned commits in `cmake/Dependencies.cmake`.
  Keep the CPM bootstrap hash check.
- Never edit fetched sources. Eigen comes from NAM's submodule only; upstream
  tools stay excluded (`DOWNLOAD_ONLY`).
- Add JUCE only if `juce_add_module` does not already exist.
- `nam_loader_core` is whole-archive linked (architectures self-register), with
  hidden visibility and `-fno-gnu-unique` on GCC. Keep all three.
- `NamLoader::NamLoader` stays INTERFACE so JUCE compiles once in the consumer.
- After changing pins or acquisition, verify in a fresh build directory.

## Audio invariants

- Load, create, construct, prepare, start/stop and destroy off the audio
  thread. `Runtime::process`, `Engine::process`, `Oversampler::process`,
  `RateBoundary::process` and `WorkerPool::forkJoin` must not allocate, lock,
  do I/O or spawn threads.
- `RateBoundary::warmTables` runs off-thread before any boundary processes;
  `measureLatency` allocates.
- `Model` is immutable; each engine instance copies the weights.
- `Runtime::process` never throws: an engine failure passes the dry input
  through from then on.
- The host owns publication, crossfades and retirement; do not add them here.
- Keep the explanatory comments (threading, ordering, host quirks): explain why,
  in prose, without Doxygen tags.

## Definition of done

- `clang-format` (repo `.clang-format`) on changed C++.
- Full build and full `ctest` pass, including the example for CMake changes.
- `reuse lint` reports nothing beyond the three `tests/files` captures, whose
  redistribution terms are unknown and which are deliberately not in
  `REUSE.toml`. Never add a license for them, or for any file, on assumption.
- README updated for API changes.

## Licensing

- All files are MIT. Files adapted from TONE3000 code carry both
  `2026 TONE3000` and `2026 Pier Luigi Fiorini`; preserve upstream holders and
  years. Never replace an upstream notice with an own-only header.
- Dependencies keep their own licenses; add text under `LICENSES/` as needed.

## Licensing hooks

Checker: `.agents/hooks/check-reuse.py`; configs in `.claude/settings.json` and
`.codex/hooks.json`. Details in [README](README.md#licensing-hooks). The checker
reports problems without assigning licenses; treat a failure as unfinished
verification and preserve actual attribution when fixing.

```sh
python3 .agents/hooks/check-reuse.py --check
python3 -m unittest discover -s tests -p 'test_reuse_hook.py' -v
```

These need Python 3.9+, Git and the REUSE CLI, and use temporary Git repos.
