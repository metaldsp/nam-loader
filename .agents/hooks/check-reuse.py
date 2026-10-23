#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
# SPDX-License-Identifier: MIT

"""Check changed files with REUSE; share the hook protocol across Claude and Codex."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 20


def run(command, root):
    return subprocess.run(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=TIMEOUT, check=False,
    )


def changed_files(root):
    """Return paths relative to this project, or None outside a Git worktree."""
    if not shutil.which("git"):
        raise RuntimeError("Git is required to identify changed files.")
    probe = run(["git", "rev-parse", "--is-inside-work-tree"], root)
    if probe.returncode != 0:
        # A copied source folder may not have a repository yet.
        return None

    paths = set()
    commands = [
        ["git", "diff", "--name-only", "--no-renames", "--relative", "-z",
         "--diff-filter=ACMRT", "--", "."],
        ["git", "diff", "--cached", "--name-only", "--no-renames", "--relative", "-z",
         "--diff-filter=ACMRT", "--", "."],
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "."],
    ]
    for command in commands:
        result = run(command, root)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace").strip() or "Git check failed")
        paths.update(Path(os.fsdecode(name)) for name in result.stdout.split(b"\0") if name)

    selected = []
    for path in sorted(paths):
        absolute = root / path
        # Deleted files, submodules and symlinks have no regular file to lint.
        # In particular, never follow a symlink outside the library.
        if absolute.is_symlink() or not absolute.is_file():
            continue
        if not absolute.resolve().is_relative_to(root.resolve()):
            continue
        selected.append(path)
    return selected


def check(root):
    executable = shutil.which("reuse")
    if executable is None:
        raise RuntimeError("REUSE is not installed. Install it with `pipx install reuse`.")
    paths = changed_files(root)
    # Licensing metadata affects files beyond the edited file itself. A full
    # lint also catches removed license texts; include deletions in this check.
    metadata_changed = False
    for command in (
        ["git", "diff", "--name-only", "--relative", "-z", "--", "."],
        ["git", "diff", "--cached", "--name-only", "--relative", "-z", "--", "."],
    ) if paths is not None else ():
        result = run(command, root)
        if result.returncode:
            raise RuntimeError("Could not inspect licensing metadata changes")
        names = [Path(os.fsdecode(name)) for name in result.stdout.split(b"\0") if name]
        metadata_changed |= any(is_metadata(name) for name in names)
    if paths == [] and not metadata_changed:
        return ""
    full = paths is None or metadata_changed or any(is_metadata(path) for path in paths)
    command = [executable, "--root", str(root), "--no-multiprocessing"]
    if full:
        batches = [[*command, "lint", "--lines"]]
    else:
        # Bound argv size for large generated patches and unusual filenames.
        batches = [
            [*command, "lint-file", "--", *(str(root / path) for path in paths[start:start + 100])]
            for start in range(0, len(paths), 100)
        ]
    for batch in batches:
        result = run(batch, root)
        if result.returncode:
            detail = (result.stdout + result.stderr).decode(errors="replace").strip()
            return "REUSE licensing check failed:\n" + (detail or "REUSE returned a failure status")
    return ""


def is_metadata(path):
    return path.name in {"REUSE.toml", ".reuse.toml", "LICENSE", "COPYING"} or \
        path.parts[0] in {"LICENSES", ".reuse"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run manually without a hook payload")
    args = parser.parse_args()
    payload = {}
    if not args.check:
        try:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except (ValueError, OSError) as error:
            print(json.dumps({"decision": "block", "reason": f"Invalid REUSE hook input: {error}"}))
            return 0
    try:
        problem = check(ROOT)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        problem = f"REUSE check could not complete: {error}"
    if args.check:
        print(problem or "result: PASS")
        return int(bool(problem))
    if not problem:
        print("{}")
    else:
        # Report unresolved errors once more without trapping the agent in an
        # endless Stop continuation when tooling or attribution needs a human.
        reason = (problem[:6000] + ("\n[output truncated]" if len(problem) > 6000 else "") +
                  "\nFix the licensing metadata while preserving its actual origin. "
                  "Do not guess a license or report this check as passing.")
        if payload.get("hook_event_name") == "Stop" and payload.get("stop_hook_active"):
            print(json.dumps({"systemMessage": reason}))
        else:
            print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
