# SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
# SPDX-License-Identifier: MIT

"""Integration tests against real Git and REUSE, isolated from the user's worktree."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
HEADER = "# SPDX-FileCopyrightText: 2026 Test Author\n# SPDX-License-Identifier: MIT\n"


@unittest.skipUnless(shutil.which("git") and shutil.which("reuse"), "Git and REUSE are required")
class ReuseHookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nam reuse hook ")
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        # Fixtures must not invoke the user's signing setup or Git hooks.
        self.environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.root = self.repository / "library with spaces"
        self.root.mkdir()
        for name in (".agents/hooks/check-reuse.py", ".claude/settings.json", ".codex/hooks.json",
                     "REUSE.toml", ".gitignore", "LICENSES/MIT.txt"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, destination)
        (self.root / "good.py").write_text(HEADER + "value = 1\n")
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Hook Test")
        self.commit()

    def git(self, *arguments):
        result = subprocess.run(["git", *arguments], cwd=self.repository, env=self.environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "Test fixture")

    def invoke(self, payload=None, *, environment=None, command=None, cwd=None):
        if payload is None:
            payload = {"hook_event_name": "PostToolUse", "tool_name": "Bash"}
        result = subprocess.run(
            command or [sys.executable, str(self.root / ".agents/hooks/check-reuse.py")],
            input=json.dumps(payload), text=True, cwd=cwd or self.root,
            env=environment or self.environment, shell=isinstance(command, str),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True,
        )
        return json.loads(result.stdout)

    def test_clean_tree_and_unrelated_unlicensed_files(self):
        (self.root / "legacy.txt").write_text("An unchanged pre-existing licensing gap\n")
        self.commit()
        (self.repository / "outside.txt").write_text("Outside the library\n")
        self.assertEqual(self.invoke(), {})
        (self.root / "new.py").write_text(HEADER)
        self.assertEqual(self.invoke(), {})

    def test_new_unlicensed_file_with_unusual_filename(self):
        (self.root / "new è file\nsecond line.py").write_text("value = 2\n")
        result = self.invoke({"hook_event_name": "PostToolUse", "tool_name": "Write"})
        self.assertEqual(result["decision"], "block")
        self.assertIn("REUSE licensing check failed", result["reason"])

    def test_edited_file_and_repair(self):
        path = self.root / "good.py"
        path.write_text("value = 2\n")
        self.assertEqual(self.invoke()["decision"], "block")
        path.write_text(HEADER + "value = 2\n")
        self.assertEqual(self.invoke(), {})

    def test_staged_new_file(self):
        path = self.root / "staged.py"
        path.write_text("value = 1\n")
        self.git("add", str(path))
        self.assertEqual(self.invoke()["decision"], "block")

    def test_renamed_file(self):
        source = self.root / "good.py"
        target = self.root / "renamed.py"
        self.git("mv", str(source), str(target))
        target.write_text("value = 1\n")
        self.assertEqual(self.invoke()["decision"], "block")

    def test_ordinary_deletion_is_skipped(self):
        (self.root / "good.py").unlink()
        self.assertEqual(self.invoke(), {})

    def test_license_deletion_checks_unchanged_files(self):
        (self.root / "LICENSES/MIT.txt").unlink()
        result = self.invoke()
        self.assertEqual(result["decision"], "block")
        self.assertIn("MIT", result["reason"])

    def test_metadata_edit_checks_unchanged_files(self):
        path = self.root / "REUSE.toml"
        path.write_text(HEADER + "version = 1\n")
        result = self.invoke()
        self.assertEqual(result["decision"], "block")
        self.assertIn("settings.json", result["reason"])

    def test_ignored_output_and_outside_symlinks(self):
        output = self.root / "build"
        output.mkdir()
        (output / "generated.cpp").write_text("Generated output\n")
        outside = self.repository / "outside.py"
        outside.write_text("Private unrelated contents\n")
        (self.root / "link.py").symlink_to(outside)
        self.assertEqual(self.invoke(), {})

    def test_stop_reports_failure_without_an_endless_continuation(self):
        (self.root / "bad.py").write_text("value = 1\n")
        self.assertEqual(self.invoke({"hook_event_name": "Stop"})["decision"], "block")
        repeated = self.invoke({"hook_event_name": "Stop", "stop_hook_active": True})
        self.assertNotIn("decision", repeated)
        self.assertIn("failed", repeated["systemMessage"])

    def test_missing_reuse_is_reported(self):
        environment = dict(self.environment, PATH=str(self.repository / "missing-bin"))
        result = self.invoke(environment=environment)
        self.assertEqual(result["decision"], "block")
        self.assertIn("REUSE is not installed", result["reason"])

    def test_configuration_commands_from_nested_cwd(self):
        nested = self.root / "subdirectory"
        nested.mkdir()
        (self.root / "bad.py").write_text("value = 1\n")
        environment = dict(self.environment, CLAUDE_PROJECT_DIR=str(self.root))
        for name in (".claude/settings.json", ".codex/hooks.json"):
            config = json.loads((self.root / name).read_text())
            for event in ("PostToolUse", "Stop"):
                command = config["hooks"][event][0]["hooks"][0]["command"]
                with self.subTest(configuration=name, event=event):
                    result = self.invoke({"hook_event_name": event}, command=command,
                                         environment=environment, cwd=nested)
                    self.assertEqual(result["decision"], "block")

    def test_folder_without_git_uses_full_lint(self):
        shutil.rmtree(self.repository / ".git")
        self.assertEqual(self.invoke(), {})
        (self.root / "bad.py").write_text("value = 1\n")
        self.assertEqual(self.invoke()["decision"], "block")


if __name__ == "__main__":
    unittest.main()
