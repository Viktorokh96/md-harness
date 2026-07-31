"""Tests for CONTROL ROOM agent — pure helpers and agent creation (no LLM)."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent import (
    _do_append,
    _do_read_file,
    _do_run_shell,
    _find_last_user_line,
    _find_subtree_end,
    create_agent,
)


# ── _find_last_user_line ────────────────────────────────────────────────────


class TestFindLastUserLine:
    def test_simple(self) -> None:
        idx, depth = _find_last_user_line(["* Hello"])
        assert idx == 0
        assert depth == 0

    def test_ignores_agent_lines(self) -> None:
        idx, depth = _find_last_user_line(["\t[*] Agent reply", "* Hello"])
        assert idx == 1
        assert depth == 0

    def test_nested_user_line(self) -> None:
        idx, depth = _find_last_user_line(["* Q1", "\t* Q2"])
        assert idx == 1
        assert depth == 1

    def test_no_user_line(self) -> None:
        idx, depth = _find_last_user_line(["\t[*] Only agent"])
        assert idx == -1
        assert depth == 0

    def test_last_user_wins(self) -> None:
        idx, depth = _find_last_user_line(["* First", "\t[*] Agent", "* Last"])
        assert idx == 2
        assert depth == 0


# ── _find_subtree_end ───────────────────────────────────────────────────────


class TestFindSubtreeEnd:
    def test_empty_subtree(self) -> None:
        """A line with no children — subtree ends right after it."""
        end = _find_subtree_end(["* Q", "* Next"], 0, 0)
        assert end == 1

    def test_one_child(self) -> None:
        end = _find_subtree_end(["* Q", "\t[*] Answer", "* Next"], 0, 0)
        assert end == 2

    def test_nested_children(self) -> None:
        lines = [
            "* Q",
            "\t[*] A1",
            "\t\t[*] Detail",
            "\t[*] A2",
            "* Next",
        ]
        end = _find_subtree_end(lines, 0, 0)
        assert end == 4

    def test_skips_empty_lines(self) -> None:
        lines = ["* Q", "", "\t[*] Answer", "", "* Next"]
        end = _find_subtree_end(lines, 0, 0)
        assert end == 4

    def test_last_line_no_sibling(self) -> None:
        lines = ["* Q", "\t[*] Answer"]
        end = _find_subtree_end(lines, 0, 0)
        assert end == 2  # past the end


# ── _do_append (tree-aware) ─────────────────────────────────────────────────


class TestDoAppend:
    def test_inserts_after_last_user_subtree(self) -> None:
        """Response goes after the last user message's subtree, not at EOF."""
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("* Q1\n\t[*] Old answer\n* Q2\n")
            _do_append(fpath, "\t[*] New answer\n")

            content = fpath.read_text()
            lines = content.splitlines()
    def test_inserts_before_sibling(self) -> None:
        """Response to Q1 goes before Q2 when after_line="* Q1"."""
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("* Q1\n* Q2\n")
            _do_append(fpath, "\t[*] Answer to Q1\n", after_line="* Q1")

            content = fpath.read_text()
            lines = content.splitlines()
            ans_idx = next(i for i, l in enumerate(lines) if "Answer to Q1" in l)
            q2_idx = next(i for i, l in enumerate(lines) if "Q2" in l)
            assert ans_idx < q2_idx

    def test_defaults_to_last_user_line(self) -> None:
        """Without after_line, reply goes to the LAST user message."""
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("* Q1\n* Q2\n")
            _do_append(fpath, "\t[*] Answer to Q2\n")

            content = fpath.read_text()
            lines = content.splitlines()
            # Answer to Q2 should be after Q2
            q2_idx = next(i for i, l in enumerate(lines) if "Q2" in l)
            ans_idx = next(i for i, l in enumerate(lines) if "Answer to Q2" in l)
            assert ans_idx > q2_idx

    def test_fallback_to_append_when_no_user_line(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("# Just a header\n")
            result = _do_append(fpath, "\t[*] Response\n")
            content = fpath.read_text()
            assert content.endswith("\t[*] Response\n")
            assert "Appended" in result

    def test_multiline_insert(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("* Q\n")
            _do_append(fpath, "\t[*] L1\n\t\t[*] L2\n\t[*] L3\n")

            content = fpath.read_text()
            lines = content.splitlines()
            assert lines[1] == "\t[*] L1"
            assert lines[2] == "\t\t[*] L2"
            assert lines[3] == "\t[*] L3"


# ── _do_read_file ───────────────────────────────────────────────────────────


class TestDoReadFile:
    def test_reads_existing_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.txt").write_text("hello\nworld\n")
            result = _do_read_file(root, "data.txt")
            assert "hello" in result
            assert "world" in result

    def test_file_not_found(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _do_read_file(Path(tmp), "nope.txt")
            assert "ERROR" in result
            assert "not found" in result

    def test_path_escape_prevented(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _do_read_file(Path(tmp), "../../../etc/passwd")
            assert "ERROR" in result
            assert "escapes" in result

    def test_truncates_long_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "long.txt").write_text(
                "\n".join(str(i) for i in range(300))
            )
            result = _do_read_file(root, "long.txt", max_lines=10)
            assert "truncated" in result
            assert "300 total lines" in result


# ── _do_run_shell ───────────────────────────────────────────────────────────


class TestDoRunShell:
    def test_echo(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _do_run_shell(Path(tmp), "echo hello")
            assert "hello" in result

    def test_cwd_respected(self) -> None:
        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "sub"
            cwd.mkdir()
            result = _do_run_shell(cwd, "pwd")
            assert str(cwd) in result

    def test_command_error(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _do_run_shell(Path(tmp), "nonexistent_cmd_xyz 2>/dev/null")
            assert "exit code" in result.lower()

    def test_timeout(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _do_run_shell(Path(tmp), "sleep 5", timeout=1)
            assert "timed out" in result.lower()


# ── create_agent ────────────────────────────────────────────────────────────


class TestCreateAgent:
    def test_creates_with_defaults(self) -> None:
        agent = create_agent()
        assert agent.name == "ControlRoom"
        assert len(agent.tools) == 5

    def test_tool_names(self) -> None:
        agent = create_agent()
        names = {t.name for t in agent.tools}
        assert names == {
            "append_to_control_room",
            "stay_silent",
            "run_shell",
            "read_file",
            "delegate_task",
        }

    def test_model_override(self) -> None:
        agent = create_agent(model="custom-model")
        assert agent.model == "custom-model"

    def test_model_from_env(self) -> None:
        os.environ["LLM_MODEL"] = "env-model"
        try:
            agent = create_agent()
            assert agent.model == "env-model"
        finally:
            del os.environ["LLM_MODEL"]

    def test_explicit_model_wins_over_env(self) -> None:
        os.environ["LLM_MODEL"] = "env-model"
        try:
            agent = create_agent(model="explicit-model")
            assert agent.model == "explicit-model"
        finally:
            del os.environ["LLM_MODEL"]
