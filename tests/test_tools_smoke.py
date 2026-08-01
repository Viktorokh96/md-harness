"""Smoke tests for all CONTROL ROOM agent tools.

These test that every tool:
1. Can be called without raising Python-level errors (syntax, NameError, etc.)
2. Returns a valid result with expected structure
3. Handles edge cases (bad input, missing files, timeouts)

NO LLM calls — tool implementation functions are tested directly.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from agent import ControlRoomContext
from agent import _impl_add_reply
from agent import _impl_batch_reply
from agent import _impl_find_nodes
from agent import _impl_read_file
from agent import _impl_read_mindmap
from agent import _impl_run_shell
from agent import _impl_stay_silent

if TYPE_CHECKING:
    from pathlib import Path

# ══ Test data ═════════════════════════════════════════════════════════════════

MINDMAP_TEMPLATE = """\
# CONTROL ROOM TEST

```agentsmindmap
root: SMOKE TEST
* user message one
  [*] agent reply one
* user message two
```
"""


# ══ Fixtures ══════════════════════════════════════════════════════════════════


@pytest.fixture
def ctx_with_mindmap(tmp_path: Path) -> ControlRoomContext:
    """Create a ControlRoomContext with a valid mindmap file."""
    md_file = tmp_path / "CONTROL_ROOM.md"
    md_file.write_text(MINDMAP_TEMPLATE)
    return ControlRoomContext(file_path=md_file)


@pytest.fixture
def ctx_empty(tmp_path: Path) -> ControlRoomContext:
    """Create a ControlRoomContext with an empty file (no mindmap block)."""
    md_file = tmp_path / "CONTROL_ROOM.md"
    md_file.write_text("")  # empty file, no agentsmindmap block
    return ControlRoomContext(file_path=md_file)


# ══ stay_silent ═══════════════════════════════════════════════════════════════


def test_stay_silent_returns_marker() -> None:
    """Pure function — no deps, no side effects."""
    result = _impl_stay_silent()
    assert isinstance(result, str)
    assert "SILENT" in result


# ══ run_shell ═════════════════════════════════════════════════════════════════


def test_run_shell_echo() -> None:
    """Executes a simple command and returns stdout."""
    result = _impl_run_shell("echo hello")
    assert isinstance(result, str)
    assert "hello" in result


def test_run_shell_no_output() -> None:
    """Commands with no stdout/stderr return (no output)."""
    result = _impl_run_shell("true")
    assert result == "(no output)"


def test_run_shell_stderr_is_captured() -> None:
    """Captures stderr and marks it clearly."""
    result = _impl_run_shell("echo err >&2")
    assert "[stderr]" in result
    assert "err" in result


def test_run_shell_exit_code() -> None:
    """Reports non-zero exit codes."""
    result = _impl_run_shell("exit 42")
    assert "[exit code: 42]" in result


def test_run_shell_timeout_is_reported() -> None:
    """Timeout exceptions produce a clear error message."""
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("cmd", 1),
    ):
        result = _impl_run_shell("sleep 999", timeout=1)
        assert "timed out" in result
        assert isinstance(result, str)


def test_run_shell_returns_string_on_error() -> None:
    """Even error conditions return strings, never raise."""
    result = _impl_run_shell("echo 'test with spaces'")
    assert isinstance(result, str)
    assert len(result) > 0


# ══ read_file ═════════════════════════════════════════════════════════════════


def test_read_file_reads_existing_file(tmp_path: Path) -> None:
    """Reads a file in the allowed directory."""
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("line1\nline2\nline3")
    result = _impl_read_file("test_data.txt", root_dir=tmp_path)
    assert "line1" in result
    assert "line2" in result


def test_read_file_missing_file(tmp_path: Path) -> None:
    """Returns a clear ERROR for missing files."""
    result = _impl_read_file("nonexistent.txt", root_dir=tmp_path)
    assert "ERROR" in result
    assert "not found" in result


def test_read_file_path_traversal_blocked(tmp_path: Path) -> None:
    """Blocks path traversal attempts escaping root_dir."""
    result = _impl_read_file("../../../etc/passwd", root_dir=tmp_path)
    assert "ERROR" in result
    assert "escapes project root" in result


def test_read_file_truncation(tmp_path: Path) -> None:
    """Truncates long files and reports total line count."""
    test_file = tmp_path / "long.txt"
    test_file.write_text("\n".join(f"line {i}" for i in range(300)))
    result = _impl_read_file("long.txt", root_dir=tmp_path, max_lines=10)
    assert "truncated" in result
    assert "300 total" in result


# ══ read_mindmap ══════════════════════════════════════════════════════════════


def test_read_mindmap_returns_outline(ctx_with_mindmap: ControlRoomContext) -> None:
    """Returns a readable outline with node IDs."""
    result = _impl_read_mindmap(ctx_with_mindmap)
    assert isinstance(result, str)
    assert "SMOKE TEST" in result
    assert "user message one" in result
    assert "agent reply one" in result
    assert "root" in result


def test_read_mindmap_empty_file_raises(ctx_empty: ControlRoomContext) -> None:
    """Raises ValueError on empty file (no agentsmindmap block)."""
    with pytest.raises(ValueError, match=r"No .* block found"):
        _impl_read_mindmap(ctx_empty)


# ══ add_reply ═════════════════════════════════════════════════════════════════


def test_add_reply_adds_child(ctx_with_mindmap: ControlRoomContext) -> None:
    """Creates a new agent node under the given parent."""
    result = _impl_add_reply(ctx_with_mindmap, "root.1", "test reply content")
    assert "Added agent reply" in result
    assert "root.1" in result
    assert "test reply content" in result
    file_content = ctx_with_mindmap.file_path.read_text()
    assert "test reply content" in file_content


def test_add_reply_invalid_parent_raises(ctx_with_mindmap: ControlRoomContext) -> None:
    """Raises KeyError on nonexistent parent_id."""
    with pytest.raises(KeyError):
        _impl_add_reply(ctx_with_mindmap, "root.999", "orphan reply")


# ══ batch_reply ═══════════════════════════════════════════════════════════════


def test_batch_reply_creates_multiple_branches(ctx_with_mindmap: ControlRoomContext) -> None:
    """Creates N separate branches under parent."""
    replies = ["idea one", "idea two", "idea three"]
    result = _impl_batch_reply(ctx_with_mindmap, "root.2", replies)
    assert "Added 3 replies" in result
    assert "root.2" in result
    file_content = ctx_with_mindmap.file_path.read_text()
    for reply in replies:
        assert reply in file_content


def test_batch_reply_empty_list(ctx_with_mindmap: ControlRoomContext) -> None:
    """Empty list adds zero replies."""
    result = _impl_batch_reply(ctx_with_mindmap, "root.2", [])
    assert "Added 0 replies" in result


# ══ find_nodes ════════════════════════════════════════════════════════════════


def test_find_nodes_finds_existing(ctx_with_mindmap: ControlRoomContext) -> None:
    """Locates nodes by substring match."""
    result = _impl_find_nodes(ctx_with_mindmap, "user")
    assert "user message one" in result
    assert "user message two" in result


def test_find_nodes_no_match(ctx_with_mindmap: ControlRoomContext) -> None:
    """Reports when nothing matches."""
    result = _impl_find_nodes(ctx_with_mindmap, "zzzzznonexistent")
    assert "No nodes found" in result


def test_find_nodes_case_insensitive(ctx_with_mindmap: ControlRoomContext) -> None:
    """Search is case-insensitive."""
    result = _impl_find_nodes(ctx_with_mindmap, "USER")
    assert "user message" in result


# ══ Cross-tool chain ══════════════════════════════════════════════════════════


def test_tool_chain_read_add_read(ctx_with_mindmap: ControlRoomContext) -> None:
    """End-to-end: read tree, add reply, verify persistence."""
    initial = _impl_read_mindmap(ctx_with_mindmap)
    assert "user message one" in initial
    _impl_add_reply(ctx_with_mindmap, "root.1", "chain test reply")
    updated = _impl_read_mindmap(ctx_with_mindmap)
    assert "chain test reply" in updated
