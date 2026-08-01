"""Tests for CONTROL ROOM watcher and agent components."""

from __future__ import annotations

import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

# Import after setting up sys.path if needed
from agent import ControlRoomContext
from agent import compute_diff
from watcher import MIND_MAP_TEMPLATE
from watcher import THINKING_MARKER
from watcher import _append_placeholder
from watcher import _remove_thinking_markers

# ── compute_diff tests ──────────────────────────────────────────────────────


class TestComputeDiff:
    """Tests for compute_diff() — line-based diff for LLM consumption."""

    def test_pure_append(self) -> None:
        old = "line1\nline2\n"
        new = "line1\nline2\nline3\nline4\n"
        result = compute_diff(old, new)
        assert "Added:" in result
        assert "+ line3" in result
        assert "+ line4" in result
        assert "-" not in result

    def test_pure_append_single_line(self) -> None:
        old = "hello\n"
        new = "hello\nworld\n"
        result = compute_diff(old, new)
        assert "+ world" in result

    def test_pure_truncation(self) -> None:
        old = "a\nb\nc\n"
        new = "a\n"
        result = compute_diff(old, new)
        assert "Removed:" in result
        assert "- b" in result
        assert "- c" in result

    def test_inline_change(self) -> None:
        old = "line1\nline2\nline3\n"
        new = "line1\nline2_changed\nline3\n"
        result = compute_diff(old, new)
        assert "@@ line 2:" in result
        assert "- line2" in result
        assert "+ line2_changed" in result

    def test_no_changes(self) -> None:
        content = "a\nb\nc\n"
        result = compute_diff(content, content)
        assert "no visible changes" in result

    def test_empty_to_content(self) -> None:
        result = compute_diff("", "hello\n")
        assert "+ hello" in result

    def test_content_to_empty(self) -> None:
        result = compute_diff("hello\n", "")
        assert "- hello" in result

    def test_multiple_inline_changes(self) -> None:
        old = "a\nb\nc\nd\ne\n"
        new = "a\nB\nc\nD\ne\n"
        result = compute_diff(old, new)
        assert "- b" in result
        assert "+ B" in result
        assert "- d" in result
        assert "+ D" in result

    def test_complete_rewrite(self) -> None:
        old = "old1\nold2\n"
        new = "new1\nnew2\nnew3\n"
        result = compute_diff(old, new)
        assert "- old1" in result
        assert "- old2" in result
        assert "+ new1" in result
        assert "+ new2" in result
        assert "+ new3" in result

    def test_mind_map_diff(self) -> None:
        """Realistic mind map append scenario."""
        old = textwrap.dedent("""\
        # CONTROL ROOM

        * Hello
          [*] Hi there!
        """)
        new = textwrap.dedent("""\
        # CONTROL ROOM

        * Hello
          [*] Hi there!
        * New question
        """)
        result = compute_diff(old, new)
        assert "Added:" in result
        assert "+ * New question" in result


# ── Placeholder tests ───────────────────────────────────────────────────────


class TestPlaceholder:
    """Tests for thinking placeholder write/cleanup."""

    def test_appends_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("# Test\n\n```agentsmindmap\nroot: R\n```\n")

            ctx = ControlRoomContext(file_path=fpath)
            _append_placeholder(ctx)

            content = fpath.read_text()
            assert THINKING_MARKER in content

    def test_appends_even_without_trailing_newline(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text("text without newline")

            ctx = ControlRoomContext(file_path=fpath)
            _append_placeholder(ctx)

            assert THINKING_MARKER in fpath.read_text()

    def test_remove_thinking_markers(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            fpath.write_text(f"# Header\n\n```agentsmindmap\nroot: R\n```\n{THINKING_MARKER}\n")

            ctx = ControlRoomContext(file_path=fpath)
            result = _remove_thinking_markers(ctx)

            assert THINKING_MARKER not in result
            assert "root: R" in result

    def test_remove_thinking_markers_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test.md"
            original = "# Just text\n"
            fpath.write_text(original)

            ctx = ControlRoomContext(file_path=fpath)
            result = _remove_thinking_markers(ctx)

            assert result.rstrip() == original.rstrip()


# ── Template tests ──────────────────────────────────────────────────────────


class TestTemplate:
    """Tests for mind map template."""

    def test_template_has_header(self) -> None:
        assert "# CONTROL ROOM" in MIND_MAP_TEMPLATE

    def test_template_has_tree_block(self) -> None:
        assert "```agentsmindmap" in MIND_MAP_TEMPLATE
        assert "root: CONTROL ROOM" in MIND_MAP_TEMPLATE

    def test_template_no_thinking_marker(self) -> None:
        assert THINKING_MARKER not in MIND_MAP_TEMPLATE


# ── ControlRoomContext tests ────────────────────────────────────────────────


class TestControlRoomContext:
    """Tests for the context dataclass."""

    def test_default_file_path(self) -> None:
        ctx = ControlRoomContext()
        assert ctx.file_path.name == "CONTROL_ROOM.md"

    def test_custom_file_path(self) -> None:
        ctx = ControlRoomContext(file_path=Path("/tmp/custom.md"))
        assert ctx.file_path == Path("/tmp/custom.md")

    def test_initial_state_empty(self) -> None:
        ctx = ControlRoomContext()
        assert ctx.last_content == ""
        assert ctx.last_mtime == 0.0

    def test_mutable_state(self) -> None:
        ctx = ControlRoomContext()
        ctx.last_content = "test"
        ctx.last_mtime = 42.0
        assert ctx.last_content == "test"
        assert ctx.last_mtime == 42.0


# ── Integration-style tests ─────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end workflow simulations (no LLM calls)."""

    def test_full_cycle_no_llm(self) -> None:
        """Simulate: create file, placeholder, cleanup."""
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "room.md"
            fpath.write_text(MIND_MAP_TEMPLATE)

            ctx = ControlRoomContext(file_path=fpath)
            ctx.last_content = fpath.read_text()

            # Placeholder write
            _append_placeholder(ctx)
            assert THINKING_MARKER in fpath.read_text()

            # Cleanup
            cleaned = _remove_thinking_markers(ctx)
            assert THINKING_MARKER not in cleaned
            assert "root: CONTROL ROOM" in cleaned
