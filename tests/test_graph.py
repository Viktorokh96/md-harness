"""Tests for [hide] mechanics, JSON round-trip, graph sync."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from tree_engine import (
    HIDE_RE,
    TAG_RE,
    MindNode,
    MindTree,
    graph_path,
    load_graph,
    merge_md_into_graph,
    parse_mindmap,
    save_graph,
    serialize_mindmap,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _md(body: str) -> str:
    return f"# Test\n\n```agentsmindmap\n{body}```\n"


# ── Continuation lines ──────────────────────────────────────────────────────


class TestContinuation:
    def test_appends_to_parent(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Line one\n  continues here\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert "continues here" in n.content

    def test_multiple_continuations(self) -> None:
        tree = parse_mindmap(_md("root: R\n* A\n  line2\n  line3\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert "line2\nline3" in n.content

    def test_round_trip(self) -> None:
        text = _md("root: R\n* Multi\n  line\n  [*] Reply\n")
        t1 = parse_mindmap(text)
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        assert t2.to_outline() == t1.to_outline()

    def test_serialize_indents_continuations(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Multi\n  line\n"))
        s = serialize_mindmap(tree)
        # Continuation "line" should be indented
        lines = s.splitlines()
        assert any("  line" in ln for ln in lines)


# ── [hide] parsing ──────────────────────────────────────────────────────────


class TestHideParsing:
    def test_user_hide(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[hide] Hidden topic\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.hidden
        assert n.is_user
        assert n.content == "Hidden topic"

    def test_agent_hide(self) -> None:
        tree = parse_mindmap(_md("root: R\n[*][hide] Hidden reply\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.hidden
        assert n.is_agent
        assert n.content == "Hidden reply"

    def test_hide_with_children(self) -> None:
        tree = parse_mindmap(_md(
            "root: R\n"
            "*[hide] Hidden\n"
            "  [*] child1\n"
            "  [*] child2\n"
        ))
        n = tree.get_node("root.1")
        assert n is not None and n.hidden
        assert len(n.children) == 2
        assert n.children[0].content == "child1"

    def test_round_trip(self) -> None:
        text = _md("root: R\n*[hide] H\n  [*] A\n* V\n")
        t1 = parse_mindmap(text)
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        assert t2.get_node("root.1") is not None
        assert t2.get_node("root.1").hidden
        assert t2.get_node("root.2") is not None
        assert not t2.get_node("root.2").hidden

    def test_serialize_excludes_hidden_children(self) -> None:
        tree = parse_mindmap(_md(
            "root: R\n*[hide] H\n  [*] A1\n  [*] A2\n"
        ))
        s = serialize_mindmap(tree)
        assert "*[hide] H" in s
        assert "A1" not in s
        assert "A2" not in s

    def test_outline_marks_hidden(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[hide] H\n"))
        outline = tree.to_outline()
        assert "📦" in outline

    def test_outline_shows_hidden_children(self) -> None:
        tree = parse_mindmap(_md(
            "root: R\n*[hide] H\n  [*] A1\n"
        ))
        outline = tree.to_outline()
        assert "A1" in outline  # by default show_hidden=True

    def test_outline_hides_hidden_children(self) -> None:
        tree = parse_mindmap(_md(
            "root: R\n*[hide] H\n  [*] A1\n"
        ))
        outline = tree.to_outline(show_hidden=False)
        assert "A1" not in outline

    def test_regex_hide_user(self) -> None:
        m = HIDE_RE.match("*[hide] Summary")
        assert m is not None
        assert m.group(1) == "*"

    def test_regex_hide_agent(self) -> None:
        m = HIDE_RE.match("[*][hide] Summary")
        assert m is not None
        assert m.group(1) == "[*]"

    def test_regex_no_hide(self) -> None:
        assert HIDE_RE.match("* Normal") is None
        assert TAG_RE.match("* Normal") is not None
        assert TAG_RE.match("[*] Normal") is not None


# ── JSON round-trip ─────────────────────────────────────────────────────────


class TestJsonRoundTrip:
    def test_simple(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Q\n  [*] A\n"))
        d = tree.to_dict()
        t2 = MindTree.from_dict(d)
        assert t2.to_outline() == tree.to_outline()

    def test_with_hide(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[hide] H\n  [*] A\n"))
        d = tree.to_dict()
        t2 = MindTree.from_dict(d)
        assert t2.get_node("root.1").hidden
        assert len(t2.get_node("root.1").children) == 1

    def test_with_continuations(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Multi\n  line\n"))
        d = tree.to_dict()
        t2 = MindTree.from_dict(d)
        assert "line" in t2.get_node("root.1").content

    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "test.md")
            tree = parse_mindmap(_md("root: R\n* Q\n  [*] A\n*[hide] H\n"))
            save_graph(tree, md_path)

            gp = graph_path(md_path)
            assert os.path.exists(gp)

            loaded = load_graph(md_path)
            assert loaded is not None
            assert loaded.to_outline() == tree.to_outline()

    def test_load_missing(self) -> None:
        assert load_graph("/nonexistent/path.md") is None

    def test_to_dict_fields(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Q\n"))
        d = tree.to_dict()
        assert d["id"] == "root"
        assert d["author"] == "system"
        assert not d["hidden"]
        assert len(d["children"]) == 1

    def test_json_serializable(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Q\n"))
        d = tree.to_dict()
        s = json.dumps(d)
        assert '"id"' in s
        json.loads(s)  # no errors


# ── Hide-only diff detection ────────────────────────────────────────────────


class TestHideOnlyDiff:
    """Tests for _is_hide_only_diff in watcher.py."""

    @pytest.fixture(autouse=True)
    def _import_watcher(self) -> None:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "watcher_test", "watcher.py"
        )
        self.watcher = importlib.util.module_from_spec(spec)
        sys.modules["watcher_test"] = self.watcher
        spec.loader.exec_module(self.watcher)

    def test_hide_added(self) -> None:
        old = '# H\n\n```agentsmindmap\nroot: R\n* Q\n```\n'
        new = '# H\n\n```agentsmindmap\nroot: R\n*[hide] Q\n```\n'
        assert self.watcher._is_hide_only_diff(old, new)

    def test_hide_removed(self) -> None:
        old = '# H\n\n```agentsmindmap\nroot: R\n*[hide] Q\n```\n'
        new = '# H\n\n```agentsmindmap\nroot: R\n* Q\n```\n'
        assert self.watcher._is_hide_only_diff(old, new)

    def test_new_content_not_hide_only(self) -> None:
        old = '# H\n\n```agentsmindmap\nroot: R\n* Q1\n```\n'
        new = '# H\n\n```agentsmindmap\nroot: R\n* Q1\n* Q2\n```\n'
        assert not self.watcher._is_hide_only_diff(old, new)

    def test_content_change_not_hide_only(self) -> None:
        old = '# H\n\n```agentsmindmap\nroot: R\n* Hello\n```\n'
        new = '# H\n\n```agentsmindmap\nroot: R\n* World\n```\n'
        assert not self.watcher._is_hide_only_diff(old, new)

    def test_identical_not_hide_only(self) -> None:
        text = '# H\n\n```agentsmindmap\nroot: R\n* Q\n```\n'
        assert not self.watcher._is_hide_only_diff(text, text)

    def test_agent_hide_toggle(self) -> None:
        old = '# H\n\n```agentsmindmap\nroot: R\n  [*] Reply\n```\n'
        new = '# H\n\n```agentsmindmap\nroot: R\n  [*][hide] Reply\n```\n'
        assert self.watcher._is_hide_only_diff(old, new)


# ── Graph sync ──────────────────────────────────────────────────────────────


class TestGraphSync:
    """Tests for _sync_graph in watcher.py."""

    @pytest.fixture(autouse=True)
    def _import_watcher(self) -> None:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "watcher_sync", "watcher.py"
        )
        self.watcher = importlib.util.module_from_spec(spec)
        sys.modules["watcher_sync"] = self.watcher
        spec.loader.exec_module(self.watcher)

    class FakeCtx:
        file_path: Path

    def test_creates_graph_on_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"
            md_path.write_text(_md("root: Room\n* Hello\n"))

            ctx = self.FakeCtx()
            ctx.file_path = md_path
            self.watcher._sync_graph(ctx)

            gp = graph_path(str(md_path))
            assert os.path.exists(gp)

            loaded = load_graph(str(md_path))
            assert loaded is not None
            assert loaded.get_node("root.1") is not None

    def test_merge_adds_new_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"
            # First run — create graph
            md_path.write_text(_md("root: Room\n* Q1\n"))
            ctx = self.FakeCtx()
            ctx.file_path = md_path
            self.watcher._sync_graph(ctx)

            # Second run — add new node
            md_path.write_text(_md("root: Room\n* Q1\n* Q2\n"))
            self.watcher._sync_graph(ctx)

            loaded = load_graph(str(md_path))
            assert loaded is not None
            assert loaded.get_node("root.2") is not None
            assert loaded.get_node("root.2").content == "Q2"

    def test_merge_preserves_hidden_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"

            # Build initial tree with hidden + children
            md_path.write_text(_md("root: Room\n*[hide] Hidden\n  [*] Secret\n"))
            ctx = self.FakeCtx()
            ctx.file_path = md_path
            self.watcher._sync_graph(ctx)

            # Graph should have the hidden children
            loaded = load_graph(str(md_path))
            assert loaded is not None
            h = loaded.get_node("root.1")
            assert h is not None and h.hidden
            assert len(h.children) == 1
            assert h.children[0].content == "Secret"

    def test_merge_unhide_restores_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"

            # Create graph with hidden node + children
            md_path.write_text(_md("root: Room\n*[hide] Hidden\n  [*] Secret\n* Visible\n"))
            ctx = self.FakeCtx()
            ctx.file_path = md_path
            self.watcher._sync_graph(ctx)

            # Now unhide: remove [hide]
            md_path.write_text(_md("root: Room\n* Hidden\n* Visible\n"))
            self.watcher._sync_graph(ctx)

            # Read back from .graph.json — children should be restored
            loaded = load_graph(str(md_path))
            assert loaded is not None
            h = loaded.get_node("root.1")
            assert h is not None and not h.hidden
            assert len(h.children) == 1
            assert h.children[0].content == "Secret"
