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
    diff_trees,
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


class TestDiffTrees:
    """Tests for diff_trees — tree-based structural comparison."""

    def test_hide_toggle_no_content_change(self) -> None:
        old = parse_mindmap(_md("root: R\n* Q\n"))
        new = parse_mindmap(_md("root: R\n*[hide] Q\n"))
        has_change, diff_text = diff_trees(old, new)
        assert not has_change, f"Hide toggle should not be content change, got: {diff_text}"
        assert "Hidden:" in diff_text or "hidden" in diff_text.lower()

    def test_unhide_no_content_change(self) -> None:
        old = parse_mindmap(_md("root: R\n*[hide] Q\n"))
        new = parse_mindmap(_md("root: R\n* Q\n"))
        has_change, _ = diff_trees(old, new)
        assert not has_change  # unhiding alone = no LLM needed (children in .graph.json)

    def test_new_node_is_content_change(self) -> None:
        old = parse_mindmap(_md("root: R\n* Q1\n"))
        new = parse_mindmap(_md("root: R\n* Q1\n* Q2\n"))
        has_change, diff_text = diff_trees(old, new)
        assert has_change
        assert "New nodes" in diff_text

    def test_content_change_detected(self) -> None:
        old = parse_mindmap(_md("root: R\n* Hello\n"))
        new = parse_mindmap(_md("root: R\n* World\n"))
        has_change, diff_text = diff_trees(old, new)
        assert has_change
        assert "changed" in diff_text.lower() or "Content" in diff_text

    def test_identical_no_change(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Q\n"))
        has_change, diff_text = diff_trees(tree, tree)
        assert not has_change

    def test_agent_hide_toggle(self) -> None:
        old = parse_mindmap(_md("root: R\n[*][hide] Reply\n"))
        new = parse_mindmap(_md("root: R\n[*] Reply\n"))
        has_change, _ = diff_trees(old, new)
        assert not has_change

    def test_hide_with_children_no_content_change(self) -> None:
        old = parse_mindmap(_md("root: R\n* Q\n  [*] A\n"))
        new = parse_mindmap(_md("root: R\n*[hide] Q\n  [*] A\n"))
        has_change, _ = diff_trees(old, new)
        assert not has_change  # only hidden flag changed, children still there in tree


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



# ── Archive parsing ─────────────────────────────────────────────────────────


class TestArchiveParsing:
    def test_archive_no_reason(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[archive] Old topic\n"))
        n = tree.get_node("root.1")
        assert n is not None and n.archived
        assert n.archive_reason == ""

    def test_archive_with_reason(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[archive: outdated] Old\n"))
        n = tree.get_node("root.1")
        assert n is not None and n.archived
        assert n.archive_reason == "outdated"

    def test_agent_archive(self) -> None:
        tree = parse_mindmap(_md("root: R\n[*][archive] Reply\n"))
        n = tree.get_node("root.1")
        assert n is not None and n.archived and n.is_agent

    def test_serialize_archive(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[archive] Old\n  [*] Reply\n"))
        s = serialize_mindmap(tree)
        assert "*[archive] Old" in s
        assert "[*] Reply" not in s

    def test_outline_archive(self) -> None:
        tree = parse_mindmap(_md("root: R\n*[archive] Old\n"))
        assert "🗄" in tree.to_outline()


# ── Detach / Attach ─────────────────────────────────────────────────────────


class TestDetachAttach:
    def test_detach(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Topic\n  [*] R1\n  [*] R2\n"))
        sub = tree.detach_subtree("root.1")
        assert sub is not None
        assert len(sub.all_nodes()) == 3
        assert tree.get_node("root.1") is None

    def test_attach(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Parent\n"))
        sr = MindNode(id="x", content="Child", author="agent", depth=1)
        subtree = MindTree(root=sr)
        subtree._node_index["x"] = sr
        tree.attach_subtree("root.1", subtree)
        assert tree.get_node("root.1.1") is not None

    def test_detach_attach_roundtrip(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Topic\n  [*] Reply\n"))
        sub = tree.detach_subtree("root.1")
        assert sub is not None
        tree.attach_subtree("root", sub)
        assert tree.get_node("root.1.1") is not None
        assert tree.get_node("root.1.1").content == "Reply"


# ── Renderer tests ──────────────────────────────────────────────────────────


class TestRenderMermaid:
    def test_basic(self) -> None:
        from tree_engine import render_mermaid
        tree = parse_mindmap(_md("root: R\n* Q\n  [*] A\n"))
        result = render_mermaid(tree)
        assert "```mermaid" in result
        assert "mindmap" in result
        assert "Q" in result
        assert "A" in result

    def test_hidden_nodes(self) -> None:
        from tree_engine import render_mermaid
        tree = parse_mindmap(_md("root: R\n*[hide] H\n  [*] Secret\n"))
        result = render_mermaid(tree)
        assert "📦" in result or "H" in result
        assert "Secret" not in result  # hidden children excluded

    def test_archived_nodes(self) -> None:
        from tree_engine import render_mermaid
        tree = parse_mindmap(_md("root: R\n*[archive] Old\n  [*] Gone\n"))
        result = render_mermaid(tree)
        assert "🗄" in result
        assert "Gone" not in result  # archived children excluded


class TestExtractBlock:
    def test_extracts(self) -> None:
        from tree_engine import extract_block
        text = "# H\n\n```agentsmindmap\nroot: R\n* Q\n```\nfooter\n"
        block = extract_block(text)
        assert block is not None
        assert "root: R" in block
        assert "* Q" in block

    def test_no_block(self) -> None:
        from tree_engine import extract_block
        assert extract_block("# Just text\n") is None

    def test_unclosed(self) -> None:
        from tree_engine import extract_block
        assert extract_block("# H\n```agentsmindmap\nroot: R\n") is None


# ── E2E hide cycle ──────────────────────────────────────────────────────────


class TestHideCycleE2E:
    """Simulate the full watcher flow: message → hide → unhide."""

    def test_hide_preserves_children_in_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"

            # Simulate: agent responded, graph has child
            full_tree = parse_mindmap(_md("root: R\n* Q\n  [*] A\n"))
            save_graph(full_tree, str(md_path))
            md_path.write_text(_md("root: R\n* Q\n  [*] A\n"))

            # User adds [hide]
            md_path.write_text(_md("root: R\n*[hide] Q\n"))

            # Load graphs
            old = load_graph(str(md_path))
            assert old is not None

            # Sync (merge .md into .graph)
            md_tree = parse_mindmap(md_path.read_text())
            merged = merge_md_into_graph(old, md_tree)
            save_graph(merged, str(md_path))

            new = load_graph(str(md_path))
            assert new is not None

            # diff_trees should see only hidden flag change — no content change
            has_change, diff = diff_trees(old, new)
            assert not has_change, f"Hide should NOT be content change, got: {diff}"
            assert new.get_node("root.1") is not None
            assert new.get_node("root.1").hidden
            # Child should be preserved
            assert new.get_node("root.1.1") is not None
            assert new.get_node("root.1.1").content == "A"

    def test_unhide_restores_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"

            # Start with hidden node + child in graph
            full_tree = parse_mindmap(_md("root: R\n*[hide] Q\n  [*] A\n"))
            save_graph(full_tree, str(md_path))
            md_path.write_text(_md("root: R\n*[hide] Q\n"))

            old = load_graph(str(md_path))
            assert old is not None

            # User removes [hide]
            md_path.write_text(_md("root: R\n* Q\n"))
            md_tree = parse_mindmap(md_path.read_text())
            merged = merge_md_into_graph(old, md_tree)
            save_graph(merged, str(md_path))

            new = load_graph(str(md_path))
            assert new is not None
            assert not new.get_node("root.1").hidden
            assert new.get_node("root.1.1") is not None
            assert new.get_node("root.1.1").content == "A"


# ── batch_reply ─────────────────────────────────────────────────────────────


class TestBatchReply:
    def test_creates_multiple_branches(self) -> None:
        import tempfile, os
        from tree_engine import MindTree, MindNode
        with tempfile.TemporaryDirectory() as tmp:
            root = MindNode(id="root", content="R", author="system", depth=0)
            tree = MindTree(root=root)
            tree._node_index["root"] = root
            q = tree.add_reply("root", "Q", "user")

            replies = ["Idea 1", "Idea 2", "Idea 3"]
            ids = []
            for text in replies:
                child = tree.add_reply(q.id, text, "agent")
                ids.append(child.id)

            assert len(q.children) == 3
            assert q.children[0].content == "Idea 1"
            assert q.children[1].content == "Idea 2"
            assert q.children[2].content == "Idea 3"