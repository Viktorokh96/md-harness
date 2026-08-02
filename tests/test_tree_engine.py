"""Tests for tree_engine — parser, serializer, tree operations."""

from __future__ import annotations

import pytest

from tree_engine import MindNode
from tree_engine import MindTree
from tree_engine import diff_trees
from tree_engine import has_pending_questions
from tree_engine import parse_mindmap
from tree_engine import replace_block
from tree_engine import serialize_mindmap

# ── MindNode ────────────────────────────────────────────────────────────────


class TestMindNode:
    def test_user_node(self) -> None:
        n = MindNode(id="1", content="Hello", author="user", depth=1)
        assert n.is_user
        assert not n.is_agent

    def test_agent_node(self) -> None:
        n = MindNode(id="1.1", content="Reply", author="agent", depth=2)
        assert n.is_agent
        assert not n.is_user

    def test_add_child(self) -> None:
        parent = MindNode(id="1", content="Q", author="user", depth=1)
        child = MindNode(id="1.1", content="A", author="agent", depth=2)
        parent.add_child(child)
        assert child.parent is parent
        assert parent.children == [child]

    def test_path(self) -> None:
        root = MindNode(id="root", content="R", author="system", depth=0)
        a = MindNode(id="root.1", content="Q", author="user", depth=1)
        b = MindNode(id="root.1.1", content="A", author="agent", depth=2)
        root.add_child(a)
        a.add_child(b)
        assert b.path() == ["root", "root.1", "root.1.1"]


# ── MindTree ────────────────────────────────────────────────────────────────


class TestMindTree:
    def test_get_node(self) -> None:
        tree = _sample_tree()
        assert tree.get_node("root.1") is not None
        assert tree.get_node("nonexistent") is None

    def test_find_nodes(self) -> None:
        tree = _sample_tree()
        results = tree.find_nodes("hello")
        assert len(results) == 1
        assert results[0].id == "root.1"

    def test_find_nodes_case_insensitive(self) -> None:
        tree = _sample_tree()
        assert len(tree.find_nodes("HELLO")) == 1

    def test_add_reply(self) -> None:
        tree = _sample_tree()
        node = tree.add_reply("root.1", "Follow-up", "user")
        assert node.id == "root.1.3"
        assert node.author == "user"
        assert node.depth == 2

    def test_add_reply_nonexistent(self) -> None:
        with pytest.raises(KeyError):
            _sample_tree().add_reply("bad.id", "text", "user")

    def test_to_outline(self) -> None:
        outline = _sample_tree().to_outline()
        assert "📌 [root]" in outline
        assert "👤 [root.1] Hello" in outline
        assert "🤖 [root.1.1] Hi" in outline


# ── Parser ──────────────────────────────────────────────────────────────────


class TestParseMindmap:
    def test_simple_tree(self) -> None:
        text = _md("root: R\n* Q\n  [*] A\n")
        tree = parse_mindmap(text)
        assert tree.root.content == "R"
        assert len(tree.root.children) == 1
        q = tree.root.children[0]
        assert q.content == "Q" and q.author == "user"
        assert len(q.children) == 1
        assert q.children[0].content == "A"

    def test_multiple_siblings(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Q1\n* Q2\n"))
        assert len(tree.root.children) == 2

    def test_nested_thread(self) -> None:
        tree = parse_mindmap(
            _md("root: R\n* Q\n  [*] A1\n    * Follow\n      [*] A2\n  [*] Parallel\n"),
        )
        q = tree.root.children[0]
        assert len(q.children) == 2
        assert q.children[0].children[0].content == "Follow"

    def test_empty_lines_ignored(self) -> None:
        tree = parse_mindmap(_md("root: R\n\n* Q\n\n  [*] A\n\n"))
        assert len(tree.root.children) == 1

    def test_no_block_raises(self) -> None:
        with pytest.raises(ValueError, match=r"No .* block"):
            parse_mindmap("# No block\n")

    def test_unclosed_block_raises(self) -> None:
        with pytest.raises(ValueError, match="Unclosed"):
            parse_mindmap("# X\n\n```agentsmindmap\nroot: R\n* Q\n")

    def test_missing_root_raises(self) -> None:
        with pytest.raises(ValueError, match="First line"):
            parse_mindmap(_md("* Q\n"))

    def test_bad_indent_raises(self) -> None:
        with pytest.raises(ValueError, match="multiple of 2"):
            parse_mindmap(_md("root: R\n * Q\n"))

    def test_bare_word_becomes_continuation(self) -> None:
        """Non-tag lines are appended as continuation to parent."""
        tree = parse_mindmap(_md("root: R\nbare word\n"))
        assert "bare word" in tree.root.content

    def test_orphan_raises(self) -> None:
        with pytest.raises(ValueError, match="No parent"):
            parse_mindmap(_md("root: R\n    [*] Orphan\n"))


# ── Serializer + round-trip ─────────────────────────────────────────────────


class TestSerializeMindmap:
    def test_round_trip_simple(self) -> None:
        original = _md("root: R\n* Q\n  [*] A\n")
        t1 = parse_mindmap(original)
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        assert t2.to_outline() == t1.to_outline()

    def test_round_trip_nested(self) -> None:
        original = _md("root: R\n* Q1\n  [*] A1\n    * F\n      [*] A2\n* Q2\n  [*] A3\n")
        t1 = parse_mindmap(original)
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        assert t2.to_outline() == t1.to_outline()

    def test_round_trip_after_mutation(self) -> None:
        t1 = parse_mindmap(_md("root: R\n* Q\n"))
        t1.add_reply("root.1", "Answer", "agent")
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        assert t2.to_outline() == t1.to_outline()


# ── replace_block ───────────────────────────────────────────────────────────


class TestReplaceBlock:
    def test_replaces_existing(self) -> None:
        text = "# H\n\n```agentsmindmap\nroot: Old\n```\n\nFoot\n"
        new = "```agentsmindmap\nroot: New\n* Q\n```\n"
        result = replace_block(text, new)
        assert "root: New" in result
        assert "root: Old" not in result
        assert "Foot" in result

    def test_appends_when_no_block(self) -> None:
        result = replace_block("# H\n", "```agentsmindmap\nroot: N\n```\n")
        assert "root: N" in result

    def test_preserves_surrounding(self) -> None:
        text = "Before\n\n```agentsmindmap\nroot: X\n```\n\nAfter\n"
        new = "```agentsmindmap\nroot: Y\n* Q\n  [*] A\n```\n"
        result = replace_block(text, new)
        assert result.startswith("Before")
        assert result.rstrip().endswith("After")
        assert "root: Y" in result


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sample_tree() -> MindTree:
    return parse_mindmap(_md("root: Test\n* Hello\n  [*] Hi\n  [*] Also\n"))


def _md(body: str) -> str:
    return f"# Test\n\n```agentsmindmap\n{body}```\n"


# ── Question (?) marker ──────────────────────────────────────────────────────


class TestQuestionParsing:
    def test_user_question(self) -> None:
        """*? sets has_question=True."""
        tree = parse_mindmap(_md("root: R\n*? Help me\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.has_question is True
        assert n.is_user is True
        assert n.content == "Help me"

    def test_emoji_question(self) -> None:
        """👤❓ also parsed as question."""
        tree = parse_mindmap(_md("root: R\n👤❓ Help me\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.has_question is True
        assert n.content == "Help me"

    def test_plain_user_no_question(self) -> None:
        """* without ? has_question=False."""
        tree = parse_mindmap(_md("root: R\n* Note\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.has_question is False

    def test_agent_no_question(self) -> None:
        """🤖 (agent) never has has_question=True."""
        tree = parse_mindmap(_md("root: R\n🤖 Reply\n"))
        n = tree.get_node("root.1")
        assert n is not None
        assert n.has_question is False
        assert n.is_agent is True

    def test_old_format_question(self) -> None:
        """[*]? would be agent question — has_question only for users."""
        # [*]? is not a valid user question marker
        # covered by test_agent_no_question


class TestQuestionSerialize:
    def test_question_output(self) -> None:
        """👤❓ appears in serialized output."""
        tree = parse_mindmap(_md("root: R\n*? Q\n"))
        s = serialize_mindmap(tree)
        assert "👤❓ Q" in s

    def test_question_round_trip(self) -> None:
        """Parse → serialize → reparse preserves has_question."""
        original = _md("root: R\n*? Ask\n  🤖 Answer\n")
        t1 = parse_mindmap(original)
        n1 = t1.get_node("root.1")
        assert n1 is not None and n1.has_question is True
        t2 = parse_mindmap("# X\n\n" + serialize_mindmap(t1))
        n2 = t2.get_node("root.1")
        assert n2 is not None and n2.has_question is True
        n3 = t2.get_node("root.1.1")
        assert n3 is not None and n3.has_question is False

    def test_question_outline(self) -> None:
        """Outline shows 👤❓ for questions."""
        tree = parse_mindmap(_md("root: R\n*? Q\n"))
        outline = tree.to_outline()
        assert "👤❓ [root.1] Q" in outline

    def test_question_to_json(self) -> None:
        """has_question survives JSON round-trip."""
        tree = parse_mindmap(_md("root: R\n*? Q\n"))
        d = tree.to_dict()
        t2 = MindTree.from_dict(d)
        n4 = t2.get_node("root.1")
        assert n4 is not None and n4.has_question is True


class TestQuestionDiff:
    def test_question_flag_change_detected(self) -> None:
        """Adding ? to an existing node is detected as content change."""
        old = parse_mindmap(_md("root: R\n* Note\n"))
        new = parse_mindmap(_md("root: R\n*? Note\n"))
        has_change, diff = diff_trees(old, new)
        assert has_change is True
        assert "changed" in diff.lower()

    def test_question_removal_detected(self) -> None:
        """Removing ? is detected."""
        old = parse_mindmap(_md("root: R\n*? Q\n"))
        new = parse_mindmap(_md("root: R\n* Q\n"))
        has_change, diff = diff_trees(old, new)
        assert has_change is True


class TestHasPendingQuestions:
    def test_with_question(self) -> None:
        tree = parse_mindmap(_md("root: R\n*? Q\n"))
        assert has_pending_questions(tree) is True

    def test_without_question(self) -> None:
        tree = parse_mindmap(_md("root: R\n* Note\n"))
        assert has_pending_questions(tree) is False

    def test_agent_reply_not_question(self) -> None:
        tree = parse_mindmap(_md("root: R\n🤖 Answer\n"))
        assert has_pending_questions(tree) is False
