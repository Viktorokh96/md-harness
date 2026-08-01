"""Archive branches — summarize subtree via one-shot LLM, detach to archive/ dir.

Called by the watcher's _sync_graph when a node gets `[archive]` marker.
NOT part of the ControlRoom agent — this is preprocessor logic.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from tree_engine import (
    MindNode,
    MindTree,
    load_graph,
    parse_mindmap,
    save_graph,
    serialize_mindmap,
    replace_block,
)

# ── Archive dir ─────────────────────────────────────────────────────────────


def archive_dir(md_path: str) -> Path:
    return Path(md_path).parent / "archive"


# ── Summarize ───────────────────────────────────────────────────────────────


SUMMARIZE_PROMPT = """Summarize this conversation subtree in ONE short sentence (max 15 words, in the same language).
Focus on: what was asked, what was decided/answered. No bullet points, no markdown.

Subtree:
{subtree}

Summary:"""


def summarize(tree: MindTree, node_id: str) -> str:
    """One-shot LLM call to summarize a subtree."""
    node = tree.get_node(node_id)
    if node is None:
        return ""

    lines: list[str] = []

    def _walk(n: MindNode, indent: int) -> None:
        tag = "[user]" if n.is_user else "[agent]"
        lines.append(f"{'  ' * indent}{tag} {n.content}")
        for child in n.children:
            _walk(child, indent + 1)

    _walk(node, 0)
    subtree_text = "\n".join(lines)

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(subtree=subtree_text)}],
            max_tokens=50,
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        return text.strip() if text and text.strip() else node.content[:80]
    except Exception:
        return node.content[:80]

# ── Archive ─────────────────────────────────────────────────────────────────


def archive_branch(md_path: str, node_id: str, manual_summary: str = "") -> str:
    """Archive a branch: detach subtree to archive/, insert summary marker.

    Returns the summary text.
    """
    tree = load_graph(md_path)
    if tree is None:
        tree = parse_mindmap(Path(md_path).read_text())

    node = tree.get_node(node_id)
    if node is None:
        return f"Error: node {node_id} not found"

    summary = manual_summary or summarize(tree, node_id)
    if not summary:
        summary = node.content[:80]
    # Save metadata BEFORE detach (detach sets node.parent = None)
    parent = node.parent
    author = node.author
    depth_val = node.depth

    subtree = tree.detach_subtree(node_id)
    if subtree is None:
        return f"Error: cannot detach root node"

    # Write archive file with metadata
    ad = archive_dir(md_path)
    ad.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_file = ad / f"{node_id}_{ts}.md"
    archive_file.write_text(
        f"# Archived branch: {node_id}\n\n"
        f"archived_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"summary: {summary}\n"
        f"author: {author}\n\n"
        + serialize_mindmap(subtree)
    )

    # Insert archived marker node under saved parent
    if parent is not None:
        marker = MindNode(
            id=node_id, content=summary, author=author, depth=depth_val,
            archived=True, archive_reason=manual_summary if manual_summary else "",
        )
        marker.parent = parent
        parent.add_child(marker)
        tree._node_index[node_id] = marker

    save_graph(tree, md_path)
    md_text = Path(md_path).read_text()
    Path(md_path).write_text(replace_block(md_text, serialize_mindmap(tree)))

    return summary


# ── Restore ─────────────────────────────────────────────────────────────────


def restore_branch(md_path: str, node_id: str) -> bool:
    """Restore archived branch from archive/<node_id>*.md. Returns True on success."""
    ad = archive_dir(md_path)
    if not ad.exists():
        return False

    candidates = sorted(ad.glob(f"{node_id}_*.md"), reverse=True)
    if not candidates:
        return False

    archive_text = candidates[0].read_text()
    subtree = parse_mindmap(archive_text)

    tree = load_graph(md_path)
    if tree is None:
        tree = parse_mindmap(Path(md_path).read_text())

    marker_node = tree.get_node(node_id)
    if marker_node is None:
        return False

    parent = marker_node.parent
    if parent is None:
        return False

    # Remove marker from parent
    parent.children = [c for c in parent.children if c.id != node_id]
    tree._node_index.pop(node_id, None)

    # Restore author from archive metadata
    for line in archive_text.splitlines():
        if line.startswith("author: "):
            subtree.root.author = line[8:].strip()
            break

    restored_node = tree.attach_subtree(parent.id, subtree)
    restored_node.archived = False
    restored_node.archive_reason = ""

    save_graph(tree, md_path)
    md_text = Path(md_path).read_text()
    Path(md_path).write_text(replace_block(md_text, serialize_mindmap(tree)))

    return True
