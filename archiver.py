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
    """archive/ directory next to the .md file."""
    return Path(md_path).parent / "archive"


# ── Summarize ───────────────────────────────────────────────────────────────


SUMMARIZE_PROMPT = """Summarize this conversation subtree in ONE short sentence (max 15 words, in the same language).
Focus on: what was asked, what was decided/answered. No bullet points, no markdown.

Subtree:
{subtree}

Summary:"""


def summarize(tree: MindTree, node_id: str) -> str:
    """One-shot LLM call to summarize a subtree. Returns short summary string."""
    node = tree.get_node(node_id)
    if node is None:
        return ""

    # Build a plain-text version of the subtree
    lines: list[str] = []

    def _walk(n: MindNode, indent: int) -> None:
        tag = "[user]" if n.is_user else "[agent]"
        lines.append(f"{'  ' * indent}{tag} {n.content}")
        for child in n.children:
            _walk(child, indent + 1)

    _walk(node, 0)
    subtree_text = "\n".join(lines)

    prompt = SUMMARIZE_PROMPT.format(subtree=subtree_text)

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip() if resp.choices else ""
    except Exception:
        # Fallback: use node's own content as summary
        return node.content[:80]


# ── Archive / Restore ───────────────────────────────────────────────────────


def archive_branch(md_path: str, node_id: str, manual_summary: str = "") -> str:
    """Archive a branch: detach subtree, write to archive/, summarize, update graph.

    Returns the summary text that replaces the subtree.
    """
    tree = load_graph(md_path)
    if tree is None:
        tree = parse_mindmap(Path(md_path).read_text())

    node = tree.get_node(node_id)
    if node is None:
        return f"Error: node {node_id} not found"

    # Generate summary
    summary = manual_summary or summarize(tree, node_id)

    # Detach subtree from full graph
    subtree = tree.detach_subtree(node_id)
    if subtree is None:
        return f"Error: cannot detach root node"

    # Write archive file
    ad = archive_dir(md_path)
    ad.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_file = ad / f"{node_id}_{ts}.md"
    archive_content = f"# Archived branch: {node_id}\n\n"
    archive_content += f"archived_at: {datetime.now(timezone.utc).isoformat()}\n"
    archive_content += f"summary: {summary}\n\n"
    archive_content += serialize_mindmap(subtree)
    archive_file.write_text(archive_content)

    # Replace node in tree with archived marker
    archived_node = MindNode(
        id=node_id,
        content=summary,
        author=node.author,
        depth=node.depth,
        archived=True,
        archive_reason=manual_summary if manual_summary else "",
        children=[],
    )
    if node.parent:
        # Find position and replace
        for i, c in enumerate(node.parent.children):
            if c.id == node_id:
                node.parent.children[i] = archived_node
                archived_node.parent = node.parent
                break
        tree._node_index[node_id] = archived_node

    # Save updated graph
    save_graph(tree, md_path)

    # Update .md file
    md_text = Path(md_path).read_text()
    new_content = replace_block(md_text, serialize_mindmap(tree))
    Path(md_path).write_text(new_content)

    return summary


def restore_branch(md_path: str, node_id: str) -> bool:
    """Restore an archived branch from archive/<node_id>*.md back into the graph.

    Returns True on success, False if no archive file found.
    """
    ad = archive_dir(md_path)
    if not ad.exists():
        return False

    # Find the latest archive file for this node_id
    candidates = sorted(ad.glob(f"{node_id}_*.md"), reverse=True)
    if not candidates:
        return False

    archive_file = candidates[0]
    archive_text = archive_file.read_text()

    # Parse archived tree
    subtree = parse_mindmap(archive_text)

    # Detach the archived marker node from main tree
    tree = load_graph(md_path)
    if tree is None:
        tree = parse_mindmap(Path(md_path).read_text())

    marker_node = tree.get_node(node_id)
    if marker_node is None:
        return False

    parent = marker_node.parent
    if parent is None:
        return False

    # Remove marker from parent's children and index
    parent.children = [c for c in parent.children if c.id != node_id]
    tree._node_index.pop(node_id, None)

    # Attach restored subtree
    restored_node = tree.attach_subtree(parent.id, subtree)

    # Remove archived flag from restored node
    restored_node.archived = False
    restored_node.archive_reason = ""

    save_graph(tree, md_path)
    md_text = Path(md_path).read_text()
    new_content = replace_block(md_text, serialize_mindmap(tree))
    Path(md_path).write_text(new_content)

    return True
