"""Mind map tree engine — parser, serializer, renderer for ```agentsmindmap blocks.

Format:
    ```agentsmindmap
    root: Room Name
    * user message                   # user node, visible
    *[hide] summary                  # user node, hidden
      [*] agent reply                # agent node, visible
      [*][hide] summary              # agent node, hidden
        continuation line            # appended to parent node content
    ```

Features:
- Tree structure with stable dot-separated node IDs (root.1, root.1.1, ...)
- [hide] marker: hidden subtrees collapsed to summary; .graph.json stores full tree
- Continuation lines: indented non-tag lines appended to parent node content
- JSON serialization: to_dict/from_dict for .graph.json sidecar
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── Block markers ───────────────────────────────────────────────────────────

BLOCK_START = "```agentsmindmap"
BLOCK_END = "```"

# ── Regex ───────────────────────────────────────────────────────────────────

HIDE_RE = re.compile(r"^(\*|\[\*\])\[hide\]\s")
TAG_RE = re.compile(r"^(\*|\[\*\])\s")

# ── Node ────────────────────────────────────────────────────────────────────


@dataclass
class MindNode:
    """A node in the mind map tree."""

    id: str
    content: str
    author: str  # "user" or "agent"
    depth: int
    hidden: bool = False
    children: list[MindNode] = field(default_factory=list)
    parent: MindNode | None = field(default=None, repr=False)

    @property
    def is_user(self) -> bool:
        return self.author == "user"

    @property
    def is_agent(self) -> bool:
        return self.author == "agent"

    def add_child(self, child: MindNode) -> None:
        child.parent = self
        self.children.append(child)

    def path(self) -> list[str]:
        """Return node IDs from root to this node."""
        ids: list[str] = []
        node: MindNode | None = self
        while node is not None:
            ids.append(node.id)
            node = node.parent
        ids.reverse()
        return ids

    def __repr__(self) -> str:
        prefix = "*" if self.is_user else "[*]"
        h = "[hide]" if self.hidden else ""
        return f"MindNode({self.id}, {prefix}{h} {self.content[:40]})"


# ── Tree ────────────────────────────────────────────────────────────────────


@dataclass
class MindTree:
    """The full mind map tree with parse/serialize/render capabilities."""

    root: MindNode
    _node_index: dict[str, MindNode] = field(default_factory=dict)

    def get_node(self, node_id: str) -> MindNode | None:
        return self._node_index.get(node_id)

    def all_nodes(self) -> list[MindNode]:
        return list(self._node_index.values())

    def find_nodes(self, query: str) -> list[MindNode]:
        """Case-insensitive substring search across all nodes."""
        q = query.lower()
        return [n for n in self._node_index.values() if q in n.content.lower()]

    def add_reply(self, parent_id: str, content: str, author: str) -> MindNode:
        """Add a reply node under `parent_id`. Returns the new node."""
        parent = self._node_index.get(parent_id)
        if parent is None:
            raise KeyError(f"Node not found: {parent_id}")

        child_id = f"{parent_id}.{len(parent.children) + 1}"
        child = MindNode(
            id=child_id, content=content, author=author, depth=parent.depth + 1,
        )
        parent.add_child(child)
        self._node_index[child_id] = child
        return child

    def to_outline(self, *, show_hidden: bool = True) -> str:
        """Serialize tree to indented outline format (for display/LLM).

        Args:
            show_hidden: If True, include hidden nodes marked as 📦.
                         If False, skip hidden subtrees entirely.
        """
        lines: list[str] = []

        def _walk(node: MindNode, indent: int) -> None:
            prefix = "  " * indent
            if node is self.root:
                lines.append(f"📌 [{node.id}] {node.content}")
            else:
                tag = "*" if node.is_user else "[*]"
                marker = "📦" if node.hidden else tag
                lines.append(f"{prefix}{marker} [{node.id}] {node.content}")
            if node.hidden and not show_hidden:
                return  # skip hidden subtree
            for child in node.children:
                _walk(child, indent + 1)

        _walk(self.root, 0)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tree to JSON-compatible dict."""
        def _node_to_dict(node: MindNode) -> dict[str, Any]:
            return {
                "id": node.id,
                "content": node.content,
                "author": node.author,
                "depth": node.depth,
                "hidden": node.hidden,
                "children": [_node_to_dict(c) for c in node.children],
            }
        return _node_to_dict(self.root)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MindTree:
        """Deserialize tree from JSON-compatible dict."""
        def _dict_to_node(d: dict[str, Any]) -> MindNode:
            node = MindNode(
                id=d["id"],
                content=d["content"],
                author=d["author"],
                depth=d["depth"],
                hidden=d.get("hidden", False),
            )
            for cd in d.get("children", []):
                child = _dict_to_node(cd)
                node.add_child(child)
            return node

        root = _dict_to_node(data)
        tree = cls(root=root)
        # Rebuild _node_index
        def _index(node: MindNode) -> None:
            tree._node_index[node.id] = node
            for child in node.children:
                _index(child)
        _index(root)
        return tree


# ── Parser ──────────────────────────────────────────────────────────────────


def parse_mindmap(text: str) -> MindTree:
    """Parse a ```agentsmindmap block from markdown text.

    Returns a MindTree. Raises ValueError on malformed input.
    """
    start = text.find(BLOCK_START)
    if start == -1:
        raise ValueError(f"No {BLOCK_START} block found in text")

    start = text.find("\n", start) + 1  # skip the ``` line
    end = text.find(BLOCK_END, start)
    if end == -1:
        raise ValueError(f"Unclosed {BLOCK_START} block")

    block = text[start:end].strip()
    return _parse_block(block)


def _parse_block(block: str) -> MindTree:
    """Parse the content inside a ```agentsmindmap block."""
    lines = block.splitlines()
    if not lines:
        raise ValueError("Empty mindmap block")

    # First line must be root
    root_line = lines[0].strip()
    if not root_line.startswith("root:"):
        raise ValueError(f"First line must be 'root: <name>', got: {root_line}")

    root_name = root_line[5:].strip()
    root = MindNode(id="root", content=root_name, author="system", depth=0)
    tree = MindTree(root=root)
    tree._node_index["root"] = root

    # Stack tracks parents at each depth
    stack: dict[int, MindNode] = {0: root}
    counter: dict[int, int] = {}  # depth → child counter for ID generation

    for line in lines[1:]:
        if not line.strip():
            continue

        # Determine depth from leading whitespace (2 spaces = 1 level)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        depth = indent // 2 + 1  # root is 0, first children start at 1
        if indent % 2 != 0:
            raise ValueError(f"Indent must be multiple of 2 spaces: {line!r}")

        # Check for [hide] marker: *[hide] or [*][hide]
        hide_match = HIDE_RE.match(stripped)
        if hide_match:
            tag = hide_match.group(1)
            author = "agent" if tag == "[*]" else "user"
            content = stripped[hide_match.end():].strip()
            hidden = True
        elif TAG_RE.match(stripped):
            tag_match = TAG_RE.match(stripped)
            assert tag_match is not None
            tag = tag_match.group(1)
            author = "agent" if tag == "[*]" else "user"
            content = stripped[tag_match.end():].strip()
            hidden = False
        else:
            # Continuation line — append to parent at depth-1
            parent_node = stack.get(depth - 1)
            if parent_node is None:
                raise ValueError(
                    f"No parent at depth {depth - 1} for continuation: {line!r}"
                )
            parent_node.content += "\n" + stripped
            continue

        parent = stack.get(depth - 1)
        if parent is None:
            raise ValueError(
                f"No parent at depth {depth - 1} for line (depth {depth}): {line!r}"
            )

        # Generate ID
        counter.setdefault(depth, 0)
        counter[depth] += 1
        node_id = f"{parent.id}.{counter[depth]}"

        node = MindNode(
            id=node_id, content=content, author=author, depth=depth, hidden=hidden,
        )
        parent.add_child(node)
        tree._node_index[node_id] = node

        # Update stack: this node becomes the parent for depth+1
        stack[depth] = node
        # Clear deeper stack entries (sibling at this depth replaces old subtree)
        for d in list(stack.keys()):
            if d > depth:
                del stack[d]

    return tree


# ── Serializer ──────────────────────────────────────────────────────────────


def serialize_mindmap(tree: MindTree) -> str:
    """Serialize a MindTree back to a ```agentsmindmap markdown block.

    Hidden subtrees are collapsed to a single summary line.
    Multi-line node content gets continuation indent.
    """
    def _walk(node: MindNode, depth: int, lines: list[str]) -> None:
        if node is tree.root:
            lines.append(f"root: {node.content}")
        else:
            indent = "  " * (depth - 1)
            tag = "*" if node.is_user else "[*]"
            if node.hidden:
                tag = f"{tag}[hide]"
            content_lines = node.content.split("\n")
            lines.append(f"{indent}{tag} {content_lines[0]}")
            # Continuation lines at child indent (depth * 2 spaces)
            cont_indent = "  " * depth
            for cont in content_lines[1:]:
                lines.append(f"{cont_indent}{cont}")

        if node.hidden:
            return  # don't render children of hidden nodes
        for child in node.children:
            _walk(child, depth + 1, lines)

    lines: list[str] = []
    _walk(tree.root, 0, lines)
    body = "\n".join(lines)
    return f"{BLOCK_START}\n{body}\n{BLOCK_END}\n"


def merge_md_into_graph(graph: MindTree, md_tree: MindTree) -> MindTree:
    """Merge changes from .md (visible tree) into full graph.

    .md is authoritative for content and hidden flags of visible nodes.
    .graph provides children of hidden nodes and any nodes not present in .md.
    """
    # Rebuild _node_index for both
    graph._node_index = {}
    _reindex(graph.root, graph._node_index)
    md_tree._node_index = {}
    _reindex(md_tree.root, md_tree._node_index)

    def _merge(graph_node: MindNode, md_node: MindNode | None) -> MindNode:
        """Recursively merge a graph node with an md node.

        Returns the merged node (modified in-place on graph_node).
        """
        if md_node is not None:
            was_hidden = graph_node.hidden  # save BEFORE update

            # Update content and hidden flag from .md
            graph_node.content = md_node.content
            graph_node.hidden = md_node.hidden

            # Build index of graph children by ID
            graph_children = {c.id: c for c in graph_node.children}
            new_children: list[MindNode] = []

            for md_child in md_node.children:
                if md_child.id in graph_children:
                    merged = _merge(graph_children[md_child.id], md_child)
                else:
                    merged = _copy_node(md_child)
                new_children.append(merged)
                merged.parent = graph_node

            # Restore children from graph when hidden state changed
            if was_hidden and not graph_node.hidden:
                # Unhide: restore graph children if .md had none
                if not new_children and graph_node.children:
                    new_children = [_copy_node(c) for c in graph_node.children]
                    for c in new_children:
                        c.parent = graph_node
            elif graph_node.hidden:
                # Stays hidden or newly hidden: preserve graph children
                if not new_children and graph_node.children:
                    new_children = [_copy_node(c) for c in graph_node.children]
                    for c in new_children:
                        c.parent = graph_node

            graph_node.children = new_children
        # If md_node is None: keep graph_node as-is

        return graph_node

    _merge(graph.root, md_tree.root)

    # Rebuild index
    graph._node_index = {}
    _reindex(graph.root, graph._node_index)
    return graph


def _copy_node(node: MindNode) -> MindNode:
    """Deep copy a MindNode without parent references."""
    new = MindNode(
        id=node.id,
        content=node.content,
        author=node.author,
        depth=node.depth,
        hidden=node.hidden,
    )
    for child in node.children:
        c = _copy_node(child)
        new.add_child(c)
    return new


def _reindex(node: MindNode, index: dict[str, MindNode]) -> None:
    """Recursively populate node index."""
    index[node.id] = node
    for child in node.children:
        _reindex(child, index)


def diff_trees(old: MindTree, new: MindTree) -> tuple[bool, str]:
    """Compare two trees structurally. Returns (has_content_change, diff_text).

    Content change = new nodes or changed content.
    Toggling hidden flag or unhiding is NOT a content change — no LLM needed.
    """
    new_nodes: list[str] = []
    changed_nodes: list[str] = []
    unhidden_nodes: list[str] = []
    hidden_nodes: list[str] = []

    def _compare(old_node: MindNode | None, new_node: MindNode | None, path: str) -> None:
        if old_node is None and new_node is not None:
            tag = "*" if new_node.is_user else "[*]"
            h = "[hide]" if new_node.hidden else ""
            new_nodes.append(f"  {tag}{h} {new_node.content}  [id: {new_node.id}]")
            for child in new_node.children:
                _compare(None, child, f"{path}/{new_node.id}")
            return
        if new_node is None or old_node is None:
            return

        if old_node.content != new_node.content:
            tag = "*" if new_node.is_user else "[*]"
            changed_nodes.append(
                f"  [{new_node.id}] changed: {old_node.content[:60]!r} → {new_node.content[:60]!r}"
            )

        if old_node.hidden and not new_node.hidden:
            unhidden_nodes.append(f"  [{new_node.id}] unhidden: {new_node.content[:60]}")
        elif not old_node.hidden and new_node.hidden:
            hidden_nodes.append(f"  [{new_node.id}] hidden: {new_node.content[:60]}")

        old_kids = {c.id: c for c in old_node.children}
        new_kids = {c.id: c for c in new_node.children}
        for child_id in new_kids:
            _compare(old_kids.get(child_id), new_kids[child_id], f"{path}/{child_id}")

    _compare(old.root, new.root, "")

    has_content_change = bool(new_nodes or changed_nodes)

    parts: list[str] = []
    if new_nodes:
        parts.append("New nodes:\n" + "\n".join(new_nodes))
    if changed_nodes:
        parts.append("Content changed:\n" + "\n".join(changed_nodes))
    if unhidden_nodes:
        parts.append("Unhidden:\n" + "\n".join(unhidden_nodes))
    if hidden_nodes:
        parts.append("Hidden:\n" + "\n".join(hidden_nodes))

    diff_text = "\n".join(parts) if parts else "(no structural changes)"
    return has_content_change, diff_text


# ── Renderer ────────────────────────────────────────────────────────────────
def render_tree(tree: MindTree) -> str:
    """Render tree as a compact ASCII outline for display."""
    return tree.to_outline()


def render_mermaid(tree: MindTree) -> str:
    """Render tree as a Mermaid graph (mindmap type)."""
    lines = ["```mermaid", "mindmap"]

    def _walk(node: MindNode, indent: str) -> None:
        if node is tree.root:
            lines.append(f"  {indent}{_sanitize(node.content)}")
        else:
            tag = "👤" if node.is_user else "🤖"
            marker = "📦" if node.hidden else tag
            lines.append(f"  {indent}{marker} {_sanitize(node.content)}")
        if node.hidden:
            return
        for child in node.children:
            _walk(child, indent + "  ")

    _walk(tree.root, "")
    lines.append("```")
    return "\n".join(lines)


def _sanitize(text: str) -> str:
    """Remove Mermaid-breaking characters from a single line."""
    return text.replace("\n", " ").replace('"', "'").replace("(", "[").replace(")", "]")[:80]


# ── Block extraction from markdown ──────────────────────────────────────────


def extract_block(text: str) -> str | None:
    """Extract the ```agentsmindmap block from markdown text. Returns None if not found."""
    start = text.find(BLOCK_START)
    if start == -1:
        return None
    start = text.find("\n", start) + 1
    end = text.find(BLOCK_END, start)
    if end == -1:
        return None
    return text[start:end].strip()


def replace_block(text: str, new_block: str) -> str:
    """Replace the ```agentsmindmap block in markdown text with new_block."""
    start = text.find(BLOCK_START)
    if start == -1:
        if text and not text.endswith("\n"):
            text += "\n"
        return text + "\n" + new_block

    # Find end of the block — search AFTER the opening line
    search_from = text.find("\n", start) + 1
    end = text.find(f"\n{BLOCK_END}", search_from)
    if end == -1:
        # Block at end of file without proper closing
        end = len(text)
    else:
        end += 1 + len(BLOCK_END)  # include the closing ```

    return text[:start] + new_block.rstrip("\n") + "\n" + text[end:]


# ── JSON sidecar ────────────────────────────────────────────────────────────


def graph_path(md_path: str) -> str:
    """Compute sidecar .graph.json path for a given .md file."""
    return md_path.replace(".md", ".md_graph.json")


def load_graph(md_path: str) -> MindTree | None:
    """Load full graph from .graph.json sidecar. Returns None if missing."""
    gp = graph_path(md_path)
    try:
        with open(gp) as f:
            data = json.load(f)
        return MindTree.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_graph(tree: MindTree, md_path: str) -> None:
    """Save full graph to .graph.json sidecar."""
    gp = graph_path(md_path)
    data = tree.to_dict()
    with open(gp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
