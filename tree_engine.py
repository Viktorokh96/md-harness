"""Mind map tree engine — parser, serializer, renderer for ```agentsmindmap blocks.

Format inside the block:
    root: Room Name
    * user message
      [*] agent reply
        * user follow-up
          [*] agent answer
      [*] parallel agent thought
    * another topic

Rules:
- 2-space indent per nesting level
- `* ` prefix = user node
- `[*] ` prefix = agent node
- Each node gets a stable ID for tool-based manipulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Block markers ───────────────────────────────────────────────────────────

BLOCK_START = "```agentsmindmap"
BLOCK_END = "```"

# ── Node ────────────────────────────────────────────────────────────────────


@dataclass
class MindNode:
    """A node in the mind map tree."""

    id: str
    content: str
    author: str  # "user" or "agent"
    depth: int
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
        return f"MindNode({self.id}, {prefix} {self.content[:40]})"


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
            id=child_id, content=content, author=author, depth=parent.depth + 1
        )
        parent.add_child(child)
        self._node_index[child_id] = child
        return child

    def to_outline(self) -> str:
        """Serialize tree to indented outline format (for display/LLM)."""
        lines: list[str] = []

        def _walk(node: MindNode, indent: int) -> None:
            prefix = "  " * indent
            if node is self.root:
                lines.append(f"📌 [{node.id}] {node.content}")
            else:
                tag = "*" if node.is_user else "[*]"
                lines.append(f"{prefix}{tag} [{node.id}] {node.content}")
            for child in node.children:
                _walk(child, indent + 1)

        _walk(self.root, 0)
        return "\n".join(lines)

        def _walk(node: MindNode, indent: int) -> None:
            prefix = "  " * indent
            tag = "*" if node.is_user else "[*]"
            lines.append(f"{prefix}{tag} [{node.id}] {node.content}")
            for child in node.children:
                _walk(child, indent + 1)

        _walk(self.root, 0)
        return "\n".join(lines)


# ── Parser ──────────────────────────────────────────────────────────────────


def parse_mindmap(text: str) -> MindTree:
    """Parse a ```agentsmindmap block from markdown text.

    Returns a MindTree. Raises ValueError on malformed input.
    """
    # Extract block
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

        # Determine author and content
        if stripped.startswith("[*] "):
            author = "agent"
            content = stripped[4:].strip()
        elif stripped.startswith("* "):
            author = "user"
            content = stripped[2:].strip()
        else:
            raise ValueError(f"Line must start with '* ' or '[*] ': {line!r}")

        parent = stack.get(depth - 1)
        if parent is None:
            raise ValueError(
                f"No parent at depth {depth - 1} for line (depth {depth}): {line!r}"
            )

        # Generate ID
        counter.setdefault(depth, 0)
        counter[depth] += 1
        node_id_parts = [parent.id, str(counter[depth])]
        node_id = ".".join(node_id_parts)

        node = MindNode(id=node_id, content=content, author=author, depth=depth)
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
    """Serialize a MindTree back to a ```agentsmindmap markdown block."""

    def _walk(node: MindNode, depth: int, lines: list[str]) -> None:
        if node is tree.root:
            lines.append(f"root: {node.content}")
        else:
            indent = "  " * (depth - 1)
            tag = "*" if node.is_user else "[*]"
            lines.append(f"{indent}{tag} {node.content}")
        for child in node.children:
            _walk(child, depth + 1, lines)

    lines: list[str] = []
    _walk(tree.root, 0, lines)
    body = "\n".join(lines)
    return f"{BLOCK_START}\n{body}\n{BLOCK_END}\n"


# ── Renderer ────────────────────────────────────────────────────────────────


def render_tree(tree: MindTree) -> str:
    """Render tree as a compact ASCII outline for display."""
    return tree.to_outline()


def render_mermaid(tree: MindTree) -> str:
    """Render tree as a Mermaid graph (mindmap type).

    Returns markdown with ```mermaid block.
    """
    lines = ["```mermaid", "mindmap"]
    # Track IDs to avoid duplicates
    ids: dict[str, str] = {}
    counter = [0]

    def _short_id(node: MindNode) -> str:
        if node.id not in ids:
            ids[node.id] = f"N{counter[0]}"
            counter[0] += 1
        return ids[node.id]

    def _walk(node: MindNode, depth: int) -> None:
        indent = "  " * depth
        tag = "🟢" if node.is_user else "🔵" if node.is_agent else "📌"
        content = node.content[:60].replace('"', "'")
        lines.append(f"{indent}{tag} {content}")
        for child in node.children:
            _walk(child, depth + 1)

    _walk(tree.root, 1)
    lines.append("```")
    return "\n".join(lines)


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

    # Search for closing ``` AFTER the opening line
    search_from = text.find("\n", start) + 1
    end = text.find(BLOCK_END, search_from)
    if end == -1:
        return text

    end += len(BLOCK_END)
    if end < len(text) and text[end] == "\n":
        end += 1

    return text[:start] + new_block.rstrip("\n") + "\n" + text[end:]
