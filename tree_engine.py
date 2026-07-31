"""Mind map tree engine — parser, serializer, renderer for ```agentsmindmap blocks.

Format:
    ```agentsmindmap
    root: Room Name
    * user message
    *[hide] summary                         # hidden branch
    *[archive] archived summary             # archived branch
    *[archive: reason] archived summary     # archived with reason
      [*] agent reply
    ```

Features:
- Tree with dot-separated node IDs (root.1, root.1.1, ...)
- [hide]: visual collapse, full tree in .graph.json
- [archive]: detach subtree to archive/, summary line remains
- [archive: reason]: same with explicit reason
- Continuation lines: indented non-tag lines → parent content
- JSON: to_dict/from_dict for .graph.json
- diff_trees: structural comparison
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
ARCHIVE_RE = re.compile(r"^(\*|\[\*\])\[archive\](\s.*)?$")
ARCHIVE_REASON_RE = re.compile(r"^(\*|\[\*\])\[archive:\s*(.+?)\]\s?(.*)?$")
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
    archived: bool = False
    archive_reason: str = ""
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
        ids: list[str] = []
        node: MindNode | None = self
        while node is not None:
            ids.append(node.id)
            node = node.parent
        ids.reverse()
        return ids

    def __repr__(self) -> str:
        prefix = "*" if self.is_user else "[*]"
        flags = ""
        if self.hidden:
            flags += "[hide]"
        if self.archived:
            flags += "[archive]"
        return f"MindNode({self.id}, {prefix}{flags} {self.content[:40]})"


# ── Tree ────────────────────────────────────────────────────────────────────


@dataclass
class MindTree:
    """Full mind map tree with parse/serialize/render capabilities."""

    root: MindNode
    _node_index: dict[str, MindNode] = field(default_factory=dict)

    def get_node(self, node_id: str) -> MindNode | None:
        return self._node_index.get(node_id)

    def all_nodes(self) -> list[MindNode]:
        return list(self._node_index.values())

    def find_nodes(self, query: str) -> list[MindNode]:
        q = query.lower()
        return [n for n in self._node_index.values() if q in n.content.lower()]

    def add_reply(self, parent_id: str, content: str, author: str) -> MindNode:
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

    def detach_subtree(self, node_id: str) -> MindTree | None:
        """Remove node + children, return as standalone MindTree."""
        node = self._node_index.get(node_id)
        if node is None or node is self.root:
            return None
        if node.parent:
            node.parent.children = [c for c in node.parent.children if c.id != node_id]
        node.parent = None

        def _remove(n: MindNode) -> None:
            self._node_index.pop(n.id, None)
            for c in n.children:
                _remove(c)
        _remove(node)

        subtree = MindTree(root=node)
        def _reindex(n: MindNode) -> None:
            subtree._node_index[n.id] = n
            for c in n.children:
                _reindex(c)
        _reindex(node)
        return subtree

    def attach_subtree(self, parent_id: str, subtree: MindTree) -> MindNode:
        """Attach subtree under parent_id. Returns the attached root."""
        parent = self._node_index.get(parent_id)
        if parent is None:
            raise KeyError(f"Parent not found: {parent_id}")
        node = subtree.root
        node.parent = parent
        parent.add_child(node)

        def _fix_ids(n: MindNode, prefix: str, base_depth: int) -> None:
            n.id = prefix
            n.depth = base_depth
            self._node_index[n.id] = n
            for i, c in enumerate(n.children, 1):
                _fix_ids(c, f"{prefix}.{i}", base_depth + 1)

        idx = len(parent.children)  # position among siblings
        _fix_ids(node, f"{parent_id}.{idx}", parent.depth + 1)
        return node

    def to_outline(self, *, show_hidden: bool = True) -> str:
        lines: list[str] = []

        def _walk(node: MindNode, indent: int) -> None:
            prefix = "  " * indent
            if node is self.root:
                lines.append(f"📌 [{node.id}] {node.content}")
            else:
                if node.archived:
                    marker = "🗄"
                elif node.hidden:
                    marker = "📦"
                else:
                    marker = "*" if node.is_user else "[*]"
                lines.append(f"{prefix}{marker} [{node.id}] {node.content}")
            if (node.hidden or node.archived) and not show_hidden:
                return
            for child in node.children:
                _walk(child, indent + 1)

        _walk(self.root, 0)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def _node_to_dict(node: MindNode) -> dict[str, Any]:
            d: dict[str, Any] = {
                "id": node.id, "content": node.content, "author": node.author,
                "depth": node.depth, "hidden": node.hidden,
                "children": [_node_to_dict(c) for c in node.children],
            }
            if node.archived:
                d["archived"] = True
                d["archive_reason"] = node.archive_reason
            return d
        return _node_to_dict(self.root)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MindTree:
        def _dict_to_node(d: dict[str, Any]) -> MindNode:
            node = MindNode(
                id=d["id"], content=d["content"], author=d["author"],
                depth=d["depth"], hidden=d.get("hidden", False),
                archived=d.get("archived", False),
                archive_reason=d.get("archive_reason", ""),
            )
            for cd in d.get("children", []):
                node.add_child(_dict_to_node(cd))
            return node
        root = _dict_to_node(data)
        tree = cls(root=root)
        def _index(n: MindNode) -> None:
            tree._node_index[n.id] = n
            for c in n.children:
                _index(c)
        _index(root)
        return tree


# ── Parser ──────────────────────────────────────────────────────────────────


def parse_mindmap(text: str) -> MindTree:
    start = text.find(BLOCK_START)
    if start == -1:
        raise ValueError(f"No {BLOCK_START} block found")
    start = text.find("\n", start) + 1
    end = text.find(BLOCK_END, start)
    if end == -1:
        raise ValueError(f"Unclosed {BLOCK_START} block")
    return _parse_block(text[start:end].strip())


def _parse_block(block: str) -> MindTree:
    lines = block.splitlines()
    if not lines:
        raise ValueError("Empty mindmap block")
    root_line = lines[0].strip()
    if not root_line.startswith("root:"):
        raise ValueError(f"First line must be 'root: <name>', got: {root_line}")
    root_name = root_line[5:].strip()
    root = MindNode(id="root", content=root_name, author="system", depth=0)
    tree = MindTree(root=root)
    tree._node_index["root"] = root

    stack: dict[int, MindNode] = {0: root}
    counter: dict[int, int] = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        depth = indent // 2 + 1
        if indent % 2 != 0:
            raise ValueError(f"Indent must be multiple of 2: {line!r}")

        hidden = False
        archived = False
        archive_reason = ""

        hide_m = HIDE_RE.match(stripped)
        arc_reason_m = ARCHIVE_REASON_RE.match(stripped)
        arc_m = ARCHIVE_RE.match(stripped)
        tag_m = TAG_RE.match(stripped)

        if hide_m:
            tag = hide_m.group(1)
            author = "agent" if tag == "[*]" else "user"
            content = stripped[hide_m.end():].strip()
            hidden = True
        elif arc_reason_m:
            tag = arc_reason_m.group(1)
            author = "agent" if tag == "[*]" else "user"
            archive_reason = arc_reason_m.group(2).strip()
            rest = arc_reason_m.group(3) or ""
            content = rest.strip()
            archived = True
        elif arc_m:
            tag = arc_m.group(1)
            author = "agent" if tag == "[*]" else "user"
            rest = arc_m.group(2) or ""
            content = rest.strip()
            archived = True
        elif tag_m:
            tag = tag_m.group(1)
            author = "agent" if tag == "[*]" else "user"
            content = stripped[tag_m.end():].strip()
        else:
            parent_node = stack.get(depth - 1)
            if parent_node is None:
                raise ValueError(f"No parent for continuation: {line!r}")
            parent_node.content += "\n" + stripped
            continue

        parent = stack.get(depth - 1)
        if parent is None:
            raise ValueError(f"No parent at depth {depth - 1}: {line!r}")
        counter.setdefault(depth, 0)
        counter[depth] += 1
        node_id = f"{parent.id}.{counter[depth]}"

        node = MindNode(
            id=node_id, content=content, author=author, depth=depth,
            hidden=hidden, archived=archived, archive_reason=archive_reason,
        )
        parent.add_child(node)
        tree._node_index[node_id] = node
        stack[depth] = node
        for d in list(stack.keys()):
            if d > depth:
                del stack[d]

    return tree


# ── Serializer ──────────────────────────────────────────────────────────────


def serialize_mindmap(tree: MindTree) -> str:
    """Serialize tree to ```agentsmindmap block. Hidden/archived: collapsed."""

    def _walk(node: MindNode, depth: int, lines: list[str]) -> None:
        if node is tree.root:
            lines.append(f"root: {node.content}")
        else:
            indent = "  " * (depth - 1)
            tag = "*" if node.is_user else "[*]"
            if node.archived:
                reason = node.archive_reason
                tag = f"{tag}[archive{f': {reason}' if reason else ''}]"
            elif node.hidden:
                tag = f"{tag}[hide]"
            content_lines = node.content.split("\n")
            lines.append(f"{indent}{tag} {content_lines[0]}")
            cont_indent = "  " * depth
            for cont in content_lines[1:]:
                lines.append(f"{cont_indent}{cont}")
        if node.hidden or node.archived:
            return
        for child in node.children:
            _walk(child, depth + 1, lines)

    lines: list[str] = []
    _walk(tree.root, 0, lines)
    body = "\n".join(lines)
    return f"{BLOCK_START}\n{body}\n{BLOCK_END}\n"


# ── Merge ───────────────────────────────────────────────────────────────────


def merge_md_into_graph(graph: MindTree, md_tree: MindTree) -> MindTree:
    """Merge .md changes into full graph. .md authoritative for content/flags."""
    graph._node_index = {}
    _reindex(graph.root, graph._node_index)
    md_tree._node_index = {}
    _reindex(md_tree.root, md_tree._node_index)

    def _merge(g: MindNode, m: MindNode | None) -> MindNode:
        if m is not None:
            was_hidden = g.hidden
            g.content = m.content
            g.hidden = m.hidden
            g.archived = m.archived
            g.archive_reason = m.archive_reason

            g_children = {c.id: c for c in g.children}
            new_children: list[MindNode] = []
            for mc in m.children:
                merged = _merge(g_children[mc.id], mc) if mc.id in g_children else _copy_node(mc)
                new_children.append(merged)
                merged.parent = g

            if (was_hidden and not g.hidden) or g.hidden:
                if not new_children and g.children:
                    new_children = [_copy_node(c) for c in g.children]
                    for c in new_children:
                        c.parent = g
            g.children = new_children
        return g

    _merge(graph.root, md_tree.root)
    graph._node_index = {}
    _reindex(graph.root, graph._node_index)
    return graph


def _copy_node(node: MindNode) -> MindNode:
    new = MindNode(
        id=node.id, content=node.content, author=node.author, depth=node.depth,
        hidden=node.hidden, archived=node.archived, archive_reason=node.archive_reason,
    )
    for c in node.children:
        new.add_child(_copy_node(c))
    return new


def _reindex(node: MindNode, index: dict[str, MindNode]) -> None:
    index[node.id] = node
    for c in node.children:
        _reindex(c, index)


# ── Diff ────────────────────────────────────────────────────────────────────


def diff_trees(old: MindTree, new: MindTree) -> tuple[bool, str]:
    """Structural diff. Content change = new nodes or changed content.
    Hide/archive toggles alone are NOT content changes.
    """
    new_nodes: list[str] = []
    changed: list[str] = []
    unhidden: list[str] = []
    hidden: list[str] = []
    archived: list[str] = []
    unarchived: list[str] = []

    def _compare(o: MindNode | None, n: MindNode | None, path: str) -> None:
        if o is None and n is not None:
            tag = "*" if n.is_user else "[*]"
            h = "[hide]" if n.hidden else ""
            a = "[archive]" if n.archived else ""
            new_nodes.append(f"  {tag}{h}{a} {n.content}  [id: {n.id}]")
            for c in n.children:
                _compare(None, c, f"{path}/{n.id}")
            return
        if n is None or o is None:
            return
        if o.content != n.content:
            changed.append(f"  [{n.id}] changed: {o.content[:60]!r} → {n.content[:60]!r}")
        if o.hidden and not n.hidden:
            unhidden.append(f"  [{n.id}] unhidden: {n.content[:60]}")
        elif not o.hidden and n.hidden:
            hidden.append(f"  [{n.id}] hidden: {n.content[:60]}")
        if o.archived and not n.archived:
            unarchived.append(f"  [{n.id}] unarchived: {n.content[:60]}")
        elif not o.archived and n.archived:
            archived.append(f"  [{n.id}] archived: {n.content[:60]}")
        o_kids = {c.id: c for c in o.children}
        n_kids = {c.id: c for c in n.children}
        for cid in n_kids:
            _compare(o_kids.get(cid), n_kids[cid], f"{path}/{cid}")

    _compare(old.root, new.root, "")
    has_change = bool(new_nodes or changed)

    parts: list[str] = []
    if new_nodes:
        parts.append("New nodes:\n" + "\n".join(new_nodes))
    if changed:
        parts.append("Content changed:\n" + "\n".join(changed))
    if unhidden:
        parts.append("Unhidden:\n" + "\n".join(unhidden))
    if hidden:
        parts.append("Hidden:\n" + "\n".join(hidden))
    if archived:
        parts.append("Archived:\n" + "\n".join(archived))
    if unarchived:
        parts.append("Unarchived:\n" + "\n".join(unarchived))
    return has_change, "\n".join(parts) if parts else "(no structural changes)"


# ── Renderer ────────────────────────────────────────────────────────────────


def render_mermaid(tree: MindTree) -> str:
    lines = ["```mermaid", "mindmap"]
    def _walk(node: MindNode, indent: str) -> None:
        if node is tree.root:
            lines.append(f"  {indent}{_sanitize(node.content)}")
        else:
            if node.archived:
                marker = "🗄"
            elif node.hidden:
                marker = "📦"
            else:
                marker = "👤" if node.is_user else "🤖"
            lines.append(f"  {indent}{marker} {_sanitize(node.content)}")
        if node.hidden or node.archived:
            return
        for child in node.children:
            _walk(child, indent + "  ")
    _walk(tree.root, "")
    lines.append("```")
    return "\n".join(lines)


def _sanitize(text: str) -> str:
    return text.replace("\n", " ").replace('"', "'").replace("(", "[").replace(")", "]")[:80]


# ── Block extraction ────────────────────────────────────────────────────────


def extract_block(text: str) -> str | None:
    start = text.find(BLOCK_START)
    if start == -1:
        return None
    start = text.find("\n", start) + 1
    end = text.find(BLOCK_END, start)
    return text[start:end].strip() if end != -1 else None


def replace_block(text: str, new_block: str) -> str:
    start = text.find(BLOCK_START)
    if start == -1:
        if text and not text.endswith("\n"):
            text += "\n"
        return text + "\n" + new_block
    search_from = text.find("\n", start) + 1
    end = text.find(f"\n{BLOCK_END}", search_from)
    if end == -1:
        end = len(text)
    else:
        end += 1 + len(BLOCK_END)
    return text[:start] + new_block.rstrip("\n") + "\n" + text[end:]


# ── JSON sidecar ────────────────────────────────────────────────────────────


def graph_path(md_path: str) -> str:
    return md_path.replace(".md", ".md_graph.json")


def load_graph(md_path: str) -> MindTree | None:
    gp = graph_path(md_path)
    try:
        with open(gp) as f:
            return MindTree.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_graph(tree: MindTree, md_path: str) -> None:
    with open(graph_path(md_path), "w") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
