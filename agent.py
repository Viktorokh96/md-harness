"""
CONTROL ROOM agent — built on OpenAI Agents SDK.

Tree-based mind map with ```agentsmindmap blocks.
Tools: read_mindmap, add_reply, batch_reply, find_nodes, stay_silent,
        run_shell, read_file, delegate_task (Worker).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from agents import Agent
from agents import RunContextWrapper
from agents import Runner
from agents import function_tool
from agents import set_tracing_disabled

from tree_engine import MindTree
from tree_engine import load_graph
from tree_engine import parse_mindmap
from tree_engine import replace_block
from tree_engine import save_graph
from tree_engine import serialize_mindmap

# ── Setup ───────────────────────────────────────────────────────────────────

if os.environ.get("DISABLE_TRACING", "1") != "0":
    set_tracing_disabled(True)


CONTROL_ROOM = Path(__file__).resolve().parent / "CONTROL_ROOM.md"


@dataclass
class ControlRoomContext:
    """Context passed through the agent run."""

    file_path: Path = field(default_factory=lambda: CONTROL_ROOM)
    last_content: str = ""
    last_mtime: float = 0.0


# ── Tree helpers ────────────────────────────────────────────────────────────


def _read_tree(ctx: ControlRoomContext) -> MindTree:
    """Parse the mindmap tree — from .graph.json if available, else .md."""
    tree = load_graph(str(ctx.file_path))
    if tree is not None:
        return tree
    # Fallback: parse from .md (first run, no .graph.json yet)
    return parse_mindmap(ctx.file_path.read_text())


def _write_tree(ctx: ControlRoomContext, tree: MindTree) -> None:
    """Serialize tree to both .md (visible nodes) and .graph.json (full tree)."""
    # Save full graph
    save_graph(tree, str(ctx.file_path))
    # Write visible part to .md
    content = ctx.file_path.read_text()
    block = serialize_mindmap(tree)
    new_content = replace_block(content, block)
    ctx.file_path.write_text(new_content)


# ── Tool implementations (testable without SDK) ──────────────────────────────


def _impl_read_mindmap(ctx: ControlRoomContext) -> str:
    """Read the full mind map tree with node IDs."""
    tree = _read_tree(ctx)
    return tree.to_outline()


def _impl_add_reply(ctx: ControlRoomContext, parent_id: str, content: str) -> str:
    """Add a reply node under parent_id in the tree."""
    tree = _read_tree(ctx)
    tree.add_reply(parent_id, content, author="agent")
    _write_tree(ctx, tree)
    return f"Added agent reply under {parent_id}: {content}"


def _impl_batch_reply(
    ctx: ControlRoomContext,
    parent_id: str,
    replies: list[str],
) -> str:
    """Create MULTIPLE reply branches under parent_id at once."""
    tree = _read_tree(ctx)
    ids: list[str] = []
    for text in replies:
        child = tree.add_reply(parent_id, text, author="agent")
        ids.append(child.id)
    _write_tree(ctx, tree)
    return f"Added {len(replies)} replies under {parent_id}: {', '.join(ids)}"


def _impl_find_nodes(ctx: ControlRoomContext, query: str) -> str:
    """Search the mind map tree for nodes containing query."""
    tree = _read_tree(ctx)
    results = tree.find_nodes(query)
    if not results:
        return f"No nodes found matching: {query}"
    lines: list[str] = []
    for node in results:
        path = " → ".join(node.path())
        tag = "👤❓" if node.has_question else ("👤" if node.is_user else "🤖")
        lines.append(f"  [{node.id}] {tag} {node.content}  (path: {path})")
    return "\n".join(lines)


def _impl_stay_silent() -> str:
    """Choose to say nothing."""
    return "SILENT — no response needed."


def _impl_run_shell(
    command: str,
    *,
    timeout: int = 120,
    cwd: Path | None = None,
) -> str:
    """Execute a shell command. Returns stdout/stderr/exit code."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts: list[str] = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"


def _impl_read_file(
    filepath: str,
    *,
    root_dir: Path,
    max_lines: int = 200,
) -> str:
    """Read a project file. Path is relative to root_dir. Blocks traversal."""
    root_resolved = root_dir.resolve()
    target = (root_dir / filepath).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        return f"ERROR: path escapes project root: {filepath}"
    if not target.exists():
        return f"ERROR: file not found: {filepath}"
    try:
        content = target.read_text()
        lines = content.splitlines()
        if len(lines) > max_lines:
            head = "\n".join(lines[:max_lines])
            return f"{head}\n\n... (truncated, {len(lines)} total lines, showing first {max_lines})"
        return content
    except Exception as exc:
        return f"ERROR reading {filepath}: {exc}"


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
def read_mindmap(wrapper: RunContextWrapper[ControlRoomContext]) -> str:
    """Read the full mind map tree with node IDs."""
    return _impl_read_mindmap(wrapper.context)


@function_tool
def add_reply(
    wrapper: RunContextWrapper[ControlRoomContext],
    parent_id: str,
    content: str,
) -> str:
    """Add a reply node under `parent_id` in the tree. Saves to file."""
    return _impl_add_reply(wrapper.context, parent_id, content)


@function_tool
def batch_reply(
    wrapper: RunContextWrapper[ControlRoomContext],
    parent_id: str,
    replies: list[str],
) -> str:
    """Create MULTIPLE reply branches under `parent_id` at once."""
    return _impl_batch_reply(wrapper.context, parent_id, replies)


@function_tool
def find_nodes(
    wrapper: RunContextWrapper[ControlRoomContext],
    query: str,
) -> str:
    """Search the mind map tree for nodes containing `query`."""
    return _impl_find_nodes(wrapper.context, query)


@function_tool
def stay_silent() -> str:
    """Choose to say nothing. Call this when no response is needed."""
    return _impl_stay_silent()


@function_tool
def run_shell(
    wrapper: RunContextWrapper[ControlRoomContext],
    command: str,
    *,
    timeout: int = 120,
) -> str:
    """Execute a shell command. Returns stdout/stderr/exit code."""
    return _impl_run_shell(
        command,
        timeout=timeout,
        cwd=wrapper.context.file_path.parent,
    )


@function_tool
def read_file(
    wrapper: RunContextWrapper[ControlRoomContext],
    filepath: str,
    *,
    max_lines: int = 200,
) -> str:
    """Read a project file. Path is relative to project root."""
    return _impl_read_file(
        filepath,
        root_dir=wrapper.context.file_path.parent,
        max_lines=max_lines,
    )


# ── Worker sub-agent ────────────────────────────────────────────────────────

WORKER_INSTRUCTIONS = """\
You are a Worker sub-agent — a general-purpose task executor.
You receive tasks from the CONTROL ROOM agent and execute them.

You have access to:
- `run_shell` — execute shell commands in the project
- `read_file` — read any file in the project

Rules:
- Be thorough: explore, verify, and report findings clearly.
- If a task requires multiple steps, do them in order.
- Report results concisely — the main agent will decide what to show the user.
- Never modify CONTROL_ROOM.md — only the main agent does that.
- Return your final answer as a plain text summary.
"""


def _create_worker(model: str) -> Agent[ControlRoomContext]:
    return Agent[ControlRoomContext](
        name="Worker",
        instructions=WORKER_INSTRUCTIONS,
        tools=[run_shell, read_file],
        model=model,
    )


def _make_worker_tool(model: str) -> Any:
    worker = _create_worker(model)
    return worker.as_tool(
        tool_name="delegate_task",
        tool_description=(
            "Delegate a task to a Worker sub-agent. "
            "Use for multi-step work: researching code, running commands, "
            "checking facts, exploring the project. "
            "Give it a clear, self-contained task description."
        ),
    )


# ── Agent definition ────────────────────────────────────────────────────────

INSTRUCTIONS = """\
You are the CONTROL ROOM agent — an AI operating on a tree-structured mind map
stored in a ```agentsmindmap markdown block.

## How it works
The mind map is a TREE with node IDs. You are ONLY called when the user
adds a `?` after the person emoji (`👤❓`) — this is a question for you.
Nodes with just `👤` are the user's own notes (mind map mode) — ignore them.

1. Call `read_mindmap()` to see the full tree with IDs.
2. Find the node with `👤❓` (has_question=True) — that's your target.
3. Call `add_reply(parent_id, content)` for ONE reply, or
   `batch_reply(parent_id, ["idea1", "idea2", ...])` for MULTIPLE replies.
4. After you reply, the `❓` is automatically removed — no action needed.

## batch_reply usage
- When asked for N separate ideas/responses: use `batch_reply(parent_id, [...])`
  with ALL replies in a single list. One call = N branches. Much more reliable
  than calling `add_reply` N times.
- Each string in the list becomes one `🤖` branch under parent_id.
- You are triggered by `👤❓` nodes — reply directly to the question.
- `👤` nodes = user notes (mind map — no action needed)
- `👤❓` nodes = user questions (YOUR TARGET — reply to these)
- `🤖` nodes = your (agent) messages
- Node IDs look like `root.1`, `root.1.1`, `root.1.1.1`

## Prefix icons
- (no icon) — analysis, thought, observation
- ❓ — question for the user
- ▶ — action taken (shell command, file read, etc.)
- ✅ — task completed, decision made, fact confirmed

## Rules
- ALWAYS call `read_mindmap()` first.
- Reply to the latest user message unless addressing an earlier topic.
- Be concise: 1-3 lines per reply.
- Use `find_nodes(query)` to search for topics.
- Use `run_shell` and `read_file` to explore the codebase.
- Use `delegate_task` for multi-step work.
- NEVER claim "создано N веток" unless you actually called add_reply/batch_reply.
"""


def create_agent(model: str | None = None) -> Agent[ControlRoomContext]:
    """Create the CONTROL ROOM agent instance."""
    return Agent[ControlRoomContext](
        name="ControlRoom",
        instructions=INSTRUCTIONS,
        tools=[
            read_mindmap,
            add_reply,
            batch_reply,
            find_nodes,
            stay_silent,
            run_shell,
            read_file,
            _make_worker_tool(model or os.environ.get("LLM_MODEL", "gpt-4o-mini")),
        ],
    )


# ── Diff computation ────────────────────────────────────────────────────────


def compute_diff(old: str, new: str) -> str:
    """Human-readable diff showing what was added/changed."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    if len(new_lines) > len(old_lines) and new_lines[: len(old_lines)] == old_lines:
        added = new_lines[len(old_lines) :]
        return "Added:\n" + "\n".join(f"  + {ln}" for ln in added)

    if len(new_lines) < len(old_lines) and new_lines == old_lines[: len(new_lines)]:
        removed = old_lines[len(new_lines) :]
        return "Removed:\n" + "\n".join(f"  - {ln}" for ln in removed)

    parts: list[str] = []
    max_len = max(len(old_lines), len(new_lines))
    changed = False
    for i in range(max_len):
        old_ln = old_lines[i] if i < len(old_lines) else None
        new_ln = new_lines[i] if i < len(new_lines) else None
        if old_ln != new_ln:
            if not changed:
                parts.append(f"  @@ line {i + 1}:")
                changed = True
            if old_ln is not None:
                parts.append(f"  - {old_ln}")
            if new_ln is not None:
                parts.append(f"  + {new_ln}")
        else:
            changed = False
    return "\n".join(parts) if parts else "(no visible changes)"


# ── Processing ──────────────────────────────────────────────────────────────


def process_change(
    agent: Agent[ControlRoomContext],
    ctx: ControlRoomContext,
    diff: str,
    context_lines: list[str],
) -> str:
    """Run the agent on a detected change."""
    context = "\n".join(context_lines)

    prompt = f"""\
CONTROL_ROOM.md changed. Here's the diff:

{diff}

Recent file context:
```
{context}
```

FIRST: call `read_mindmap()` to see the full tree with node IDs.
THEN: find the user message that was added, and reply with `add_reply()`.
Or call `stay_silent()` if no response is needed."""

    try:
        result = Runner.run_sync(agent, prompt, context=ctx, max_turns=100)
        return str(result.final_output)
    except Exception as exc:
        return f"Agent error: {exc}"


if __name__ == "__main__":
    ctx = ControlRoomContext()
    if CONTROL_ROOM.exists():
        ctx.last_content = CONTROL_ROOM.read_text()
    agent = create_agent()
    print(f"Agent: {agent.name}")
    print(f"Model: {agent.model}")
    print(f"Tools: {[t.name for t in agent.tools]}")
