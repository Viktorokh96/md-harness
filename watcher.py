#!/usr/bin/env python3
"""
CONTROL ROOM watcher — monitors CONTROL_ROOM.md for changes.

On each change:
1. Parse old and new trees
2. Sync .md → .graph.json (merge new nodes, [hide] toggles)
3. Compare trees structurally: if only [hide] flags changed → skip LLM
4. Otherwise → run LLM agent with structural diff

The .md file is a VIEW into the full tree stored in .graph.json.
Hidden subtrees exist only in .graph.json.
"""

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from watchfiles import Change
from watchfiles import watch

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

from agent import CONTROL_ROOM
from agent import ControlRoomContext
from agent import create_agent
from agent import process_change
from tree_engine import diff_trees
from tree_engine import has_pending_questions
from tree_engine import load_graph
from tree_engine import merge_md_into_graph
from tree_engine import parse_mindmap
from tree_engine import replace_block
from tree_engine import save_graph
from tree_engine import serialize_mindmap

MIND_MAP_TEMPLATE = """\
# CONTROL ROOM

> `👤` заметка · `👤❓` вопрос · `🤖` ответ · `👤[hide]` скрыть · `👤[archive]` архив
> Отступ 2 пробела = вложенность · скрытые/архивные ветки — в `.graph.json` / `archive/`

----

```agentsmindmap
root: CONTROL ROOM
```
"""

THINKING_MARKER = "🤖 ...thinking..."


def _append_placeholder(ctx: ControlRoomContext) -> str:
    """Write [*] ...thinking... as temporary visual feedback."""
    content = ctx.file_path.read_text()
    if THINKING_MARKER in content:
        return THINKING_MARKER
    placeholder = f"\n{THINKING_MARKER}\n"
    if not content.endswith("\n"):
        content += "\n"
    ctx.file_path.write_text(content + placeholder)
    return placeholder.strip()


def _remove_thinking_markers(ctx: ControlRoomContext) -> str:
    """Remove all [*] ...thinking... lines from the file."""
    content = ctx.file_path.read_text()
    lines = content.splitlines()
    result = "\n".join(ln for ln in lines if THINKING_MARKER not in ln)
    if result != content:
        ctx.file_path.write_text(result.rstrip("\n") + "\n")
    return result


# ── Graph sync ──────────────────────────────────────────────────────────────


def _sync_graph(ctx: ControlRoomContext) -> str:
    """Sync .md changes into .graph.json. Returns the new .md content.

    Side effects:
    - Creates/updates .graph.json
    - May rewrite .md (hidden children removed, structure normalized)
    """
    md_path = str(ctx.file_path)
    md_text = ctx.file_path.read_text()

    try:
        md_tree = parse_mindmap(md_text)
    except ValueError:
        return md_text  # No valid block yet — don't touch

    full = load_graph(md_path)
    if full is None:
        full = md_tree

    # Detect [archive] / unarchive BEFORE merge (children still intact)
    from archiver import archive_branch
    from archiver import restore_branch

    # Find newly archived nodes: in md_tree but not archived in full
    for node in md_tree.all_nodes():
        full_node = full.get_node(node.id)
        if node.archived and (full_node is None or not full_node.archived):
            # Newly archived — detach subtree from full while children exist
            if full_node is not None and full_node.children:
                print(f"[watcher] Archiving branch {node.id}...")
                archive_branch(md_path, node.id, node.archive_reason)

    # Reload: archive_branch modified .graph.json independently
    full = load_graph(md_path)
    if full is None:
        full = md_tree

    old_archived = {n.id for n in full.all_nodes() if n.archived}
    merged = merge_md_into_graph(full, md_tree)

    # Cleanup: remove ❓ from answered questions (agent has replied)
    for node in merged.all_nodes():
        if node.has_question and node.is_user:
            has_agent_reply = any(child.is_agent for child in node.children)
            if has_agent_reply:
                node.has_question = False

    save_graph(merged, md_path)
    for node in merged.all_nodes():
        if node is merged.root:
            continue
        if not node.archived and node.id in old_archived:
            print(f"[watcher] Restoring branch {node.id}...")
            restore_branch(md_path, node.id)

    block = serialize_mindmap(merged)
    new_content = replace_block(md_text, block)
    ctx.file_path.write_text(new_content)
    return new_content


def run_once(agent: Any, ctx: ControlRoomContext, *, dry_run: bool = False) -> None:
    """Process current state of CONTROL_ROOM.md once."""
    fpath = ctx.file_path
    if not fpath.exists():
        fpath.write_text("# CONTROL ROOM\n\n")
        print(f"[watcher] Created {fpath.name}")
        return

    new_content = fpath.read_text()
    try:
        parse_mindmap(new_content)
    except ValueError:
        print("[watcher] Could not parse mindmap block — skipping.")
        return

    md_path = str(fpath)
    old_graph = load_graph(md_path)
    old_count = len(old_graph.all_nodes()) if old_graph else 0
    print(f"[watcher] old_graph: {old_count} nodes")

    synced_content = _sync_graph(ctx)
    ctx.last_content = synced_content
    ctx.last_mtime = fpath.stat().st_mtime

    new_graph = load_graph(md_path)
    new_count = len(new_graph.all_nodes()) if new_graph else 0
    print(f"[watcher] new_graph: {new_count} nodes")

    has_change = False
    diff_text = ""
    if old_graph is not None and new_graph is not None:
        has_change, diff_text = diff_trees(old_graph, new_graph)
    elif new_graph is not None:
        has_change = len(new_graph.all_nodes()) > 1
        if has_change:
            diff_text = f"Initial tree with {len(new_graph.all_nodes())} nodes."

    context_lines = synced_content.splitlines()[-40:]
    node_count = len(new_graph.all_nodes()) if new_graph else 0
    print(f"[watcher] Processing {fpath.name} ({len(context_lines)} lines, {node_count} nodes)...")
    for ln in context_lines[:8]:
        print(f"  | {ln}")
    if diff_text:
        print(f"\n[watcher] Diff:\n{diff_text}")

    if dry_run:
        print("[watcher] Dry run — skipping agent call.")
        return
    has_questions = has_pending_questions(new_graph) if new_graph else False
    if not has_questions:
        print("[watcher] No questions — mind map mode (skipping LLM).")
        return

    _append_placeholder(ctx)
    print("[watcher] Running agent...")
    result = process_change(agent, ctx, diff_text, context_lines)
    print(f"[watcher] Agent finished: {result[:300]}")
    _remove_thinking_markers(ctx)
    # Re-sync to cleanup answered questions (remove ❓)
    _sync_graph(ctx)
    ctx.last_content = fpath.read_text()
    ctx.last_mtime = fpath.stat().st_mtime


def run_watch(agent: Any, ctx: ControlRoomContext, *, dry_run: bool = False) -> None:
    """Watch the mind map file continuously using inotify (via watchfiles)."""
    fpath = ctx.file_path.resolve()
    print(f"[watcher] Watching {fpath.name} (inotify, real-time)...")
    print("[watcher] Press Ctrl+C to stop.\n")

    if fpath.exists():
        ctx.last_content = fpath.read_text()
        ctx.last_mtime = fpath.stat().st_mtime
    else:
        fpath.write_text(MIND_MAP_TEMPLATE)
        print("[watcher] Created template.")
        ctx.last_content = MIND_MAP_TEMPLATE
        ctx.last_mtime = fpath.stat().st_mtime

    try:
        for changes in watch(fpath.parent):
            for change_type, path in changes:
                if Path(path).resolve() != fpath:
                    continue
                if change_type not in (Change.modified, Change.added):
                    continue
                if not fpath.exists():
                    continue
                content = fpath.read_text()
                if content == ctx.last_content:
                    continue
                run_once(agent, ctx, dry_run=dry_run)
                break
    except KeyboardInterrupt:
        print("\n[watcher] Stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CONTROL ROOM watcher — monitors mind map file for changes.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=str(CONTROL_ROOM),
        help=f"Mind map file to watch (default: {CONTROL_ROOM})",
    )
    parser.add_argument("--once", "-1", action="store_true", help="Process once and exit.")
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show diff without calling the agent.",
    )
    parser.add_argument("--model", "-m", default=None, help="Override LLM model.")

    args = parser.parse_args()
    agent = create_agent(model=args.model)
    ctx = ControlRoomContext(file_path=Path(args.file))

    if args.once:
        run_once(agent, ctx, dry_run=args.dry_run)
    else:
        run_watch(agent, ctx, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
