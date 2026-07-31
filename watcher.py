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
import os
from pathlib import Path

from dotenv import load_dotenv
from watchfiles import Change, watch

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

from agent import (
    CONTROL_ROOM,
    ControlRoomContext,
    create_agent,
    process_change,
)
from tree_engine import (
    diff_trees,
    load_graph,
    merge_md_into_graph,
    parse_mindmap,
    replace_block,
    save_graph,
    serialize_mindmap,
)

MIND_MAP_TEMPLATE = """\
# CONTROL ROOM

> `*` сообщение · `[*]` ответ · `*[hide]` скрыть · `*[archive]` архивировать · `*[archive: причина]`
> Отступ 2 пробела = вложенность · скрытые/архивные ветки — в `.graph.json` / `archive/`

----

```agentsmindmap
root: CONTROL ROOM
```
"""

THINKING_MARKER = "[*] ...thinking..."

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
        save_graph(md_tree, md_path)
        return md_text

    # Detect [archive] / unarchive BEFORE merge (children still intact)
    from archiver import archive_branch, restore_branch

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
    merged = merge_md_into_graph(full, md_tree)

    # Handle unarchive (restore)
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


def run_once(agent, ctx: ControlRoomContext, *, dry_run: bool = False) -> None:
    """Process current state of CONTROL_ROOM.md once."""
    fpath = ctx.file_path
    if not fpath.exists():
        fpath.write_text("# CONTROL ROOM\n\n")
        print(f"[watcher] Created {fpath.name}")
        return

    new_content = fpath.read_text()
    old_content = ctx.last_content or ""

    try:
        old_tree = parse_mindmap(old_content) if old_content else None
        new_tree = parse_mindmap(new_content)
    except ValueError:
        print("[watcher] Could not parse mindmap block — skipping.")
        return

    synced_content = _sync_graph(ctx)
    ctx.last_content = synced_content
    ctx.last_mtime = fpath.stat().st_mtime

    has_change = False
    diff_text = ""
    if old_tree is not None:
        has_change, diff_text = diff_trees(old_tree, new_tree)
    else:
        has_change = len(new_tree.all_nodes()) > 1
        if has_change:
            diff_text = f"Initial tree with {len(new_tree.all_nodes())} nodes."

    context_lines = synced_content.splitlines()[-40:]
    print(f"[watcher] Processing {fpath.name} ({len(context_lines)} lines)...")
    for ln in context_lines[:8]:
        print(f"  | {ln}")
    if len(context_lines) > 8:
        print(f"  ... ({len(context_lines) - 8} more lines)")

    if diff_text:
        print(f"\n[watcher] Diff:\n{diff_text}")

    if dry_run:
        print("[watcher] Dry run — skipping agent call.")
        return

    if not has_change:
        print("[watcher] No content change — skipping LLM.")
        return

    _append_placeholder(ctx)
    print("[watcher] Running agent...")
    result = process_change(agent, ctx, diff_text, context_lines)
    print(f"[watcher] Agent finished: {result[:300]}")

    _remove_thinking_markers(ctx)
    ctx.last_content = fpath.read_text()
    ctx.last_mtime = fpath.stat().st_mtime


def run_watch(agent, ctx: ControlRoomContext, *, dry_run: bool = False) -> None:
    """Watch the mind map file continuously using inotify (via watchfiles)."""
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
            # Filter: only care about our target file
            for change_type, path in changes:
                if Path(path).resolve() != fpath:
                    continue
                if change_type not in (Change.modified, Change.added):
                    continue
                if not fpath.exists():
                    continue

                content = fpath.read_text()
                if content == ctx.last_content:
                    continue  # debounce: same content, likely double-write

                run_once(agent, ctx, dry_run=dry_run)
                break  # one trigger per batch
    except KeyboardInterrupt:
        print("\n[watcher] Stopped.")
# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CONTROL ROOM watcher — monitors mind map file for changes.",
    )
    parser.add_argument(
        "file", nargs="?", default=str(CONTROL_ROOM),
        help=f"Mind map file to watch (default: {CONTROL_ROOM})",
    )
    parser.add_argument("--once", "-1", action="store_true", help="Process once and exit.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show diff without calling the agent.")
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
