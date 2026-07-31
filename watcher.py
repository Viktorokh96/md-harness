#!/usr/bin/env python3
"""
CONTROL ROOM watcher — monitors CONTROL_ROOM.md for changes.

On each change:
1. Sync .md → .graph.json (merge new nodes, [hide] toggles)
2. If change is [hide]-only → process locally (no LLM call)
3. Otherwise → run LLM agent

The .md file is a VIEW into the full tree stored in .graph.json.
Hidden subtrees exist only in .graph.json.
"""

import argparse
import os
import re
import time
from pathlib import Path

# ── Load .env ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip()

from agent import (
    CONTROL_ROOM,
    ControlRoomContext,
    compute_diff,
    create_agent,
    process_change,
)
from tree_engine import (
    load_graph,
    merge_md_into_graph,
    parse_mindmap,
    replace_block,
    save_graph,
    serialize_mindmap,
)

MIND_MAP_TEMPLATE = """\
# CONTROL ROOM

```agentsmindmap
root: CONTROL ROOM
```
"""
THINKING_MARKER = "[*] ...thinking..."

# ── Placeholder ─────────────────────────────────────────────────────────────


def _append_placeholder(ctx: ControlRoomContext, _user_content: str = "") -> str:
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


# ── Hide-only diff detection ────────────────────────────────────────────────


def _is_hide_only_diff(old: str, new: str) -> bool:
    """Check if diff only changes [hide] markers — no content additions."""
    if old == new:
        return False

    old_lines = set(old.splitlines())
    new_lines = set(new.splitlines())

    added = new_lines - old_lines
    removed = old_lines - new_lines

    if not added and not removed:
        return False

    for line in added | removed:
        if not line.strip():
            continue
        counterpart = _find_counterpart(line, added | removed)
        if not _is_hide_toggle(line, counterpart):
            return False

    return True


def _is_hide_toggle(a: str, b: str | None) -> bool:
    """Check if two lines differ only by [hide] presence."""
    if b is not None:
        return a.replace("[hide]", "").strip() == b.replace("[hide]", "").strip()
    return "[hide]" in a and (
        a.strip().startswith("*[hide] ") or a.strip().startswith("[*][hide] ")
    )


def _find_counterpart(line: str, all_lines: set[str]) -> str | None:
    """Find the counterpart line (with/without [hide]) in the set."""
    without = line.replace("[hide]", "")
    with_hide = _insert_hide(line)
    if without in all_lines and without != line:
        return without
    if with_hide in all_lines and with_hide != line:
        return with_hide
    # Try prefix variations
    for cand in all_lines:
        if cand.replace("[hide]", "").strip() == line.replace("[hide]", "").strip() and cand != line:
            return cand
    return None


def _insert_hide(line: str) -> str:
    """Insert [hide] after the tag marker."""
    stripped = line.strip()
    indent = line[:len(line) - len(stripped)]
    if stripped.startswith("[*] ") and "[hide]" not in stripped:
        return f"{indent}[*][hide] {stripped[4:]}"
    if stripped.startswith("* ") and "[hide]" not in stripped:
        return f"{indent}*[hide] {stripped[2:]}"
    return line


# ── Graph sync ──────────────────────────────────────────────────────────────


def _sync_graph(ctx: ControlRoomContext) -> None:
    """Sync .md changes into .graph.json.

    - First run: create .graph.json from .md
    - Subsequent runs: merge .md changes into full graph
    """
    md_path = str(ctx.file_path)
    md_text = ctx.file_path.read_text()

    try:
        md_tree = parse_mindmap(md_text)
    except ValueError:
        return  # No valid block yet

    full = load_graph(md_path)
    if full is None:
        save_graph(md_tree, md_path)
    else:
        merged = merge_md_into_graph(full, md_tree)
        save_graph(merged, md_path)
        block = serialize_mindmap(merged)
        new_content = replace_block(md_text, block)
        ctx.file_path.write_text(new_content)


# ── Run ─────────────────────────────────────────────────────────────────────


def run_once(agent, ctx: ControlRoomContext, *, dry_run: bool = False) -> None:
    """Process current state of CONTROL_ROOM.md once."""
    fpath = ctx.file_path
    if not fpath.exists():
        fpath.write_text("# CONTROL ROOM\n\n")
        print(f"[watcher] Created {fpath.name}")
        return

    content = fpath.read_text()
    context_lines = content.splitlines()[-60:]
    diff = compute_diff(ctx.last_content, content)

    print(f"[watcher] Processing {fpath.name} ({len(context_lines)} lines)...")
    for ln in context_lines[:10]:
        print(f"  | {ln}")
    if len(context_lines) > 10:
        print(f"  ... ({len(context_lines) - 10} more lines)")

    if diff and diff != "(no visible changes)":
        print(f"\n[watcher] Diff:\n{diff}")

    if dry_run:
        print("[watcher] Dry run — skipping agent call.")
        return

    _sync_graph(ctx)

    if _is_hide_only_diff(ctx.last_content, content):
        print("[watcher] [hide]-only change — processing locally (no LLM).")
        ctx.last_content = content
        ctx.last_mtime = fpath.stat().st_mtime
        return

    _append_placeholder(ctx, content)
    print("[watcher] Running agent...")
    result = process_change(agent, ctx, diff, context_lines)
    print(f"[watcher] Agent finished: {result[:300]}")

    _remove_thinking_markers(ctx)


def run_watch(agent, ctx: ControlRoomContext, *, dry_run: bool = False, interval: float = 2.0) -> None:
    """Watch the mind map file continuously, processing changes."""
    fpath = ctx.file_path
    print(f"[watcher] Watching {fpath} (poll every {interval}s)...")
    print("[watcher] Press Ctrl+C to stop.\n")

    if fpath.exists():
        ctx.last_content = fpath.read_text()
        ctx.last_mtime = fpath.stat().st_mtime
    else:
        fpath.write_text(MIND_MAP_TEMPLATE)
        print("[watcher] Created template.")
        ctx.last_content = MIND_MAP_TEMPLATE
        ctx.last_mtime = fpath.stat().st_mtime

    while True:
        try:
            if not fpath.exists():
                time.sleep(interval)
                continue

            mtime = fpath.stat().st_mtime
            if mtime != ctx.last_mtime:
                content = fpath.read_text()
                if content != ctx.last_content:
                    run_once(agent, ctx, dry_run=dry_run)
                    ctx.last_content = content
                    ctx.last_mtime = mtime

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[watcher] Stopped.")
            break
        except Exception as exc:
            print(f"[watcher] Error: {exc}")
            time.sleep(2)


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
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Poll interval (default: 2.0).")

    args = parser.parse_args()
    agent = create_agent(model=args.model)
    ctx = ControlRoomContext(file_path=Path(args.file))

    if args.once:
        run_once(agent, ctx, dry_run=args.dry_run)
    else:
        run_watch(agent, ctx, dry_run=args.dry_run, interval=args.interval)


if __name__ == "__main__":
    main()
