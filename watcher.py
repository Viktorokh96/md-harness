#!/usr/bin/env python3
"""
CONTROL ROOM watcher — monitors CONTROL_ROOM.md for changes,
triggers the ControlRoomAgent on each change.

Usage:
    python watcher.py                  # watch continuously
    python watcher.py --once           # process once and exit
    python watcher.py --dry-run        # show diffs without calling agent
    python watcher.py --model gpt-4o   # override model

Config via .env file or env vars:
    OPENAI_API_KEY     — API key
    OPENAI_BASE_URL    — API base URL (optional)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Load .env ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val

from agent import (
    CONTROL_ROOM,
    ControlRoomContext,
    compute_diff,
    create_agent,
    process_change,
)


MIND_MAP_TEMPLATE = """\
# CONTROL ROOM

```agentsmindmap
root: CONTROL ROOM
```
"""
THINKING_MARKER = "[*] ...thinking..."

def _append_placeholder(ctx: ControlRoomContext, user_content: str) -> str:
    """Write [*] ...thinking... at correct nesting depth. Returns the exact line written."""
    # Find last user line (*) depth
    last_star_depth = 0
    for line in user_content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("*") and not stripped.startswith("[*]"):
            last_star_depth = len(line) - len(stripped)

    indent = "\t" * (last_star_depth + 1)
    placeholder = f"{indent}{THINKING_MARKER}\n"

    current = ctx.file_path.read_text()
    if not current.endswith("\n"):
        placeholder = "\n" + placeholder
    ctx.file_path.write_text(current + placeholder)
    return placeholder.strip()


def _remove_thinking_markers(ctx: ControlRoomContext) -> str:
    """Remove all [*] ...thinking... lines from the mind map. Returns cleaned content."""
    content = ctx.file_path.read_text()
    lines = content.splitlines()
    cleaned = [ln for ln in lines if THINKING_MARKER not in ln]
    result = "\n".join(cleaned) + "\n"
    ctx.file_path.write_text(result)
    return result


def run_once(
    agent,
    ctx: ControlRoomContext,
    *,
    dry_run: bool = False,
) -> None:
    """Process current state of CONTROL_ROOM.md once."""
    fpath = ctx.file_path
    if not fpath.exists():
        fpath.write_text("# CONTROL ROOM\n\n")
        print(f"[watcher] Created {fpath.name}")
        return

    content = fpath.read_text()
    context_lines = content.splitlines()[-60:]
    diff = compute_diff(ctx.last_content, content)

    print(f"[watcher] Processing {fpath.name} ({len(context_lines)} lines of context)...")
    for ln in context_lines[:10]:
        print(f"  | {ln}")
    if len(context_lines) > 10:
        print(f"  ... ({len(context_lines) - 10} more lines)")

    if diff and diff != "(no visible changes)":
        print(f"\n[watcher] Diff:\n{diff}")

    if dry_run:
        print("[watcher] Dry run — skipping agent call.")
        return

    # Show thinking placeholder, then run agent
    _append_placeholder(ctx, content)

    print("[watcher] Running agent...")
    result = process_change(agent, ctx, diff, context_lines)
    print(f"[watcher] Agent finished: {result[:300]}")

    # Clean up placeholder — keep only agent's real response
    _remove_thinking_markers(ctx)

def run_watch(
    agent,
    ctx: ControlRoomContext,
    *,
    dry_run: bool = False,
    interval: float = 1.0,
) -> None:
    """Watch the mind map file continuously, processing changes."""
    fpath = ctx.file_path
    if not fpath.exists():
        fpath.write_text("# CONTROL ROOM\n\n")

    # Initialize
    ctx.last_content = fpath.read_text()
    ctx.last_mtime = fpath.stat().st_mtime

    print(f"[watcher] Watching {fpath.name} (poll every {interval}s)...")
    print("[watcher] Edit the file to trigger the agent. Ctrl+C to stop.")
    print(f"[watcher] Agent: {agent.name} | Model: {agent.model}")

    while True:
        try:
            time.sleep(interval)

            if not fpath.exists():
                continue

            current_mtime = fpath.stat().st_mtime
            if current_mtime == ctx.last_mtime:
                continue

            # File changed — compute diff BEFORE adding placeholder
            new_content = fpath.read_text()
            diff = compute_diff(ctx.last_content, new_content)

            print(f"\n{'─' * 60}")
            print(f"[watcher] {time.strftime('%H:%M:%S')} — change detected in {fpath.name}:")
            print(diff)
            print(f"{'─' * 60}")

            # Show thinking placeholder immediately
            _append_placeholder(ctx, new_content)

            context_lines = new_content.splitlines()[-60:]

            if dry_run:
                print("[watcher] Dry run — skipping agent call.")
            else:
                print("[watcher] Running agent...")
                result = process_change(agent, ctx, diff, context_lines)
                print(f"[watcher] Done: {result[:200]}")

                # Clean up placeholder
                new_content = _remove_thinking_markers(ctx)

            # Update tracked state
            ctx.last_content = new_content
            ctx.last_mtime = fpath.stat().st_mtime

        except KeyboardInterrupt:
            print("\n[watcher] Stopped.")
            break
        except Exception as exc:
            print(f"[watcher] Error: {exc}", file=sys.stderr)
            time.sleep(2)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CONTROL ROOM watcher — LLM agent powered mind map dialogue"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="CONTROL_ROOM.md",
        help="Path to mind map file (default: CONTROL_ROOM.md). Created from template if missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show diffs without calling the agent",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process once and exit (no watching)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (default: $LLM_MODEL or gpt-4o-mini)",
    )
    args = parser.parse_args()

    # Resolve file path
    file_path = (PROJECT_ROOT / args.file).resolve()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "[watcher] OPENAI_API_KEY not set. Set it or use --dry-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create from template if missing
    if not file_path.exists():
        file_path.write_text(MIND_MAP_TEMPLATE)
        print(f"[watcher] Created {file_path.name} from template")

    agent = create_agent(model=args.model)
    ctx = ControlRoomContext(file_path=file_path)
    ctx.last_content = file_path.read_text()
    ctx.last_mtime = file_path.stat().st_mtime

    if args.once:
        run_once(agent, ctx, dry_run=args.dry_run)
    else:
        run_watch(agent, ctx, dry_run=args.dry_run, interval=args.interval)


if __name__ == "__main__":
    main()
