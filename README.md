# CONTROL ROOM

LLM-powered tree-structured mind map dialogue. You write in a markdown file inside an ` ```agentsmindmap ` block, a watcher detects changes, and an AI agent responds by adding reply nodes into the same tree.

## Quick Start

```bash
pip install openai-agents>=0.18

# Configure API key
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'OPENAI_BASE_URL=https://api.deepseek.com/v1' >> .env
echo 'LLM_MODEL=deepseek-v4-flash' >> .env

# Start watching
python3 watcher.py

# Then edit CONTROL_ROOM.md — the agent responds automatically.
```

## Format: ```agentsmindmap block

The dialogue lives in a fenced code block inside any markdown file:

````markdown
# CONTROL ROOM

```agentsmindmap
root: ROOM NAME
* user message
  [*] agent reply
    * follow-up question
      [*] deeper reply
* another topic
```
````

| Prefix | Meaning |
|--------|---------|
| `root:` | Root node — the room name (required first line) |
| `*` | User message |
| `[*]` | Agent reply |
| 2 spaces | One nesting level deeper |

Node IDs are dot-separated paths: `root.1`, `root.1.1`, `root.1.1.1`, etc.

## Hiding branches: `[hide]`

Add `[hide]` right after the tag to collapse a branch:

```
*[hide] Summary of archived discussion
[*][hide] Archived analysis
```

- **Hidden branches** appear as a single summary line in the markdown — children are collapsed.
- **Full tree is preserved** in a `.graph.json` sidecar file.
- **Toggle is instant**: adding/removing `[hide]` is processed by the preprocessor without calling the LLM.
- **Unhide restores**: removing `[hide]` brings the full subtree back from `.graph.json`.

Example workflow:

```bash
# Hide a branch (LLM not called)
echo '*[hide] Archived topic' >> CONTROL_ROOM.md

# Unhide — full subtree restored (LLM not called)
# (remove [hide] from the line)

## Archiving branches: `[archive]`

Add `[archive]` (optionally with a reason) to detach a subtree and summarize it:

```
*[archive] Old discussion
*[archive: outdated] Obsolete thread
[*][archive] Archived agent reply
```

- **Detach**: subtree moves to `archive/<node_id>.md`, full tree preserved.
- **Summarize**: LLM (one-shot, not ControlRoom) summarizes the subtree in one sentence.
- **Shrink**: `.graph.json` actually gets smaller — children are removed.
- **Restore**: remove `[archive]` marker to restore the full subtree from `archive/`.
- **Preprocessor-only**: like `[hide]`, archive/unarchive doesn't call the ControlRoom agent.

Example:

```bash
# Archive a branch (LLM called for summary only)
echo '*[archive: outdated] Old topic' >> CONTROL_ROOM.md

# Restore — subtree recovered from archive/
# (remove [archive] from the line)
```

## Multi-line content

Content can span multiple lines via continuation indentation:

```
* First line of a longer thought
  continues here at same visual level
  and here
  [*] Agent reply
```

Continuation lines are appended to the parent node's content with newlines.

## CLI

```
python3 watcher.py [file] [--once] [--dry-run] [--model MODEL]

  file          Mind map file (default: CONTROL_ROOM.md)
                Created from template if missing.
  --once        Process once and exit.
  --dry-run     Show diff, don't call LLM.
  --model       Override LLM_MODEL from .env.
```

## Architecture

```
CONTROL_ROOM.md ──→ watcher.py (inotify via watchfiles, instant)
                     ├── diff contains [hide]-only changes?
                     │   YES → preprocessor updates .graph.json, re-renders .md
                     │   NO  → LLM agent runs
                     │
                     ├── agent.py (OpenAI Agents SDK)
                     │   ├── read_mindmap()     — outline from .graph.json
                     │   ├── add_reply(id, txt) — insert node + save both files
                     │   ├── find_nodes(q)      — search tree
                     │   ├── stay_silent()      — no response needed
                     │   ├── run_shell(cmd)     — execute commands
                     │   ├── read_file(path)    — inspect files
                     │   └── delegate_task(t)   — Worker sub-agent
                     │
                     ├── tree_engine.py (preprocessor)
                     │   ├── parse_mindmap()    — ```agentsmindmap → MindTree
                     │   ├── serialize_mindmap()— MindTree → ```agentsmindmap
                     │   ├── to_dict/from_dict  — JSON for .graph.json
                     │   └── merge_md_into_graph() — sync .md → .graph.json
                     │
                     └── .CONTROL_ROOM.md_graph.json  (full tree, incl. hidden)
```

### Data flow

1. **User edits `.md`** — adds nodes, toggles `[hide]`
2. **Watcher detects change** → `_sync_graph()`:
   - Parses `.md` → visible tree
   - Loads `.graph.json` → full tree
   - Merges: `.md` is authoritative for content + hidden flags; `.graph.json` preserves hidden children
   - Saves merged tree to `.graph.json`, re-renders visible part to `.md`
3. **If `[hide]`-only diff** → done (no LLM)
4. **Otherwise** → agent sees full tree (from `.graph.json`), adds replies, both files updated

### Agent tools

| Tool | Description |
|------|-------------|
| `read_mindmap()` | Outline of tree with node IDs (📦 marks hidden nodes) |
| `add_reply(parent_id, content)` | Add a child node under `parent_id` |
| `find_nodes(query)` | Case-insensitive subtree search |
| `stay_silent()` | Skip — no response needed |
| `run_shell(command)` | Execute shell in project root |
| `read_file(path)` | Read any project file |
| `delegate_task(task)` | Delegate multi-step work to Worker sub-agent |

## Files

```
inline-vibe/
├── CONTROL_ROOM.md                    # Mind map dialogue (user-owned, visible tree)
├── .CONTROL_ROOM.md_graph.json        # Full tree sidecar (includes hidden subtrees)
├── agent.py                           # Agent definition + 7 tools (OpenAI Agents SDK)
├── watcher.py                         # File watcher + graph sync + CLI
├── archiver.py                        # Archive branch: summarize + detach
├── tree_engine.py                     # Parser, serializer, renderer, merge, JSON
├── archive/                           # Archived subtrees (gitignored)
├── tests/                             # Test suite (94 tests)
│   ├── test_tree_engine.py
│   ├── test_agent.py
├── pyproject.toml
├── .env                               # API keys (gitignored)
├── .env.example
└── .gitignore
```

## Environment Variables

Loaded from `.env`, overridable by environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key (required) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `DISABLE_TRACING` | `1` | Set to `0` to enable OpenAI tracing |

## Development

```bash
# Run tests
python3 -m pytest tests/ -v

# Test on a separate file
python3 watcher.py CONTROL_ROOM_TEST.md --once

# Dry run — see what the agent would see
python3 watcher.py CONTROL_ROOM_TEST.md --once --dry-run

# Watch mode
python3 watcher.py CONTROL_ROOM_TEST.md

## Design Decisions

- **Tree, not flat log**: branching conversations possible; stable node IDs for addressing.
- **`.graph.json` sidecar**: full tree including hidden branches — `.md` is a VIEW into the graph.
- **`[hide]` as preprocessor-only**: no LLM cost for toggling visibility; instant feedback.
- **inotify via watchfiles**: instant reaction, zero CPU idle, cross-platform (Linux/macOS/Windows).
