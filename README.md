# CONTROL ROOM

LLM-powered mind map dialogue. You write in a markdown file, an AI agent reads the diff, responds in the same file.

```
* Build a weather CLI                     <- you (depth 0)
	[*] Analyzing requirements...            <- agent (depth 1)
		[*] ❓ Python or Rust?                 <- agent asks (depth 2)
			* Python                             <- you answer (depth 3)
		[*] ✅ Using Python                    <- agent confirms (depth 2)
	[*] ▶ reading pyproject.toml             <- agent acts (depth 1)
```

## Quick start

```bash
# 1. Install
pip install openai-agents>=0.18

# 2. Configure API key
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'OPENAI_BASE_URL=https://api.deepseek.com/v1' >> .env  # for DeepSeek
echo 'LLM_MODEL=deepseek-v4-flash' >> .env

# 3. Start watching
python3 watcher.py

# Then edit CONTROL_ROOM.md — the agent responds automatically.
```

## CLI

```
python3 watcher.py [file] [--once] [--dry-run] [--model MODEL] [--interval SEC]

  file          Mind map file (default: CONTROL_ROOM.md)
                Created from template if missing.
  --once        Process once and exit.
  --dry-run     Show diff, don't call LLM.
  --model       Override LLM_MODEL from .env.
  --interval    Poll interval in seconds (default: 1.0).
```

## Message format

| Prefix | Source | Example |
|---|---|---|
| `*` | User | `* Build a calculator` |
| `[*]` | Agent | `[*] Analyzing requirements...` |
| `[*] ❓` | Agent question | `[*] ❓ Python or Rust?` |
| `[*] ▶` | Agent action | `[*] ▶ grep -r "calc" .` |
| `[*] ✅` | Agent conclusion | `[*] ✅ Done — calc.py created` |

Tab characters (`\t`) create nesting. Reply one level deeper than the message you're addressing.

## Architecture

```
watcher.py          poll-based file monitor (mtime, 1s interval)
    │
    ├── change detected → compute diff
    ├── writes [*] ...thinking... placeholder
    ├── runs ControlRoom agent (OpenAI Agents SDK)
    │   ├── read_file("CONTROL_ROOM.md")  — full context
    │   ├── delegate_task → Worker agent  — Agent.as_tool()
    │   │   ├── run_shell()               — shell commands
    │   │   └── read_file()               — file inspection
    │   ├── append_to_control_room()      — write response
    │   └── stay_silent()                 — skip
    └── removes placeholder
```

### Agent tools

| Tool | Description |
|---|---|
| `append_to_control_room(text)` | Append `[*]` lines to mind map |
| `stay_silent()` | Explicit no-op |
| `run_shell(command)` | Execute shell in project root |
| `read_file(path)` | Read any project file |
| `delegate_task(task)` | Delegate to Worker sub-agent |

### Worker sub-agent

A separate `Agent` converted to a tool via `Agent.as_tool()`. Has `run_shell` + `read_file`.
Receives a task description, executes it, returns a plain-text summary.
The ControlRoom agent decides what to write to the mind map.

## Files

```
inline-vibe/
├── CONTROL_ROOM.md       # Mind map dialogue (user-owned)
├── CONTROL_ROOM_TEST.md  # Test file (for development)
├── agent.py              # Agent definition + tools (OpenAI Agents SDK)
├── watcher.py            # File watcher + CLI entry point
├── pyproject.toml        # Dependencies
├── .env                  # API keys (gitignored)
├── .env.example          # Template
├── .gitignore
├── AGENTS.md             # Instructions for LLM agents
└── README.md             # This file
```

## Environment variables

Loaded from `.env` file, overridable by environment:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (required) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `DISABLE_TRACING` | `1` | Set to `0` to enable OpenAI tracing |

## Development

```bash
# Test on a separate file
python3 watcher.py CONTROL_ROOM_TEST.md --once

# Dry run — see what the agent would see
python3 watcher.py CONTROL_ROOM_TEST.md --once --dry-run

# Watch mode
python3 watcher.py CONTROL_ROOM_TEST.md --interval 0.5
```

### Design decisions

- **Poll, not inotify**: simpler, cross-platform, no dependencies. 1-second poll is fast enough for human-paced dialogue.
- **Line-based diff, not unified**: optimized for LLM readability. Three branches: append, truncate, inline changes.
- **Agent thinks first, speaks later**: placeholder `[*] ...thinking...` gives instant feedback, then gets cleaned up.
- **Worker via `as_tool()`, not handoff**: ControlRoom stays in control. Worker does the research, ControlRoom decides what to say.
