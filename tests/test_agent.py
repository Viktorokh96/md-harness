"""Tests for CONTROL ROOM agent — agent creation (no LLM)."""

from __future__ import annotations

from agent import create_agent


class TestCreateAgent:
    def test_creates_with_defaults(self) -> None:
        agent = create_agent()
        assert agent.name == "ControlRoom"
        assert len(agent.tools) == 8

    def test_tool_names(self) -> None:
        agent = create_agent()
        names = {t.name for t in agent.tools}
        assert names == {
            "read_mindmap",
            "add_reply",
            "batch_reply",
            "find_nodes",
            "stay_silent",
            "run_shell",
            "read_file",
            "delegate_task",
        }
