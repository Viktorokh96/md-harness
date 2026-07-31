"""Tests for CONTROL ROOM agent — agent creation (no LLM)."""

from __future__ import annotations

import os

import pytest

from agent import create_agent


class TestCreateAgent:
    def test_creates_with_defaults(self) -> None:
        agent = create_agent()
        assert agent.name == "ControlRoom"
        assert len(agent.tools) == 7

    def test_tool_names(self) -> None:
        agent = create_agent()
        names = {t.name for t in agent.tools}
        assert names == {
            "read_mindmap",
            "add_reply",
            "find_nodes",
            "stay_silent",
            "run_shell",
            "read_file",
            "delegate_task",
        }

    def test_model_override(self) -> None:
        agent = create_agent(model="custom-model")
        assert agent.model == "custom-model"

    def test_model_from_env(self) -> None:
        os.environ["LLM_MODEL"] = "env-model"
        try:
            agent = create_agent()
            assert agent.model == "env-model"
        finally:
            del os.environ["LLM_MODEL"]

    def test_explicit_model_wins_over_env(self) -> None:
        os.environ["LLM_MODEL"] = "env-model"
        try:
            agent = create_agent(model="explicit-model")
            assert agent.model == "explicit-model"
        finally:
            del os.environ["LLM_MODEL"]
