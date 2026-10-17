"""Tests for Agentic SLM and unconstrained reasoning synthesis."""

import pytest
from agent import AgenticSLM, run_agent


def test_agentic_slm_decision_and_execution():
    """Verify that AgenticSLM plans steps and executes tools autonomously."""
    class MockAgenticSLM:
        def agentic_plan(self, query, known_files):
            return "Thought: User asked for calculation", "calculate", {"expr": "25*4"}

        def agentic_answer(self, query, tool_name, tool_result, sources):
            return f"Agentic Answer: {tool_result}"

    mock_slm = MockAgenticSLM()
    ans, trace, sources = run_agent(
        "calculate 25*4",
        chunks=[],
        embeddings=None,
        use_slm=True,
        agentic_slm=mock_slm,
    )
    assert "100" in ans
    assert any(step["tool"] == "calculate" for step in trace)
    assert any(step["tool"] == "agent_reasoning" for step in trace)


def test_agentic_slm_no_strict_rules_synthesis():
    """Verify that synthesis prompt gives agency without rigid handcuffs."""
    class MockAgenticSLM(AgenticSLM):
        def __init__(self):
            pass

        def generate(self, prompt, system=None):
            assert "without rigid restrictions" in prompt or "agency" in system
            return "Here is a well-reasoned answer combining your notes and broader concepts."

    mock_slm = MockAgenticSLM()
    ans = mock_slm.agentic_answer(
        "Explain photosynthesis",
        "search_notes",
        "Found 2 chunks",
        [{"source": "bio.txt", "text": "Plants use light."}],
    )
    assert "broader concepts" in ans


def test_agentic_slm_fallback_and_conversational():
    """Verify agentic response for conversational small talk."""
    class MockAgenticSLM:
        def agentic_plan(self, query, known_files):
            return "Thought: Greeting user", "small_talk", {"reply": "Hi! Ready to help."}

        def agentic_answer(self, query, tool_name, tool_result, sources):
            return tool_result

    mock_slm = MockAgenticSLM()
    ans, trace, sources = run_agent(
        "hello there",
        chunks=[],
        embeddings=None,
        use_slm=True,
        agentic_slm=mock_slm,
    )
    assert "Ready to help" in ans
    assert len(sources) == 0
    assert trace[0]["tool"] == "agent_reasoning"
