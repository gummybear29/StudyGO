"""Unit and Integration Tests for Ask Your Notes.

Covers:
1. chunk_text: 150 words window, 30 overlap, source preserved
2. small_talk detection: greetings and pleasantries return direct friendly reply
3. calculate: safe math evaluation and strict security rejection of __import__, eval, etc.
4. abstention threshold: scores below 0.35 abstain and yield no sources
5. route: routes summarize, calculate, list_files, small_talk, and search_notes correctly
"""

import sys
from pathlib import Path

# Add current directory to path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pytest

from agent import Router, run_agent
from rag import chunk_text, embed_chunks, retrieve
from tools import calculate, list_files, search_notes, summarize_file


# ── 1. Chunking Tests ────────────────────────────────────────────────────


def test_chunk_text_window_and_overlap():
    # 350 words text
    words = [f"word{i}" for i in range(350)]
    text = " ".join(words)
    chunks = chunk_text(text, source_name="biology_lecture.txt", chunk_size=150, overlap=30)

    assert len(chunks) == 3
    # First chunk: 150 words (0 to 150)
    assert chunks[0]["word_count"] == 150
    assert chunks[0]["source"] == "biology_lecture.txt"
    assert chunks[0]["chunk_id"] == 0

    # Second chunk starts at 150 - 30 = 120, ends at min(120+150, 350) = 270 (150 words)
    assert chunks[1]["word_count"] == 150
    assert chunks[1]["source"] == "biology_lecture.txt"
    assert chunks[1]["chunk_id"] == 1

    # Third chunk starts at 270 - 30 = 240, ends at 350 (110 words)
    assert chunks[2]["word_count"] == 110
    assert chunks[2]["source"] == "biology_lecture.txt"
    assert chunks[2]["chunk_id"] == 2


def test_chunk_text_empty():
    assert chunk_text("", source_name="empty.txt") == []
    assert chunk_text("   ", source_name="spaces.txt") == []


# ── 2. Small Talk Tests ──────────────────────────────────────────────────


def test_small_talk_detection():
    router = Router()
    greetings = ["hi", "hello", "Hey", "good morning", "thanks", "thank you", "how are you"]
    for g in greetings:
        tool, args = router.route(g)
        assert tool == "small_talk", f"Expected small_talk for '{g}', got {tool}"
        assert "reply" in args
        assert "Ask Your Notes" in args["reply"] or "Hello" in args["reply"]


def test_small_talk_run_agent():
    ans, trace, sources = run_agent(
        query="hi",
        chunks=[],
        embeddings=None,
    )
    assert "Hello" in ans or "Ask Your Notes" in ans
    assert len(sources) == 0
    assert trace[0]["tool"] == "small_talk"


# ── 3. Calculator & Security Tests ───────────────────────────────────────


def test_calculate_safe_math():
    assert calculate("12*(3+4)") == "84"
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("10 / 2") == "5"
    assert calculate("2 ** 5") == "32"
    assert calculate("15 % 4") == "3"
    assert calculate("-(5 + 3)") == "-8"


def test_calculate_division_by_zero():
    res = calculate("10 / 0")
    assert "Division by zero" in res


def test_calculate_security_rejections():
    # Security test: must NEVER execute or allow imports/calls/eval
    injections = [
        "__import__('os').system('ls')",
        "eval('2+2')",
        "os.system('whoami')",
        "open('passwords.txt')",
        "exec('print(1)')",
        "lambda x: x",
        "globals()",
        "locals()",
    ]
    for inj in injections:
        res = calculate(inj)
        assert "Error: Invalid or disallowed math expression" in res, f"Failed to reject injection: {inj}"


# ── 4. Abstention Threshold Tests (Score < 0.35) ─────────────────────────


def test_abstention_threshold():
    chunks = [
        {"chunk_id": 0, "source": "photosynthesis.txt", "text": "Chlorophyll absorbs red and blue light to convert water and carbon dioxide into oxygen and glucose."},
        {"chunk_id": 1, "source": "photosynthesis.txt", "text": "The light-dependent reactions take place on the thylakoid membranes of chloroplasts."},
    ]

    # Mock embeddings where similarity is low (< 0.35)
    # Using normalized orthogonal vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    class MockModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # Query vector is orthogonal with a very tiny component: dot product = 0.1 (< 0.35)
            return np.array([[0.1, 0.1, 0.9, 0.0]], dtype=np.float32)

    ans, sources = search_notes(
        query="What is the capital of Peru?",
        chunks=chunks,
        embeddings=embeddings,
        threshold=0.35,
        model=MockModel(),
    )

    # Must abstain
    assert "below the 0.35 confidence threshold" in ans or "could not find" in ans
    assert len(sources) == 0


def test_high_score_answer():
    chunks = [
        {"chunk_id": 0, "source": "capital.txt", "text": "The capital of France is Paris, located on the Seine River."},
    ]

    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    class MockModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # High similarity: dot product = 0.95 (>= 0.35)
            return np.array([[0.95, 0.05, 0.0]], dtype=np.float32)

    ans, sources = search_notes(
        query="What is the capital of France?",
        chunks=chunks,
        embeddings=embeddings,
        threshold=0.35,
        model=MockModel(),
    )

    # Must succeed and return sources
    assert len(sources) == 1
    assert "Paris" in ans
    assert sources[0]["score"] >= 0.35


# ── 5. Router Tests ──────────────────────────────────────────────────────


def test_router_all_tools():
    router = Router(known_files=["notes.txt", "lecture.pdf"])

    # 1. Small talk
    tool, args = router.route("hello there")
    assert tool == "small_talk"

    # 2. Calculate
    tool, args = router.route("what is 12*(3+4)")
    assert tool == "calculate"
    assert args["expr"] == "12*(3+4)"

    tool, args = router.route("calculate 25 * 4")
    assert tool == "calculate"
    assert "25 * 4" in args["expr"]

    # 3. Summarize file
    tool, args = router.route("summarize notes.txt")
    assert tool == "summarize_file"
    assert args["name"] == "notes.txt"

    # 4. List files
    tool, args = router.route("what files are uploaded?")
    assert tool == "list_files"

    tool, args = router.route("list files")
    assert tool == "list_files"

    # 5. Search notes (default)
    tool, args = router.route("What are the key themes of chapter 3?")
    assert tool == "search_notes"
    assert "chapter 3" in args["query"]


# ── 6. End-to-End Agent Execution ────────────────────────────────────────


def test_run_agent_summarize():
    chunks = [
        {"chunk_id": 0, "source": "history.txt", "text": "The Renaissance began in Florence in the 14th century."},
        {"chunk_id": 1, "source": "history.txt", "text": "It was marked by a resurgence of interest in classical antiquity."},
    ]
    ans, trace, sources = run_agent("summarize history.txt", chunks=chunks, embeddings=None)
    assert "Summary of `history.txt`" in ans
    assert "Renaissance" in ans
    assert trace[0]["tool"] == "summarize_file"


def test_run_agent_calculate():
    ans, trace, sources = run_agent("what is 12*(3+4)", chunks=[], embeddings=None)
    assert "84" in ans
    assert trace[0]["tool"] == "calculate"


def test_run_agent_list_files():
    chunks = [
        {"chunk_id": 0, "source": "doc1.txt", "text": "text1"},
        {"chunk_id": 1, "source": "doc1.txt", "text": "text2"},
        {"chunk_id": 2, "source": "doc2.pdf", "text": "text3"},
    ]
    ans, trace, sources = run_agent("what files do I have?", chunks=chunks, embeddings=None)
    assert "doc1.txt" in ans
    assert "doc2.pdf" in ans
    assert trace[0]["tool"] == "list_files"
