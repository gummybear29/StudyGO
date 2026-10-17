"""Ask Your Notes - Tools Layer.

Tools available to the agent:
1. search_notes(query, chunks, embeddings) -> search with 0.35 abstention threshold
2. summarize_file(name, chunks) -> extractive summary from first 5 chunks
3. calculate(expr) -> safe arithmetic evaluator via AST (never eval)
4. list_files(chunks) -> list loaded files with chunk counts

All tools return strings.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from rag import retrieve

# Global threshold for retrieval abstention
ABSTENTION_THRESHOLD = 0.35


class NoteStore:
    """In-memory note store for chunks, embeddings, and uploaded files."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: Any = None
        self.last_sources: list[dict[str, Any]] = []

    def set_notes(self, files: dict[str, str], chunks: list[dict[str, Any]], embeddings: Any) -> None:
        self.files = files
        self.chunks = chunks
        self.embeddings = embeddings
        self.last_sources = []

    def clear(self) -> None:
        self.files = {}
        self.chunks = []
        self.embeddings = None
        self.last_sources = []


_STORE_INSTANCE = NoteStore()


def get_store() -> NoteStore:
    return _STORE_INSTANCE


# Safe operators for AST math evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> int | float:
    """Recursively and safely evaluate an AST math expression."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value)}")

    # Support Python < 3.8 ast.Num if present
    if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        return getattr(node, "n")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Disallowed operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        # Prevent oversized exponentiation denial-of-service
        if op_type is ast.Pow and (right > 1000 or left > 10000):
            raise ValueError("Exponentiation values too large")
        return _SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    # Any Call, Name, Attribute, Import, etc. is strictly rejected
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def calculate(expr: str) -> str:
    """Safely calculate a mathematical expression using AST parsing.
    
    Rejects function calls, names, variables, imports, and attributes.
    Never uses eval().
    """
    clean_expr = expr.strip()
    if not clean_expr:
        return "Error: Empty expression."

    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_ast_node(parsed.body)
        # Format integers cleanly without trailing .0 if integer
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error: Invalid or disallowed math expression ({exc})."


def list_files(chunks: list[dict[str, Any]]) -> str:
    """Return a formatted list of all uploaded files and their chunk counts."""
    if not chunks:
        return "No files are currently loaded. Please upload .txt or .pdf files in the sidebar."

    file_counts: dict[str, int] = {}
    for c in chunks:
        src = c.get("source", "Unknown")
        file_counts[src] = file_counts.get(src, 0) + 1

    lines = [f"📁 **Uploaded Files ({len(file_counts)} total):**"]
    for src, count in sorted(file_counts.items()):
        lines.append(f"- **{src}** — {count} chunk{'s' if count != 1 else ''}")

    return "\n".join(lines)


def summarize_file(
    name: str,
    chunks: list[dict[str, Any]],
    max_chunks: int = 5,
) -> str:
    """Generate an extractive summary using the first 5 chunks of the specified file."""
    if not chunks:
        return "No notes loaded. Please upload files in the sidebar first."

    name_clean = name.strip().lower()
    # Find matching chunks by exact name or substring
    matching_chunks = [
        c for c in chunks
        if c.get("source", "").lower() == name_clean
    ]
    if not matching_chunks:
        matching_chunks = [
            c for c in chunks
            if name_clean in c.get("source", "").lower() or c.get("source", "").lower() in name_clean
        ]

    available = sorted({c.get("source", "") for c in chunks})

    # If still not matched, check if name is generic or only 1 file is loaded
    if not matching_chunks:
        generic_names = ("uploaded document", "this document", "this file", "this paper", "document", "paper", "file", "notes")
        if len(available) == 1 or any(g in name_clean for g in generic_names):
            if available:
                matching_chunks = [c for c in chunks if c.get("source", "") == available[0]]

    if not matching_chunks:
        avail_str = ", ".join(f"'{a}'" for a in available) if available else "None"
        return f"File '{name}' was not found in your uploaded notes. Available files: {avail_str}."

    actual_filename = matching_chunks[0]["source"]
    selected_chunks = matching_chunks[:max_chunks]

    passages = []
    for i, c in enumerate(selected_chunks, 1):
        passages.append(f"**[Passage {i}]:** {c['text'].strip()}")

    summary_body = "\n\n".join(passages)
    return (
        f"📄 **Summary of `{actual_filename}`** (Extracted from first {len(selected_chunks)} chunks):\n\n"
        f"{summary_body}"
    )


def search_notes(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: Any,
    k: int = 3,
    threshold: float = ABSTENTION_THRESHOLD,
    model: Any | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Search notes with cosine retrieval and strict 0.35 abstention gating.
    
    Returns: (answer_string, list_of_matching_chunks)
    If highest score < threshold, returns abstention message and empty sources list.
    """
    if not chunks or embeddings is None or len(embeddings) == 0:
        return (
            "No notes have been uploaded yet. Please upload your .txt or .pdf files in the sidebar.",
            [],
        )

    results = retrieve(query, chunks, embeddings, model=model, k=k)
    if not results:
        return ("I could not find any relevant information in your notes.", [])

    top_score = results[0]["score"]

    # Abstention threshold check (0.35)
    if top_score < threshold:
        return (
            f"I could not find information about this in your notes (highest similarity score was {top_score:.2f}, which is below the 0.35 confidence threshold).",
            [],
        )

    # Build response from top retrieved chunks
    lines = [f"Based on your notes (top similarity: {top_score:.2f}):\n"]
    for i, res in enumerate(results, 1):
        lines.append(
            f"**[{i}] From `{res['source']}`** (Similarity: {res['score']:.2f}):\n"
            f"> {res['text']}\n"
        )

    return ("\n".join(lines).strip(), results)
