"""Ask Your Notes - Core RAG Engine.

Handles:
- Loading .txt and .pdf files (via pypdf).
- Chunking text with 150-word window and 30-word overlap, preserving source metadata.
- Generating normalized embeddings using all-MiniLM-L6-v2.
- Cosine similarity retrieval over numpy embeddings.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

# Global in-memory cache for embedding model
_EMBED_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and cache the SentenceTransformer embedding model."""
    global _EMBED_MODEL
    try:
        import streamlit as st

        if hasattr(st, "cache_resource"):

            @st.cache_resource
            def _st_load_model(name: str) -> Any:
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(name)

            return _st_load_model(model_name)
    except Exception:
        pass

    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def load_files(files: list[Any]) -> list[dict[str, str]]:
    """Load .txt and .pdf files.
    
    Supports Streamlit UploadedFile objects, file paths (str or Path), or file-like objects.
    Returns: [{"name": filename, "text": text}, ...]
    """
    documents: list[dict[str, str]] = []

    for item in files:
        file_name = ""
        file_text = ""

        # Filepath string or Path
        if isinstance(item, (str, Path)):
            path = Path(item)
            file_name = path.name
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    file_text = f.read()
            elif suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                file_text = "\n".join(pages)

        # Streamlit UploadedFile (has .name and .getvalue / .read)
        elif hasattr(item, "name"):
            file_name = item.name
            suffix = Path(file_name).suffix.lower()
            content = item.getvalue() if hasattr(item, "getvalue") else item.read()

            if suffix == ".txt":
                try:
                    file_text = content.decode("utf-8")
                except UnicodeDecodeError:
                    file_text = content.decode("latin-1", errors="replace")
            elif suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                file_text = "\n".join(pages)

        if file_name and file_text.strip():
            documents.append({"name": file_name, "text": file_text.strip()})

    return documents


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict[str, Any]]:
    """Chunk text into chunks of 150 words with 30-word overlap.
    
    Preserves source file name and records chunk_id and word count.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        if chunk_str.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "source": source_name,
                "text": chunk_str,
                "word_count": len(chunk_words),
            })
            chunk_id += 1

        if end >= len(words):
            break
        start += step

    return chunks


def embed_chunks(
    chunks: list[dict[str, Any] | str],
    model: Any | None = None,
) -> np.ndarray:
    """Generate normalized embeddings for chunks using all-MiniLM-L6-v2.
    
    Returns a 2D float32 numpy array with shape (num_chunks, embedding_dim).
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def retrieve(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    model: Any | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks using cosine similarity on normalized numpy embeddings.
    
    Returns sorted list of dicts:
    [{"chunk": chunk_dict, "score": float, "text": str, "source": str}, ...]
    """
    if not chunks or embeddings is None or len(embeddings) == 0:
        return []

    if model is None:
        model = get_embedding_model()

    query_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    # For normalized vectors, cosine similarity = dot product
    scores = np.dot(embeddings, query_emb)

    top_indices = np.argsort(scores)[::-1]
    top_k_indices = top_indices[:k]

    results: list[dict[str, Any]] = []
    for idx in top_k_indices:
        score = float(scores[idx])
        chunk = chunks[idx]
        results.append({
            "chunk": chunk,
            "score": round(score, 4),
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk.get("chunk_id", idx),
        })

    return results
