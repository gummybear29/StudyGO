# StudyGO 📝

> **Ask Your Notes**: An agentic mini-RAG application with tool routing, strict abstention gating, and privacy-first local inference.

---

## 🌟 Features

- **Document Ingestion & Chunking**: Upload `.txt` and `.pdf` documents; text is chunked into 150-word windows with 30-word overlap preserving source citations.
- **Fast Local Retrieval**: Powered by `sentence-transformers/all-MiniLM-L6-v2` and normalized cosine similarity matrix operations.
- **Agentic Routing Layer**:
  - `small_talk`: Greetings and pleasantries handled before retrieval.
  - `calculate`: Safe mathematical evaluation via AST (never `eval()`).
  - `summarize_file`: Extractive summary using the first 5 chunks of a document.
  - `list_files`: Lists all uploaded files with chunk counts.
  - `search_notes`: Answers grounded in uploaded documents.
- **Strict Abstention Gating**: If the top similarity score is `< 0.35`, the agent abstains and displays no sources to prevent hallucinations.
- **🔒 100% Local & Private**: CPU-only, zero paid APIs, under 1 GB RAM footprint.
- **Optional Local SLM**: Toggle on `Qwen2.5-0.5B-Instruct` for natural synthesis directly on CPU.

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

### 3. Run Test Suite
```bash
pytest -v
```

---

## 📁 Repository Structure

```
StudyGO/
├── app.py             # Streamlit web UI and chat interface
├── rag.py             # Document loading, chunking, embedding, and cosine retrieval
├── tools.py           # Core agent tools (calculate, search, summarize, list)
├── agent.py           # Intent router and multi-step agent execution loop
├── test_app.py        # Comprehensive test suite (16 test cases)
├── requirements.txt   # Dependencies
├── .streamlit/
│   └── config.toml    # Streamlit configuration
└── README.md          # Project documentation
```
