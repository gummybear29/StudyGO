"""Ask Your Notes - Streamlit UI + Chat Application.

Stack: Streamlit, sentence-transformers (all-MiniLM-L6-v2), numpy, optional SmolLM-135M.
Free, CPU-only, under 1 GB RAM. Instant response by default, optional local SLM.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
import numpy as np

from rag import load_files, chunk_text, embed_chunks, get_embedding_model
from tools import get_store, ABSTENTION_THRESHOLD
from agent import Router, run_agent

# Page Configuration
st.set_page_config(
    page_title="Ask Your Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 2px;
        color: #1e293b;
    }

    .sub-title {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .source-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }

    .source-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
        font-weight: 600;
        color: #334155;
    }

    .source-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #dbeafe;
        color: #1d4ed8;
    }

    .step-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #22c55e;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-family: monospace;
    }

    .mode-badge-fast {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .mode-badge-slm {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .privacy-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def parse_trace_step(step_item: Any) -> tuple[str, str, str]:
    """Safely unpack an agent execution trace step whether dict, tuple, or StepRecord."""
    if isinstance(step_item, dict):
        return (
            str(step_item.get("tool", "action")),
            str(step_item.get("args", "")),
            str(step_item.get("result", "")),
        )
    elif isinstance(step_item, (list, tuple)):
        if len(step_item) == 3:
            return str(step_item[0]), str(step_item[1]), str(step_item[2])
        elif len(step_item) >= 4:
            return str(step_item[1]), str(step_item[2]), str(step_item[3])
        elif len(step_item) == 2:
            return str(step_item[0]), "", str(step_item[1])
        elif len(step_item) == 1:
            return str(step_item[0]), "", ""
    return "action", "", str(step_item)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_data" not in st.session_state:
    st.session_state.uploaded_files_data = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Sync with NoteStore singleton
store = get_store()
store.set_notes(
    st.session_state.uploaded_files_data,
    st.session_state.chunks,
    st.session_state.embeddings,
)


# Optional SLM Pipeline: Qwen/Qwen2.5-0.5B-Instruct (Recommended Local Model on CPU)
@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B-Instruct model (CPU)...")
def get_slm_model():
    """Load Qwen2.5-0.5B-Instruct on CPU."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as exc:
        return None, str(exc)


def slm_generate_answer(query: str, sources: List[Dict[str, Any]]) -> str:
    """Generate answer strictly from retrieved chunks using Qwen2.5-0.5B on CPU."""
    if not sources:
        return "I could not find information about this in your notes."

    tokenizer, model = get_slm_model()
    if tokenizer is None:
        context_preview = "\n\n".join([f"[{s['source']}]: {s['text']}" for s in sources])
        return f"(Local model unavailable: {model}. Showing retrieved text):\n\n{context_preview}"

    context_str = "\n\n".join([f"Source: {s.get('source', 'Note')}\n{s.get('text', '')}" for s in sources[:3]])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant answering questions strictly based on the provided notes context. If the answer cannot be found in the context, say 'I don't know based on the provided notes.'",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        },
    ]

    try:
        import torch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer if answer else sources[0].get("text", "")
    except Exception as exc:
        return f"Error running SLM: {exc}"


# --- SIDEBAR ---
with st.sidebar:
    st.header("📁 Notes & Knowledge Base")
    st.write("Upload `.txt` or `.pdf` files to query and summarize.")

    uploaded = st.file_uploader(
        "Upload files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # Ingest newly uploaded files
    if uploaded:
        current_names = {f.name for f in uploaded}
        cached_names = set(st.session_state.uploaded_files_data.keys())

        if current_names != cached_names:
            with st.spinner("Chunking and embedding uploaded notes..."):
                docs = load_files(uploaded)
                new_files_data = {}
                all_chunks = []

                for doc in docs:
                    new_files_data[doc["name"]] = doc["text"]
                    chunks = chunk_text(doc["text"], source_name=doc["name"], chunk_size=150, overlap=30)
                    all_chunks.extend(chunks)

                st.session_state.uploaded_files_data = new_files_data
                st.session_state.chunks = all_chunks

                if all_chunks:
                    embed_model = get_embedding_model()
                    embeddings = embed_chunks(all_chunks, model=embed_model)
                    st.session_state.embeddings = embeddings
                else:
                    st.session_state.embeddings = None

                store.set_notes(
                    st.session_state.uploaded_files_data,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                )
                st.success(f"Loaded {len(new_files_data)} file(s) and {len(all_chunks)} chunks.")

    # File and Chunk count metrics
    num_files = len(st.session_state.uploaded_files_data)
    num_chunks = len(st.session_state.chunks)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files", num_files)
    with col2:
        st.metric("Chunks", num_chunks)

    st.markdown('<div class="privacy-badge">🔒 100% Local & Private</div>', unsafe_allow_html=True)

    st.divider()

    # SLM Toggle (Off by default for instant responses)
    use_slm = st.toggle(
        "Use SLM to write answer",
        value=False,
        help="When OFF: Instant extractive retrieval (<0.1s). When ON: Qwen2.5-0.5B rewrites the answer strictly from retrieved chunks.",
    )

    if use_slm:
        st.markdown('<div class="mode-badge-slm">🤖 SLM Active: Qwen2.5-0.5B (Recommended)</div>', unsafe_allow_html=True)
        st.caption("Answers are synthesized by Qwen2.5-0.5B locally on CPU.")
    else:
        st.markdown('<div class="mode-badge-fast">⚡ Fast Mode: Instant Extractive RAG (&lt;0.1s)</div>', unsafe_allow_html=True)
        st.caption("Direct cosine retrieval with top cited passages. Answers appear instantly.")

    st.divider()

    # Action buttons: Clear chat and Clear notes
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with bcol2:
        if st.button("Clear notes", use_container_width=True):
            st.session_state.uploaded_files_data = {}
            st.session_state.chunks = []
            st.session_state.embeddings = None
            store.clear()
            st.session_state.uploader_key += 1
            st.rerun()

    st.caption("🔒 **Abstention rule**: If highest similarity is < 0.35, the agent abstains and displays no sources.")


# --- MAIN CHAT AREA ---
st.markdown('<div class="main-title">Ask Your Notes 📝</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Agentic mini-RAG with tool routing, strict abstention, and source citations.</div>', unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Sources expander
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)} cited)"):
                for idx, src in enumerate(sources, 1):
                    score = src.get("score", 0.0)
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>[{idx}] {src.get('source', 'Unknown')}</span>
                            <span class="source-pill">Score: {score:.3f}</span>
                        </div>
                        <div>{src.get('text', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Agent steps expander
        trace = msg.get("trace", [])
        if trace:
            with st.expander(f"Agent steps ({len(trace)} step{'s' if len(trace) > 1 else ''})"):
                for s_idx, step in enumerate(trace, 1):
                    tool_name, tool_args, tool_res = parse_trace_step(step)
                    st.markdown(f"""
                    <div class="step-box">
                        <strong>Step {s_idx} — Tool:</strong> {tool_name}<br/>
                        <strong>Input:</strong> {tool_args}<br/>
                        <strong>Result:</strong> {tool_res}
                    </div>
                    """, unsafe_allow_html=True)


# Chat Input
user_query = st.chat_input("Ask a question about your notes, calculate math, or summarize a file...")

if user_query:
    # 1. Display and record user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Execute Agent
    with st.chat_message("assistant"):
        spinner_msg = "Synthesizing answer with Qwen2.5-0.5B..." if use_slm else "Routing and retrieving..."
        with st.spinner(spinner_msg):
            answer, trace, sources = run_agent(
                query=user_query,
                chunks=st.session_state.chunks,
                embeddings=st.session_state.embeddings,
                max_steps=3,
                store=store,
                use_slm=use_slm,
                slm_fn=slm_generate_answer if use_slm else None,
                return_sources=True,
            )

        st.markdown(answer)

        # Render expanders for the new response
        if sources:
            with st.expander(f"Sources ({len(sources)} cited)"):
                for idx, src in enumerate(sources, 1):
                    score = src.get("score", 0.0)
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>[{idx}] {src.get('source', 'Unknown')}</span>
                            <span class="source-pill">Score: {score:.3f}</span>
                        </div>
                        <div>{src.get('text', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if trace:
            with st.expander(f"Agent steps ({len(trace)} step{'s' if len(trace) > 1 else ''})"):
                for s_idx, step in enumerate(trace, 1):
                    tool_name, tool_args, tool_res = parse_trace_step(step)
                    st.markdown(f"""
                    <div class="step-box">
                        <strong>Step {s_idx} — Tool:</strong> {tool_name}<br/>
                        <strong>Input:</strong> {tool_args}<br/>
                        <strong>Result:</strong> {tool_res}
                    </div>
                    """, unsafe_allow_html=True)

    # 3. Record assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "trace": trace,
    })
