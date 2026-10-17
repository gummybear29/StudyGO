"""Ask Your Notes - Agentic Routing & Execution Layer.

Features:
- Router class with rule-based route(query) -> (tool_name, args)
- Topic / Overview intent routing ("what is the topic about")
- AgenticSLM class powered by local SmolLM-135M on CPU
- run_agent(query, chunks, embeddings, max_steps=3, slm_fn=None, ...)
  - Returns (final_answer, trace, sources)
"""

from __future__ import annotations

import re
from typing import Any, Callable

from tools import calculate, list_files, search_notes, summarize_file


class Router:
    """Rule-based tool router that classifies user intent into tool calls."""

    def __init__(self, known_files: list[str] | None = None) -> None:
        self.known_files = known_files or []

    def route(
        self,
        query: str,
        file_names: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Determine which tool and arguments to execute for a given query."""
        q_clean = query.strip()
        q_lower = q_clean.lower()
        files = file_names or self.known_files

        # 1. Small Talk Detection (Checked before retrieval)
        small_talk_patterns = (
            r"^(hi|hello|hey|hola|greetings|howdy)(\s+(there|friend|assistant|bot|everyone))?[\s\W]*$"
            r"|^good\s+(morning|afternoon|evening|day)[\s\W]*$"
            r"|^how\s+are\s+you(\s+doing)?[\s\W]*$"
            r"|^(thanks|thank\s+you|thx)[\s\W]*$"
            r"|^(who\s+are\s+you|what\s+can\s+you\s+do)[\s\W]*$"
            r"|^(bye|goodbye|see\s+you)[\s\W]*$"
        )
        if re.search(small_talk_patterns, q_lower) or q_lower in ("hi", "hello", "hey", "thanks", "thank you", "how are you"):
            return (
                "small_talk",
                {
                    "reply": (
                        "Hello! 👋 I'm your **Ask Your Notes** assistant. "
                        "Upload your `.txt` or `.pdf` notes in the sidebar, and I can search them, "
                        "summarize documents, calculate math, or list your files!"
                    ),
                },
            )

        # 2. File Listing Intent
        if re.search(r"\b(what\s+files|list\s+files|show\s+files|which\s+files|available\s+files|files\s+uploaded|my\s+notes)\b", q_lower):
            return ("list_files", {})

        # 3. Document Topic / Overview / Summarization Intent
        sum_match = re.search(r"\b(summarize|summary\s+of)\s+([a-zA-Z0-9_\-./\\]+(\.(txt|pdf))?|\w+)", q_clean, re.IGNORECASE)
        topic_match = re.search(
            r"\b("
            r"what\s+(?:is|are)\s+(?:the\s+|this\s+)?(?:topic|subject|paper|doc|document|file|notes?|content|pdf)"
            r"|what\s+is\s+(?:this|it)\s+(?:all\s+)?about"
            r"|what\s+is\s+(?:the\s+|this\s+)?(?:note|document|paper|file)\s+about"
            r"|what\s+does\s+this\s+(?:paper|doc|document|file|note)\s+say"
            r"|what\s+do\s+(?:the\s+)?notes\s+say"
            r"|what(?:'s|\s+is)\s+in\s+(?:the\s+|this\s+)?(?:notes?|doc|document|paper|pdf|file)"
            r"|tell\s+me\s+about\s+(?:this\s+|the\s+)?(?:file|doc|paper|document|note|notes|pdf|topic)"
            r"|explain\s+(?:this\s+|the\s+)?(?:file|doc|paper|document|note|notes|pdf)"
            r"|give\s+(?:me\s+)?(?:an\s+)?overview"
            r"|overview|summarize|summary"
            r")\b",
            q_lower,
        )
        if sum_match:
            candidate_file = sum_match.group(2).strip()
            if files:
                for f in files:
                    if candidate_file.lower() in f.lower() or f.lower() in candidate_file.lower():
                        candidate_file = f
                        break
            return ("summarize_file", {"name": candidate_file})
        elif topic_match:
            candidate_file = files[0] if files else "uploaded document"
            return ("summarize_file", {"name": candidate_file})

        # 4. Calculator / Math Intent
        math_match = re.search(r"^(calculate|what\s+is\s+the\s+result\s+of|what\s+is|solve)?\s*([-+*/0-9().\s^%//]+)$", q_clean, re.IGNORECASE)
        if math_match:
            expr_candidate = math_match.group(2).strip()
            has_digit = any(c.isdigit() for c in expr_candidate)
            has_op = any(c in "+-*/%^" for c in expr_candidate)
            if has_digit and (has_op or math_match.group(1)):
                return ("calculate", {"expr": expr_candidate})

        # 5. Default Fallback -> Search Notes
        return ("search_notes", {"query": q_clean})


# ── Agentic SLM Layer (Qwen2.5-0.5B-Instruct on CPU - Recommended) ──────

_SLM_SINGLETON: AgenticSLM | None = None


class AgenticSLM:
    """Agentic Small Language Model that runs locally on CPU."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 150,
        temperature: float = 0.3,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _init(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Run text generation on CPU using Hugging Face chat templates."""
        try:
            self._init()
            import torch

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([input_text], return_tensors="pt")

            if inputs["input_ids"].shape[1] > 2048:
                inputs["input_ids"] = inputs["input_ids"][:, -2048:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -2048:]

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=max(0.1, self.temperature),
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            gen_tokens = outputs[:, inputs.input_ids.shape[1] :]
            return self._tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            return f"(SLM Note: {exc})"

    def agentic_plan(
        self,
        query: str,
        known_files: list[str],
    ) -> tuple[str, str, dict[str, Any]]:
        heuristic_router = Router(known_files=known_files)
        h_tool, h_args = heuristic_router.route(query, file_names=known_files)
        return "Determined tool via intent analysis.", h_tool, h_args

    def agentic_answer(
        self,
        query: str,
        tool_name: str,
        tool_result: str,
        sources: list[dict[str, Any]],
    ) -> str:
        if not sources:
            return tool_result

        context_parts = []
        for s in sources:
            src = s.get("source", "document")
            text = s.get("text", "")
            context_parts.append(f"[{src}]: {text}")
        context_str = "\n\n".join(context_parts)

        prompt = (
            f"User Question: {query}\n\n"
            f"Grounded Document Context:\n{context_str}\n\n"
            f"Provide a natural, helpful, and insightful response answering the user question based on the document context without rigid restrictions."
        )
        system = (
            "You are an intelligent, helpful AI assistant with full agency. Synthesize a clear and accurate answer based on the provided document context."
        )
        ans = self.generate(prompt=prompt, system=system)
        return ans.strip() if ans and len(ans) > 15 else tool_result


def get_agentic_slm(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> AgenticSLM:
    """Retrieve or initialize the cached AgenticSLM instance."""
    global _SLM_SINGLETON
    if _SLM_SINGLETON is None:
        _SLM_SINGLETON = AgenticSLM(model_name=model_name)
    return _SLM_SINGLETON


# ── Agent Loop ─────────────────────────────────────────────────────────────


def run_agent(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    embeddings: Any = None,
    max_steps: int = 3,
    slm_fn: Callable[[str, list[dict[str, Any]]], str] | None = None,
    model: Any | None = None,
    router: Router | None = None,
    use_slm: bool = True,
    agentic_slm: AgenticSLM | Any | None = None,
    store: Any = None,
    return_sources: bool = True,
) -> Any:
    """Execute the agent loop for a user query.
    
    Returns:
    - final_answer: str
    - trace: list of execution steps
    - sources: list of retrieved chunk dicts
    """
    if store is not None:
        if chunks is None:
            chunks = getattr(store, "chunks", [])
        if embeddings is None:
            embeddings = getattr(store, "embeddings", None)

    if chunks is None:
        chunks = []

    known_files = list({c.get("source", "") for c in chunks})
    trace: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    final_answer = ""
    current_query = query.strip()

    active_router = router or Router(known_files=known_files)

    for step in range(1, max_steps + 1):
        if agentic_slm is not None and hasattr(agentic_slm, "agentic_plan"):
            thought, tool_name, args = agentic_slm.agentic_plan(current_query, known_files)
            trace.append({
                "step": len(trace) + 1,
                "tool": "agent_reasoning",
                "args": {"thought": thought},
                "result": f"Planned tool: {tool_name}",
            })
        else:
            tool_name, args = active_router.route(current_query, file_names=known_files)

        if tool_name == "small_talk":
            reply_text = args.get("reply") or "Hello! 👋 I'm your **Ask Your Notes** assistant. Upload your `.txt` or `.pdf` notes in the sidebar, and I can search them, summarize documents, calculate math, or list your files!"
            if agentic_slm is not None and hasattr(agentic_slm, "agentic_answer"):
                final_answer = agentic_slm.agentic_answer(current_query, "small_talk", reply_text, sources)
            else:
                final_answer = reply_text
            trace.append({
                "step": len(trace) + 1,
                "tool": "small_talk",
                "args": args,
                "result": final_answer,
            })
            break

        elif tool_name == "calculate":
            calc_result = calculate(args["expr"])
            calc_formatted = f"🔢 **Calculation Result:** `{calc_result}`"
            if agentic_slm is not None and hasattr(agentic_slm, "agentic_answer"):
                final_answer = agentic_slm.agentic_answer(current_query, "calculate", str(calc_result), sources)
            else:
                final_answer = calc_formatted
            trace.append({
                "step": len(trace) + 1,
                "tool": "calculate",
                "args": args,
                "result": calc_result,
            })
            break

        elif tool_name == "list_files":
            files_result = list_files(chunks)
            if agentic_slm is not None and hasattr(agentic_slm, "agentic_answer"):
                final_answer = agentic_slm.agentic_answer(current_query, "list_files", files_result, sources)
            else:
                final_answer = files_result
            trace.append({
                "step": len(trace) + 1,
                "tool": "list_files",
                "args": args,
                "result": files_result,
            })
            break

        elif tool_name == "summarize_file":
            summary_result = summarize_file(args.get("name", ""), chunks)
            target_name = args.get("name", "").lower()
            matching = [c for c in chunks if target_name in c.get("source", "").lower() or c.get("source", "").lower() in target_name]
            if not matching and chunks:
                matching = chunks[:5]
            sources = matching[:5]
            trace.append({
                "step": len(trace) + 1,
                "tool": "summarize_file",
                "args": args,
                "result": f"Extracted overview chunks from {args.get('name', 'file')}.",
            })
            if agentic_slm is not None and hasattr(agentic_slm, "agentic_answer"):
                final_answer = agentic_slm.agentic_answer(current_query, "summarize_file", summary_result, sources)
            elif slm_fn is not None and sources:
                try:
                    final_answer = slm_fn(current_query, sources)
                except Exception:
                    final_answer = summary_result
            else:
                final_answer = summary_result
            break

        elif tool_name == "search_notes":
            search_query = args.get("query", current_query)
            search_ans, matched_sources = search_notes(
                query=search_query,
                chunks=chunks,
                embeddings=embeddings,
                k=3,
                threshold=0.35,
                model=model,
            )
            sources = matched_sources
            trace.append({
                "step": len(trace) + 1,
                "tool": "search_notes",
                "args": args,
                "result": f"Retrieved {len(sources)} source chunk(s) (Abstained: {len(sources) == 0}).",
            })

            # Intelligent fallback: If search abstained but query clearly asks about notes/documents
            if not sources and chunks and any(w in current_query.lower() for w in ["note", "notes", "document", "paper", "pdf", "file", "topic", "about"]):
                target_name = (known_files[0] if known_files else "").lower()
                matching = [c for c in chunks if target_name in c.get("source", "").lower()] or chunks[:5]
                sources = matching[:5]
                trace.append({
                    "step": len(trace) + 1,
                    "tool": "summarize_file",
                    "args": {"name": known_files[0] if known_files else "uploaded document"},
                    "result": "Fallback engaged: Retrieved top overview chunks from notes.",
                })

            # Grounded SLM synthesis on retrieved sources
            if sources:
                if agentic_slm is not None and hasattr(agentic_slm, "agentic_answer"):
                    final_answer = agentic_slm.agentic_answer(search_query, "search_notes", search_ans, sources)
                elif slm_fn is not None:
                    try:
                        final_answer = slm_fn(search_query, sources)
                    except Exception:
                        final_answer = search_ans
                else:
                    final_answer = search_ans
            else:
                final_answer = search_ans
            break

        else:
            final_answer = "I'm not sure how to handle this request."
            trace.append({
                "step": len(trace) + 1,
                "tool": "unknown",
                "args": args,
                "result": final_answer,
            })
            break

    if store is not None:
        store.last_sources = sources
    if not return_sources:
        return final_answer, trace
    return final_answer, trace, sources
