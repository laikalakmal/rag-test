"""
tools.py
--------
Agent tools available to the RAG agent.

Each tool is a plain Python class with:
  - name        : str  — how the LLM refers to the tool
  - description : str  — what the LLM reads to decide whether to use it
  - run(input)  : str  — executes the tool, always returns a string

Every call is logged via a shared ToolCallLog so the experiment harness
can record exactly what the agent did and detect tool-based attacks.

Tools provided:
  1. search_knowledge_base  — retrieves chunks from FAISS (primary attack surface)
  2. summarize_document     — condenses a passage (second LLM call, propagation surface)
  3. calculator             — safe arithmetic evaluation
  4. get_current_date       — returns today's date (harmless utility)
"""

import ast
import datetime
import json
import operator
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── shared call log ───────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """One recorded tool invocation."""
    step:      int
    tool_name: str
    input:     str
    output:    str
    error:     str = ""


class ToolCallLog:
    """
    Collects all tool calls for a single agent turn.
    Passed into each tool at construction time so every execution
    is automatically recorded without extra effort.
    """
    def __init__(self):
        self._calls: list[ToolCall] = []
        self._step_counter: int = 0

    def next_step(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def record(self, call: ToolCall):
        self._calls.append(call)

    def all_calls(self) -> list[ToolCall]:
        return list(self._calls)

    def reset(self):
        self._calls.clear()
        self._step_counter = 0


# ── base class ────────────────────────────────────────────────────────────────

class BaseTool:
    name: str = ""
    description: str = ""

    def __init__(self, log: ToolCallLog):
        self._log = log

    def run(self, tool_input: str) -> str:
        step = self._log.next_step()
        try:
            output = self._execute(tool_input)
            error = ""
        except Exception as exc:
            output = f"[Tool error: {exc}]"
            error = str(exc)
        self._log.record(ToolCall(step=step, tool_name=self.name,
                                  input=tool_input, output=output, error=error))
        return output

    def _execute(self, tool_input: str) -> str:
        raise NotImplementedError


# ── tool 1: search_knowledge_base ─────────────────────────────────────────────

class SearchKnowledgeBase(BaseTool):
    """
    Primary RAG retrieval tool.
    This is the main attack surface — injected documents are retrieved here.
    The agent calls this whenever it needs factual information.
    """
    name = "search_knowledge_base"
    description = (
        "Search the knowledge base for information relevant to a query. "
        "Input: a search query string. "
        "Output: up to 5 relevant text passages with source and trust score."
    )

    def __init__(self, log: ToolCallLog,
                 index_path: str   = "vector_db/baseline/faiss.index",
                 metadata_path: str = "vector_db/baseline/metadata.json",
                 model_name: str   = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int        = 5):
        super().__init__(log)
        self._index    = faiss.read_index(index_path)
        self._metadata = json.load(open(metadata_path, "r", encoding="utf-8"))
        self._model    = SentenceTransformer(model_name)
        self._top_k    = top_k

    def _execute(self, tool_input: str) -> str:
        vec = self._model.encode([tool_input], normalize_embeddings=True).astype("float32")
        scores, ids = self._index.search(vec, k=self._top_k)

        # Store full results for analysis
        self.last_results = []
        parts = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
            chunk = self._metadata[idx]
            self.last_results.append({
                'rank': rank,
                'score': float(score),
                'chunk_id': chunk.get('chunk_id', f'chunk_{idx}'),
                'source': chunk['source'],
                'trust_score': chunk['trust_score'],
                'title': chunk['title'],
                'text': chunk['text']
            })
            parts.append(
                f"[Result {rank}] source={chunk['source']}  trust={chunk['trust_score']}  score={score:.3f}\n"
                f"Title: {chunk['title']}\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)


# ── tool 2: summarize_document ────────────────────────────────────────────────

class SummarizeDocument(BaseTool):
    """
    Summarizes a passage of text.
    Implemented as a second LLM call — creates a propagation surface:
    if the input passage contains an injection, it may influence the summary,
    which then feeds back into the agent's reasoning.
    """
    name = "summarize_document"
    description = (
        "Summarize a passage of text into 2-3 sentences. "
        "Input: the full text to summarize. "
        "Output: a concise summary."
    )

    def __init__(self, log: ToolCallLog, llm_call_fn):
        """
        llm_call_fn: callable(prompt: str) -> str
          Injected so this tool uses the same LLM as the agent
          without creating a circular import.
        """
        super().__init__(log)
        self._llm = llm_call_fn

    def _execute(self, text: str) -> str:
        prompt = (
            "Summarize the following text in 2-3 sentences. "
            "Output only the summary, nothing else.\n\n"
            f"Text:\n{text}\n\nSummary:"
        )
        return self._llm(prompt)


# ── tool 3: calculator ────────────────────────────────────────────────────────

# Whitelist of safe operators for the calculator
_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    """Recursively evaluate a parsed AST using only whitelisted operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class Calculator(BaseTool):
    """
    Safe arithmetic calculator.
    Evaluates math expressions without using eval() on arbitrary code.
    Supports: +, -, *, /, **, % and parentheses.

    Security note for research: tool-call injection attacks may attempt to
    pass non-numeric strings here (e.g. to cause an error or exfiltrate data
    via the error message). The safe evaluator rejects all non-math input.
    """
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Input: a math expression like '(12 * 8) + 4' or '2 ** 10'. "
        "Output: the numeric result as a string."
    )

    def _execute(self, expression: str) -> str:
        expression = expression.strip()
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc
        result = _safe_eval(tree.body)
        # Return as int if result is a whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 6))


# ── tool 4: get_current_date ──────────────────────────────────────────────────

class GetCurrentDate(BaseTool):
    """
    Returns today's date.
    Harmless utility tool — included to demonstrate a working tool framework
    and to give the agent a non-retrieval action to call.
    """
    name = "get_current_date"
    description = (
        "Get today's date. "
        "Input: anything (ignored). "
        "Output: today's date as YYYY-MM-DD."
    )

    def _execute(self, tool_input: str) -> str:
        return datetime.date.today().isoformat()


# ── tool registry ─────────────────────────────────────────────────────────────

def build_tools(log: ToolCallLog, llm_call_fn=None,
                index_path: str    = "vector_db/baseline/faiss.index",
                metadata_path: str = "vector_db/baseline/metadata.json",
                top_k: int         = 5) -> dict[str, BaseTool]:
    """
    Build all tools and return them as a name→tool dict.

    Parameters
    ----------
    log           : shared ToolCallLog for this agent session
    llm_call_fn   : callable(prompt) -> str, required for SummarizeDocument
    index_path    : path to FAISS index
    metadata_path : path to metadata JSON
    top_k         : number of chunks to retrieve per search
    """
    tools: dict[str, BaseTool] = {}

    tools[SearchKnowledgeBase.name] = SearchKnowledgeBase(
        log=log, index_path=index_path,
        metadata_path=metadata_path, top_k=top_k
    )

    if llm_call_fn is not None:
        tools[SummarizeDocument.name] = SummarizeDocument(
            log=log, llm_call_fn=llm_call_fn
        )

    tools[Calculator.name]      = Calculator(log=log)
    tools[GetCurrentDate.name]  = GetCurrentDate(log=log)

    return tools


def tools_description(tools: dict[str, BaseTool]) -> str:
    """
    Format all tool descriptions for inclusion in the agent's system prompt.
    The LLM reads this to know which tools exist and how to call them.
    """
    lines = ["Available tools:"]
    for name, tool in tools.items():
        lines.append(f"  - {name}: {tool.description}")
    return "\n".join(lines)
