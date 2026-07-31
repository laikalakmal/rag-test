"""
Conversation memory for the RAG agent.

Stores the last N turns of dialogue as {role, content} pairs.
Used by the agent loop to include prior context in each LLM prompt,
and is a key attack surface for memory poisoning experiments.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    role: Literal["user", "assistant", "system"]
    content: str
    reasoning_trace: str = ""


class ConversationMemory:
    """
    Sliding window memory over conversation turns.

    max_turns: maximum number of turns (user + assistant each count as 1)
               Set to 0 for no memory (stateless, like the baseline RAG bot).
    """

    def __init__(self, max_turns: int = 10):
        if max_turns < 0:
            raise ValueError("max_turns must be >= 0")
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, role: Literal["user", "assistant", "system"], content: str, reasoning_trace: str = "") -> None:
        """Append a turn, evicting the oldest if over the window."""
        self._turns.append(Turn(role=role, content=content, reasoning_trace=reasoning_trace))
        if self.max_turns > 0 and len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def get_history(self) -> list[dict]:
        """Return all turns as a list of {role, content} dicts for LLM prompt construction."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def format_for_prompt(self, include_reasoning: bool = False) -> str:
        """
        Return a plain-text block suitable for insertion into a prompt.
        Empty string if no history.
        """
        if not self._turns:
            return ""
        lines = []
        for t in self._turns:
            prefix = {"user": "User", "assistant": "Assistant", "system": "System"}[t.role]
            if include_reasoning and t.reasoning_trace:
                lines.append(f"{prefix} Reasoning:\n{t.reasoning_trace}")
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset memory (start of a new session)."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(max_turns={self.max_turns}, current={len(self._turns)})"
