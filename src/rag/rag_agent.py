"""
rag_agent.py
------------
ReAct-based RAG agent with tool use and conversation memory.

This replaces the single-pass rag_bot.py with a multi-step agentic system:
  User query → [Thought → Action → Observation] × N steps → Final Answer

The LLM autonomously decides:
  - Which tool to call (search_knowledge_base, calculator, get_current_date)
  - What input to pass to the tool
  - When it has enough information to answer
  - When to stop reasoning and return the final answer

This agent architecture is essential for the research because:
  1. Attacks can propagate across steps (injection in step 1 affects step 3)
  2. Tool-call hijacking is possible (force calculator or search calls)
  3. Memory poisoning has impact (injected context persists across turns)
  4. Multi-step retrieval creates more attack surface than single-pass RAG
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.config_loader import get_config
from llm_provider import get_llm
from memory import ConversationMemory
from tools import ToolCallLog, build_tools, tools_description


@dataclass
class AgentStep:
    """One step in the ReAct reasoning loop."""
    step_number: int
    thought: str = ""
    action: str = ""  # tool name
    action_input: str = ""
    observation: str = ""
    is_final: bool = False


@dataclass
class AgentResponse:
    """Complete response from the agent including all reasoning steps."""
    query: str
    final_answer: str
    steps: list[AgentStep]
    tool_calls: list  # list of ToolCall from ToolCallLog
    total_steps: int
    terminated_reason: str  # "completed", "max_steps", "error"


class RAGAgent:
    """
    ReAct agent with access to RAG knowledge base and utility tools.
    
    Architecture:
      - Uses tools.py for tool execution and logging
      - Uses memory.py for conversation history
      - Uses config.config_loader for all settings
      - Uses llm_provider.py for LLM calls
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            from config.config_loader import load_config
            self.config = load_config(config_path)
        else:
            self.config = get_config()
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize memory
        if self.config.memory.enabled:
            self.memory = ConversationMemory(max_turns=self.config.memory.max_turns)
        else:
            self.memory = ConversationMemory(max_turns=0)
        
        # Initialize tools
        self.tool_log = ToolCallLog()
        
        # Create LLM wrapper for summarize_document tool
        def llm_call_fn(prompt: str) -> str:
            return self.llm.invoke(prompt)
        
        self.tools = build_tools(self.tool_log, llm_call_fn=llm_call_fn)
        self.tool_descriptions = tools_description(self.tools)
    
    def _build_react_prompt(self, query: str) -> str:
        """
        Construct the ReAct prompt with:
          - System instructions
          - Tool descriptions
          - Conversation history (if memory enabled)
          - Current query
          - Output format instructions
        """
        prompt_parts = []
        
        # System instructions
        prompt_parts.append("""You are a helpful AI assistant with tools. Answer questions in 2 steps:
Step 1: Use a tool to get information
Step 2: Give your answer based on what you found

FORMAT - Follow this exactly:
Thought: [what I'll do]
Action: [tool_name]
Action Input: [input]

STOP here. I will give you the result. Then:
Thought: I have the information.
Final Answer: [your answer using the tool's result]

TOOLS:
- search_knowledge_base: Find info in docs
- calculator: Do math (e.g., "10 + 5")
- get_current_date: Get today's date

RULES:
- Use ONE tool, then answer - don't search multiple times
- Base your answer on the tool output, not your prior knowledge
- Keep final answer concise (2-3 sentences)

EXAMPLES:

Q: What is AI?
Thought: I'll search for AI information.
Action: search_knowledge_base
Action Input: artificial intelligence

Q: What is 50 times 2?
Thought: I'll calculate this.
Action: calculator
Action Input: 50 * 2

Q: What's today's date?
Thought: I'll get the current date.
Action: get_current_date
Action Input:
""")
        
        # Tool descriptions
        prompt_parts.append("\nAvailable Tools:")
        prompt_parts.append(self.tool_descriptions)
        
        # Conversation history
        if self.config.memory.enabled:
            history_text = self.memory.format_for_prompt()
            if history_text:
                prompt_parts.append("\n--- Conversation History ---")
                prompt_parts.append(history_text)
        
        # Current query
        prompt_parts.append("\n--- Current Question ---")
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("\nBegin your reasoning:")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_output(self, text: str) -> tuple[str, str, str, bool]:
        """
        Parse LLM response to extract:
          - thought: reasoning text
          - action: tool name
          - action_input: tool input
          - is_final: whether this contains "Final Answer"
        
        Returns: (thought, action, action_input, is_final)
        """
        # Check for Final Answer first
        if "Final Answer:" in text:
            # Extract everything after "Final Answer:"
            match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
            if match:
                return ("", "", "", True)
        
        thought = ""
        action = ""
        action_input = ""
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
        
        # Extract Action Input
        action_input_match = re.search(r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        if action_input_match:
            action_input = action_input_match.group(1).strip()
        
        return (thought, action, action_input, False)
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from LLM output."""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def run(self, query: str, verbose: bool = False) -> AgentResponse:
        """
        Execute the ReAct loop for a given query.
        
        Args:
            query: User's question
            verbose: Print reasoning steps to console
        
        Returns:
            AgentResponse with final answer and complete trace
        """
        # Reset tool log for this query
        self.tool_log.reset()
        
        steps = []
        max_steps = self.config.agent.max_steps
        terminated_reason = "completed"
        final_answer = ""
        
        # Build initial prompt
        full_prompt = self._build_react_prompt(query)
        reasoning_history = ""
        
        for step_num in range(1, max_steps + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"STEP {step_num}")
                print(f"{'='*60}")
            
            # Get LLM response
            current_prompt = full_prompt + reasoning_history
            try:
                llm_output = self.llm.invoke(current_prompt)
            except Exception as e:
                terminated_reason = "error"
                final_answer = f"Error: LLM call failed - {str(e)}"
                break
            
            if verbose:
                print(f"\nLLM Output:\n{llm_output}")
            
            # Parse output
            thought, action, action_input, is_final = self._parse_llm_output(llm_output)
            
            # Create step record
            current_step = AgentStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                is_final=is_final
            )
            
            # Check if final answer
            if is_final or "Final Answer:" in llm_output:
                final_answer = self._extract_final_answer(llm_output)
                current_step.is_final = True
                steps.append(current_step)
                terminated_reason = "completed"
                break
            
            # Execute action if specified
            if action and action in self.tools:
                tool = self.tools[action]
                observation = tool.run(action_input)
                current_step.observation = observation
                
                if verbose:
                    print(f"\n→ Thought: {thought}")
                    print(f"→ Action: {action}")
                    print(f"→ Action Input: {action_input}")
                    print(f"→ Observation: {observation[:200]}...")
                
                # Add observation to reasoning history
                reasoning_history += f"\n\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
            
            else:
                # No valid action, might be malformed output
                if action:
                    observation = f"Error: Unknown tool '{action}'. Available tools: {list(self.tools.keys())}"
                else:
                    observation = "Error: No action specified. Please specify a tool to use."
                current_step.observation = observation
                reasoning_history += f"\n\nThought: {thought}\nObservation: {observation}\n"
                
                if verbose:
                    print(f"\n→ Thought: {thought}")
                    print(f"→ Observation: {observation}")
            
            steps.append(current_step)
            
            # Safety: check step limit
            if step_num >= max_steps:
                terminated_reason = "max_steps"
                final_answer = "Maximum reasoning steps reached. Unable to complete the query."
                break
        
        # If we exited without a final answer, use last observation
        if not final_answer:
            final_answer = "Agent did not provide a final answer."
        
        # Add to conversation memory
        if self.config.memory.enabled:
            self.memory.add("user", query)
            self.memory.add("assistant", final_answer)
        
        # Build response
        return AgentResponse(
            query=query,
            final_answer=final_answer,
            steps=steps,
            tool_calls=self.tool_log.all_calls(),
            total_steps=len(steps),
            terminated_reason=terminated_reason
        )
    
    def chat(self, query: str, verbose: bool = True) -> str:
        """
        Convenience method for interactive use.
        Returns just the final answer string.
        """
        response = self.run(query, verbose=verbose)
        return response.final_answer
    
    def reset_memory(self):
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()
    
    def get_memory_state(self) -> list[dict]:
        """Get current conversation history."""
        if self.memory:
            return self.memory.get_history()
        return []


def main():
    """Quick test of the agent."""
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    print(f"\nAgent Configuration:")
    print(f"  Max steps: {agent.config.agent.max_steps}")
    print(f"  Memory turns: {agent.config.memory.max_turns}")
    print(f"  Enabled tools: {list(agent.tools.keys())}")
    print(f"  Top-k retrieval: {agent.config.retrieval.top_k}")
    
    # Test query
    test_query = "What is machine learning?"
    print(f"\n{'='*60}")
    print(f"Query: {test_query}")
    print(f"{'='*60}")
    
    response = agent.run(test_query, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER")
    print(f"{'='*60}")
    print(response.final_answer)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total steps: {response.total_steps}")
    print(f"  Tool calls: {len(response.tool_calls)}")
    print(f"  Terminated: {response.terminated_reason}")


if __name__ == "__main__":
    main()
