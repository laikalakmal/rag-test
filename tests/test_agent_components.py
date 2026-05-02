#!/usr/bin/env python3
"""
Interactive test script for agent components.
Run: python3 test_agent_components.py
"""

import sys
sys.path.insert(0, 'src/rag')

from tools import ToolCallLog, build_tools
from memory import ConversationMemory


def test_tools():
    print("\n" + "="*60)
    print("TESTING TOOLS")
    print("="*60)
    
    # Create log and tools
    log = ToolCallLog()
    tools = build_tools(log, llm_call_fn=None)
    
    print(f"\n✓ Available tools: {list(tools.keys())}")
    
    # Test calculator
    print("\n--- Calculator Tool ---")
    result = tools['calculator'].run("2 + 2")
    print(f"  2 + 2 = {result}")
    
    result = tools['calculator'].run("(12 * 8) + 4")
    print(f"  (12 * 8) + 4 = {result}")
    
    result = tools['calculator'].run("100 / 5")
    print(f"  100 / 5 = {result}")
    
    # Test security
    print("\n--- Security Test ---")
    result = tools['calculator'].run("__import__('os').system('whoami')")
    print(f"  Malicious input blocked: {result}")
    
    # Test date
    print("\n--- Get Current Date Tool ---")
    result = tools['get_current_date'].run("")
    print(f"  Current date: {result}")
    
    # Test search
    print("\n--- Search Knowledge Base Tool ---")
    result = tools['search_knowledge_base'].run("machine learning")
    print(f"  Search 'machine learning':")
    if isinstance(result, str) and result.startswith("Error"):
        print(f"    {result}")
    else:
        print(f"    Found {len(result)} results")
        if result:
            print(f"    Top result: {result[0]['text'][:100]}...")
    
    # Check log
    print("\n--- Tool Call Log ---")
    all_calls = log.all_calls()
    print(f"  Total calls logged: {len(all_calls)}")
    for i, call in enumerate(all_calls[:3], 1):
        print(f"  {i}. Step {call.step}, Tool: {call.tool_name}, Success: {call.error is None}")
    
    print("\n✓ Tools test complete!")


def test_memory():
    print("\n" + "="*60)
    print("TESTING MEMORY")
    print("="*60)
    
    # Create memory with 4-turn window
    memory = ConversationMemory(max_turns=4)
    
    print(f"\n✓ Created memory: {memory}")
    
    # Add some turns
    print("\n--- Adding conversation turns ---")
    memory.add("user", "What is RAG?")
    print(f"  Added user turn. Memory size: {len(memory)}")
    
    memory.add("assistant", "RAG is Retrieval-Augmented Generation.")
    print(f"  Added assistant turn. Memory size: {len(memory)}")
    
    memory.add("user", "What are prompt injections?")
    print(f"  Added user turn. Memory size: {len(memory)}")
    
    memory.add("assistant", "They are adversarial inputs in retrieved content.")
    print(f"  Added assistant turn. Memory size: {len(memory)}")
    
    # Should be at window limit
    assert len(memory) == 4, "Expected 4 turns"
    
    # Add another - should evict oldest
    print("\n--- Testing sliding window ---")
    print(f"  Before: {len(memory)} turns")
    memory.add("user", "How do defenses work?")
    print(f"  After adding 5th turn: {len(memory)} turns")
    print(f"  ✓ Oldest turn evicted (window maintained at {memory.max_turns})")
    
    # Check history
    print("\n--- Memory history ---")
    history = memory.get_history()
    for i, turn in enumerate(history, 1):
        preview = turn['content'][:50] + "..." if len(turn['content']) > 50 else turn['content']
        print(f"  {i}. {turn['role']}: {preview}")
    
    # Format for prompt
    print("\n--- Formatted prompt ---")
    prompt_text = memory.format_for_prompt()
    print(prompt_text)
    
    # Test clear
    print("\n--- Testing clear ---")
    memory.clear()
    print(f"  After clear: {len(memory)} turns")
    print(f"  Formatted: '{memory.format_for_prompt()}'")
    
    print("\n✓ Memory test complete!")


def interactive_test():
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nYou can now test tools and memory interactively.")
    print("\nExamples:")
    print("  - tools['calculator'].run('123 * 456')")
    print("  - tools['get_current_date'].run('')")
    print("  - memory.add('user', 'Hello!')")
    print("  - memory.get_history()")
    print("  - log.all_calls()")
    print("\nType 'exit' to quit.\n")
    
    log = ToolCallLog()
    tools = build_tools(log, llm_call_fn=None)
    memory = ConversationMemory(max_turns=4)
    
    import code
    code.interact(local=locals(), banner="")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test agent components")
    parser.add_argument("--tools", action="store_true", help="Test tools only")
    parser.add_argument("--memory", action="store_true", help="Test memory only")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # If no specific flags, run all tests
    if not any([args.tools, args.memory, args.interactive]):
        test_tools()
        test_memory()
    else:
        if args.tools:
            test_tools()
        if args.memory:
            test_memory()
        if args.interactive:
            interactive_test()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE ✓")
    print("="*60)
