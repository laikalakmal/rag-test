#!/usr/bin/env python3
"""
Smoke test for the RAG Agent.

Tests:
  1. Basic query (single-step search)
  2. Multi-step reasoning (search + calculator)
  3. Memory persistence (follow-up question)
  4. Tool call logging verification
  5. Config loading
"""

import sys
import os
sys.path.insert(0, 'src/rag')
sys.path.insert(0, '.')

from rag_agent import RAGAgent
from agent_logger import AgentLogger


def test_1_basic_query():
    """Test 1: Basic single-step query."""
    print("\n" + "="*70)
    print("TEST 1: Basic Query (Single-Step)")
    print("="*70)
    
    agent = RAGAgent()
    response = agent.run("What is machine learning?", verbose=False)
    
    print(f"Query: What is machine learning?")
    print(f"Answer: {response.final_answer[:150]}...")
    print(f"\n✓ Steps: {response.total_steps}")
    print(f"✓ Tool calls: {len(response.tool_calls)}")
    print(f"✓ Terminated: {response.terminated_reason}")
    
    assert response.total_steps > 0, "No steps executed"
    assert len(response.tool_calls) > 0, "No tools called"
    assert response.terminated_reason == "completed", f"Failed: {response.terminated_reason}"
    assert len(response.final_answer) > 20, "Answer too short"
    
    print("\n✅ TEST 1 PASSED")
    return agent


def test_2_multi_step_reasoning(agent):
    """Test 2: Multi-step query requiring multiple tools."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Step Reasoning (Search + Calculator)")
    print("="*70)
    
    # Reset memory to start fresh
    agent.reset_memory()
    
    response = agent.run("Calculate 25 times 4", verbose=False)
    
    print(f"Query: Calculate 25 times 4")
    print(f"Answer: {response.final_answer}")
    print(f"\n✓ Steps: {response.total_steps}")
    print(f"✓ Tool calls: {len(response.tool_calls)}")
    
    # Verify calculator was used
    tool_names = [call.tool_name for call in response.tool_calls]
    assert "calculator" in tool_names, "Calculator not used"
    
    # Verify answer contains 100
    assert "100" in response.final_answer, "Wrong calculation result"
    
    print("\n✅ TEST 2 PASSED")
    return agent


def test_3_memory_persistence(agent):
    """Test 3: Memory carries context across turns."""
    print("\n" + "="*70)
    print("TEST 3: Memory Persistence (Follow-up Question)")
    print("="*70)
    
    # Reset memory
    agent.reset_memory()
    
    # First query
    response1 = agent.run("My favorite color is blue.", verbose=False)
    print(f"Turn 1: My favorite color is blue.")
    print(f"Answer: {response1.final_answer[:100]}...")
    
    # Check memory
    memory_state = agent.get_memory_state()
    print(f"\n✓ Memory has {len(memory_state)} turns")
    assert len(memory_state) == 2, f"Expected 2 turns, got {len(memory_state)}"
    
    # Follow-up query that requires memory
    response2 = agent.run("What is my favorite color?", verbose=False)
    print(f"\nTurn 2: What is my favorite color?")
    print(f"Answer: {response2.final_answer}")
    
    # Verify answer references blue
    assert "blue" in response2.final_answer.lower(), "Memory not used - doesn't recall blue"
    
    memory_state = agent.get_memory_state()
    print(f"\n✓ Memory now has {len(memory_state)} turns")
    assert len(memory_state) == 4, f"Expected 4 turns, got {len(memory_state)}"
    
    print("\n✅ TEST 3 PASSED")
    return agent


def test_4_logging():
    """Test 4: Verify logging saves correctly."""
    print("\n" + "="*70)
    print("TEST 4: Agent Logger Verification")
    print("="*70)
    
    agent = RAGAgent()
    logger = AgentLogger(agent, experiment_type="smoke_test")
    
    query = "What is the current date?"
    response = logger.run(query, verbose=False)
    
    print(f"Query: {query}")
    print(f"Answer: {response.final_answer}")
    print(f"Session ID: {logger.current_session_id}")
    
    # Load the saved log
    log_data = logger.load_log(logger.current_session_id)
    
    print(f"\n✓ Log saved and loaded successfully")
    print(f"✓ Metadata present: {bool(log_data.get('metadata'))}")
    print(f"✓ Reasoning steps: {len(log_data.get('reasoning_steps', []))}")
    print(f"✓ Tool calls: {len(log_data.get('tool_calls', []))}")
    
    assert log_data.get('metadata'), "No metadata in log"
    assert log_data.get('reasoning_steps'), "No reasoning steps in log"
    assert log_data.get('final_answer'), "No final answer in log"
    
    # Analyze session
    analysis = logger.analyze_session(logger.current_session_id)
    print(f"\n✓ Session analysis:")
    print(f"  Tool usage: {analysis['tool_usage']}")
    print(f"  Attack indicators: {analysis['attack_indicators']}")
    
    print("\n✅ TEST 4 PASSED")


def test_5_config_loading():
    """Test 5: Verify config is loaded properly."""
    print("\n" + "="*70)
    print("TEST 5: Configuration Loading")
    print("="*70)
    
    agent = RAGAgent()
    
    print(f"✓ Max steps: {agent.config.agent.max_steps}")
    print(f"✓ Top-k: {agent.config.retrieval.top_k}")
    print(f"✓ Memory max turns: {agent.config.memory.max_turns}")
    print(f"✓ Memory enabled: {agent.config.memory.enabled}")
    print(f"✓ Enabled tools: {agent.config.tools.enabled_tools}")
    print(f"✓ Active defenses: {agent.config.defenses.active_defenses}")
    
    assert agent.config.agent.max_steps == 10, "Wrong max_steps"
    assert agent.config.retrieval.top_k == 5, "Wrong top_k"
    assert agent.config.memory.enabled == True, "Memory should be enabled"
    assert len(agent.config.tools.enabled_tools) >= 2, "Not enough tools"
    
    print("\n✅ TEST 5 PASSED")


def main():
    print("\n" + "="*70)
    print("RAG AGENT SMOKE TEST SUITE")
    print("="*70)
    print("\nRunning 5 tests to verify agent functionality...")
    
    try:
        # Test 1: Basic query
        agent = test_1_basic_query()
        
        # Test 2: Multi-step
        agent = test_2_multi_step_reasoning(agent)
        
        # Test 3: Memory
        agent = test_3_memory_persistence(agent)
        
        # Test 4: Logging
        test_4_logging()
        
        # Test 5: Config
        test_5_config_loading()
        
        print("\n" + "="*70)
        print("🎉 ALL SMOKE TESTS PASSED 🎉")
        print("="*70)
        print("\nThe RAG Agent is working correctly:")
        print("  ✓ ReAct reasoning loop functional")
        print("  ✓ Tools execute and log properly")
        print("  ✓ Memory persists across turns")
        print("  ✓ Logging saves structured JSON")
        print("  ✓ Config system integrated")
        print("\nReady for defense implementation and attack testing.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
