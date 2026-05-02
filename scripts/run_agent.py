#!/usr/bin/env python3
"""
Interactive RAG Agent CLI

Usage:
  # Interactive mode (chat back and forth)
  python3 run_agent.py

  # Single query mode
  python3 run_agent.py --query "What is machine learning?"

  # Quiet mode (just the answer)
  python3 run_agent.py --query "Calculate 25 * 4" --quiet

  # With logging
  python3 run_agent.py --query "Your question" --log --experiment-type baseline
"""

import argparse
import sys
import os

# Add parent directory to path to find src/rag modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_agent import RAGAgent
from rag.agent_logger import AgentLogger


def interactive_mode(agent, use_logging=False, experiment_type="interactive"):
    """Interactive chat mode."""
    print("\n" + "="*70)
    print("RAG AGENT - INTERACTIVE MODE")
    print("="*70)
    print("\nType your questions below. Commands:")
    print("  'quit' or 'exit' - Exit")
    print("  'clear' - Clear conversation memory")
    print("  'memory' - Show conversation history")
    print("  'debug on/off' - Toggle debug mode (show retrieved docs)")
    print("  'help' - Show this help")
    print("="*70 + "\n")
    
    logger = None
    if use_logging:
        logger = AgentLogger(agent, experiment_type=experiment_type)
        print(f"✓ Logging enabled (experiment: {experiment_type})\n")
    
    debug_mode = True  # Default to True for research visibility
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            elif query.lower() == 'clear':
                agent.reset_memory()
                print("✓ Memory cleared\n")
                continue
            
            elif query.lower() == 'memory':
                history = agent.get_memory_state()
                if not history:
                    print("(No conversation history)\n")
                else:
                    print("\nConversation History:")
                    for i, turn in enumerate(history, 1):
                        role = turn['role'].capitalize()
                        content = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
                        print(f"  {i}. {role}: {content}")
                    print()
                continue
            
            elif query.lower() == 'debug on':
                debug_mode = True
                print("✓ Debug mode ON - will show retrieved documents\n")
                continue
            
            elif query.lower() == 'debug off':
                debug_mode = False
                print("✓ Debug mode OFF\n")
                continue
            
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit")
                print("  'clear' - Clear conversation memory")
                print("  'memory' - Show conversation history")
                print("  'debug on/off' - Toggle debug mode (show retrieved docs)")
                print("  'help' - Show this help\n")
                continue
            
            # Run query
            if not debug_mode:
                print("\n[Agent thinking...]")
            
            if logger:
                response = logger.run(query, verbose=False)
            else:
                response = agent.run(query, verbose=False)
            
            # Show retrieved documents in debug mode
            if debug_mode:
                print("\n" + "="*70)
                print("RETRIEVED DOCUMENTS")
                print("="*70)
                
                # Get the search tool to access last results
                search_tool = agent.tools.get('search_knowledge_base')
                if search_tool and hasattr(search_tool, 'last_results'):
                    for result in search_tool.last_results:
                        print(f"\n[{result['rank']}] Score: {result['score']:.3f} | Source: {result['source']} | Trust: {result['trust_score']}")
                        print(f"Title: {result['title']}")
                        print(f"Text: {result['text'][:200]}...")
                        if len(result['text']) > 200:
                            print(f"  ... ({len(result['text'])} chars total)")
                else:
                    print("(No search performed)")
                
                print("\n" + "="*70)
                print("AGENT RESPONSE")
                print("="*70)
            
            print(f"\nAgent: {response.final_answer}")
            
            if debug_mode:
                print(f"\n[Stats: {response.total_steps} steps, {len(response.tool_calls)} tool calls, {response.terminated_reason}]")
            else:
                print(f"       ({response.total_steps} steps, {len(response.tool_calls)} tool calls)\n")
            
            if debug_mode:
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            if debug_mode:
                traceback.print_exc()


def single_query_mode(agent, query, verbose=False, quiet=False, use_logging=False, experiment_type="single_query", show_docs=False):
    """Single query mode."""
    logger = None
    if use_logging:
        logger = AgentLogger(agent, experiment_type=experiment_type)
        response = logger.run(query, verbose=verbose)
    else:
        response = agent.run(query, verbose=verbose)
    
    if quiet:
        # Just print the answer
        print(response.final_answer)
    else:
        # Print formatted output
        print("\n" + "="*70)
        print("QUERY")
        print("="*70)
        print(query)
        
        # Show retrieved documents if requested
        if show_docs:
            print("\n" + "="*70)
            print("RETRIEVED DOCUMENTS")
            print("="*70)
            search_tool = agent.tools.get('search_knowledge_base')
            if search_tool and hasattr(search_tool, 'last_results'):
                for result in search_tool.last_results:
                    print(f"\n[{result['rank']}] Score: {result['score']:.3f} | Source: {result['source']} | Trust: {result['trust_score']}")
                    print(f"Title: {result['title']}")
                    print(f"Text: {result['text'][:300]}...")
                    if len(result['text']) > 300:
                        print(f"  ... ({len(result['text'])} chars total)")
            else:
                print("(No search performed)")
        
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(response.final_answer)
        
        print("\n" + "="*70)
        print("DETAILS")
        print("="*70)
        print(f"  Total steps: {response.total_steps}")
        print(f"  Tool calls: {len(response.tool_calls)}")
        print(f"  Terminated: {response.terminated_reason}")
        
        if response.tool_calls:
            print(f"\n  Tools used:")
            for call in response.tool_calls:
                input_preview = call.input[:50] + "..." if len(call.input) > 50 else call.input
                output_preview = call.output[:50] + "..." if len(call.output) > 50 else call.output
                print(f"    - {call.tool_name}({input_preview}) → {output_preview}")
        
        if use_logging and logger:
            print(f"\n  Session logged: {logger.current_session_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive RAG Agent for querying the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  python3 run_agent.py

  # Single question
  python3 run_agent.py --query "What is machine learning?"

  # Quiet output (just answer)
  python3 run_agent.py --query "Calculate 100 / 5" --quiet

  # With logging enabled
  python3 run_agent.py --log --experiment-type baseline

  # Verbose mode (show reasoning)
  python3 run_agent.py --query "Your question" --verbose
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to ask (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed reasoning steps'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode - only print the answer (for single query mode)'
    )
    
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable logging (saves sessions to logs/agent_runs/)'
    )
    
    parser.add_argument(
        '--experiment-type',
        type=str,
        default='manual',
        help='Experiment type for logging (baseline, attack_test, etc.)'
    )
    
    parser.add_argument(
        '--show-docs',
        action='store_true',
        help='Show retrieved documents (for debugging/research)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (default: config/agent_config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    if not args.quiet:
        print("Initializing RAG Agent...")
    
    if args.config:
        agent = RAGAgent(config_path=args.config)
    else:
        agent = RAGAgent()
    
    if not args.quiet:
        print("✓ Agent ready\n")
    
    # Run in appropriate mode
    if args.query:
        # Single query mode
        single_query_mode(
            agent,
            args.query,
            verbose=args.verbose,
            quiet=args.quiet,
            use_logging=args.log,
            experiment_type=args.experiment_type,
            show_docs=args.show_docs
        )
    else:
        # Interactive mode
        interactive_mode(
            agent,
            use_logging=args.log,
            experiment_type=args.experiment_type
        )


if __name__ == "__main__":
    main()
