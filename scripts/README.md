# Scripts Directory

User-facing command-line scripts for the RAG agent system.

## Available Scripts

### `run_agent.py` - Interactive RAG Agent CLI
Main interface for querying the agent.

**Usage:**
```bash
# Interactive chat mode
python3 scripts/run_agent.py

# Single query
python3 scripts/run_agent.py --query "What is machine learning?"

# Quiet mode (just the answer)
python3 scripts/run_agent.py --query "Calculate 50 * 2" --quiet

# With logging enabled
python3 scripts/run_agent.py --log --experiment-type baseline

# Show reasoning steps
python3 scripts/run_agent.py --query "Your question" --verbose
```

**Commands in interactive mode:**
- `quit` or `exit` - Exit the program
- `clear` - Clear conversation memory
- `memory` - Show conversation history
- `help` - Show available commands

## Corpus Building Scripts

Corpus building scripts are in `src/corpus/`:
- `download_wikipedia.py` - Download Wikipedia articles
- `download_arxiv.py` - Download arXiv abstracts
- `create_synthetic.py` - Generate synthetic documents
- `combine_corpus.py` - Merge all sources
- `chunk_documents.py` - Split into chunks
- `build_vector_db.py` - Build FAISS index
