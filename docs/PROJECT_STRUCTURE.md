# Project Structure - RAG Agent Research

## Directory Organization

```
/data/learn-ai/rag-test/
├── config/                      # Configuration files
│   ├── agent_config.yaml        # Main agent settings
│   └── config_loader.py         # Config loader with typed dataclasses
│
├── scripts/                     # User-facing CLI scripts
│   └── run_agent.py            # ⭐ Main agent interface
│
├── src/                         # Source code
│   ├── rag/                    # RAG agent implementation
│   │   ├── rag_agent.py        # ⭐ ReAct agent (multi-step reasoning)
│   │   ├── agent_logger.py     # Session logging for experiments
│   │   ├── tools.py            # Agent tools (search, calculator, etc.)
│   │   ├── memory.py           # Conversation memory
│   │   ├── llm_provider.py     # LLM abstraction layer
│   │   └── rag_bot_legacy.py   # Old single-pass bot (reference only)
│   │
│   ├── corpus/                 # Corpus building pipeline
│   │   ├── download_wikipedia.py
│   │   ├── download_arxiv.py
│   │   ├── create_synthetic.py
│   │   ├── combine_corpus.py
│   │   ├── chunk_documents.py
│   │   └── build_vector_db.py
│   │
│   ├── defenses/               # Defense mechanisms (to be implemented)
│   ├── attacks/                # Attack generation (to be implemented)
│   ├── evaluation/             # Experiment runners (to be implemented)
│   └── utils/                  # Shared utilities
│
├── tests/                       # Test files
│   ├── test_smoke_agent.py     # ⭐ Smoke tests for agent
│   └── test_agent_components.py # Component-level tests
│
├── data/                        # Generated data
│   ├── raw/                    # Downloaded sources
│   │   ├── wikipedia/
│   │   ├── arxiv/
│   │   └── synthetic/
│   └── processed/              # Processed corpus
│       ├── complete_corpus.json
│       └── chunked_corpus.json
│
├── vector_db/                   # FAISS vector database
│   └── baseline/
│       ├── faiss.index         # 34,427 embedded chunks
│       └── metadata.json       # Parallel metadata array
│
├── logs/                        # Agent execution logs
│   └── agent_runs/             # JSON session logs
│
├── results/                     # Experiment results
├── experiments/                 # Experiment configurations
├── notebooks/                   # Jupyter notebooks for analysis
└── docs/                        # Documentation

```

## Key Files

### For Using the Agent
- **`scripts/run_agent.py`** - Main CLI interface
- **`config/agent_config.yaml`** - Tune all parameters here

### For Development
- **`src/rag/rag_agent.py`** - Core agent implementation
- **`src/rag/tools.py`** - Tool definitions
- **`src/rag/agent_logger.py`** - Experiment logging

### For Research
- **`logs/agent_runs/*.json`** - All logged sessions
- **`config/agent_config.yaml`** - Experiment parameters
- **`src/defenses/`** - Where defenses will go
- **`src/attacks/`** - Where attack generation will go

## How to Run

### 1. Interactive Agent
```bash
cd /data/learn-ai/rag-test
source venv/bin/activate
python3 scripts/run_agent.py
```

### 2. Single Query
```bash
python3 scripts/run_agent.py --query "Your question" --quiet
```

### 3. With Logging (for experiments)
```bash
python3 scripts/run_agent.py --log --experiment-type baseline
```

### 4. Run Tests
```bash
python3 tests/test_smoke_agent.py
```

## Files No Longer Used

- **`src/rag/rag_bot_legacy.py`** - Old single-pass bot, kept for reference
  - The new agent (`rag_agent.py`) replaces this
  - Uses ReAct pattern with tools and memory
  - Required for "agentic RAG" research

## Next Steps

1. **Implement Defenses** in `src/defenses/`:
   - `hierarchical_prompt.py`
   - `pattern_matching.py`
   - `embedding_filter.py`
   - `relevance_filter.py`
   - `response_verifier.py`
   - `trust_scoring.py`

2. **Build Attack Dataset** in `data/attacks/`:
   - Static attacks (instruction override, data exfiltration, etc.)
   - Benign queries for false positive testing

3. **Create Evaluation Framework** in `src/evaluation/`:
   - Experiment runner
   - Metrics calculator (ASR, FPR, TPR)
   - Results analyzer

4. **Run Experiments**:
   - Baseline (no defenses)
   - Individual defenses
   - Defense combinations
   - Adaptive attacks
