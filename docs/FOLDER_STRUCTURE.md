# RAG Research Project - Folder Structure

## Current Structure (Simple Test Bot)
```
/data/learn-ai/rag-test/
├── docs/              # 5-6 simple .txt files
├── rag_bot.py         # Basic RAG implementation
├── llm_provider.py    # LLM configuration
└── venv/              # Python virtual environment
```

**Problems with current structure:**
- ❌ No separation between raw and processed data
- ❌ No place for attack patterns
- ❌ No organization for experiments
- ❌ Can't scale to 700+ documents
- ❌ No version control for datasets
- ❌ Mixing test data with production code

---

## Recommended Research Structure

```
/data/learn-ai/rag-test/
│
├── README.md                       # Project overview
├── CORPUS_SETUP_GUIDE.md          # Setup instructions (already exists)
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (LLM keys)
├── .gitignore                     # Git ignore rules
│
├── data/                          # 🗂️ ALL DATA FILES
│   ├── raw/                       # 📥 Downloaded/original data (never modify)
│   │   ├── wikipedia/
│   │   │   └── wikipedia_500.json         # 500 Wikipedia articles
│   │   ├── arxiv/
│   │   │   └── arxiv_150.json             # 150 arXiv abstracts
│   │   ├── synthetic/
│   │   │   └── synthetic_50.json          # 50 synthetic docs
│   │   └── README.md                      # Data source documentation
│   │
│   ├── processed/                 # 📤 Processed/chunked data
│   │   ├── complete_corpus.json           # Combined all sources
│   │   ├── chunked_corpus.json            # After chunking (optional)
│   │   └── corpus_stats.json              # Statistics about corpus
│   │
│   ├── attacks/                   # 💣 Attack patterns for testing
│   │   ├── static_attacks.json            # 100-150 basic attacks
│   │   ├── adaptive_attacks.json          # 150-200 evolved attacks
│   │   ├── attack_categories.json         # Organized by type
│   │   └── README.md                      # Attack documentation
│   │
│   ├── benign/                    # ✅ Legitimate queries for testing
│   │   ├── benign_queries.json            # 200-300 normal questions
│   │   ├── query_categories.json          # Organized by topic
│   │   └── README.md                      # Query documentation
│   │
│   └── cache/                     # 💾 Downloaded models cache
│       └── huggingface/                   # Sentence-transformers cache
│
├── vector_db/                     # 🔍 FAISS vector database
│   ├── baseline/                  # Undefended RAG database
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── with_defenses/             # Defended versions (optional)
│   │   ├── hierarchical/
│   │   ├── embedding_filter/
│   │   └── pattern_matching/
│   └── README.md                  # Database documentation
│
├── src/                           # 💻 SOURCE CODE
│   ├── __init__.py
│   │
│   ├── corpus/                    # 📚 Corpus building scripts
│   │   ├── __init__.py
│   │   ├── download_wikipedia.py          # Step 1: Download Wikipedia
│   │   ├── download_arxiv.py              # Step 2: Download arXiv
│   │   ├── create_synthetic.py            # Step 3: Create synthetic docs
│   │   ├── combine_corpus.py              # Step 4: Combine all
│   │   ├── chunk_documents.py             # Step 5: Chunk into pieces
│   │   └── build_vector_db.py             # Step 6-7: Embed + FAISS
│   │
│   ├── rag/                       # 🤖 RAG system implementation
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py                # Main RAG pipeline class
│   │   ├── retriever.py                   # Document retrieval logic
│   │   ├── llm_provider.py                # LLM configuration (move from root)
│   │   └── prompts.py                     # Prompt templates
│   │
│   ├── defenses/                  # 🛡️ Defense mechanisms
│   │   ├── __init__.py
│   │   ├── base_defense.py                # Abstract defense class
│   │   ├── hierarchical_prompting.py      # Defense 1
│   │   ├── embedding_filter.py            # Defense 2
│   │   ├── pattern_matching.py            # Defense 3
│   │   ├── relevance_filter.py            # Defense 4
│   │   ├── response_verification.py       # Defense 5
│   │   ├── trust_scoring.py               # Defense 6
│   │   └── ensemble.py                    # Combined defenses
│   │
│   ├── attacks/                   # ⚔️ Attack generation
│   │   ├── __init__.py
│   │   ├── static_attacks.py              # Create basic attacks
│   │   ├── adaptive_attacks.py            # Generate evolved attacks
│   │   ├── attack_evaluator.py            # Check if attack succeeded
│   │   └── paraphraser.py                 # Paraphrase attacks
│   │
│   ├── evaluation/                # 📊 Metrics and evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                     # ASR, FPR, TPR calculations
│   │   ├── evaluator.py                   # Run evaluations
│   │   └── visualizations.py              # Generate plots
│   │
│   └── utils/                     # 🔧 Utility functions
│       ├── __init__.py
│       ├── config.py                      # Configuration management
│       ├── logger.py                      # Logging setup
│       └── helpers.py                     # Common helper functions
│
├── scripts/                       # 🚀 Executable scripts
│   ├── 01_setup_corpus.sh                 # One-click corpus setup
│   ├── 02_test_baseline.py               # Test undefended RAG
│   ├── 03_implement_defenses.py           # Add defenses
│   ├── 04_generate_adaptive_attacks.py    # Create evolved attacks
│   ├── 05_run_experiments.py              # Full evaluation
│   └── 06_analyze_results.py              # Generate report
│
├── experiments/                   # 🧪 EXPERIMENT RESULTS
│   ├── baseline/                  # Undefended RAG results
│   │   ├── results.json
│   │   ├── metrics.json
│   │   └── logs/
│   ├── defense_individual/        # Each defense tested alone
│   │   ├── hierarchical/
│   │   ├── embedding_filter/
│   │   └── pattern_matching/
│   ├── defense_combinations/      # Multiple defenses together
│   │   ├── hier_embed/
│   │   └── all_defenses/
│   ├── adaptive_testing/          # Adaptive attack results
│   │   └── by_defense/
│   └── figures/                   # Generated plots
│       ├── asr_comparison.png
│       ├── fpr_analysis.png
│       └── tradeoff_curves.png
│
├── notebooks/                     # 📓 Jupyter notebooks (optional)
│   ├── 01_explore_corpus.ipynb            # Data exploration
│   ├── 02_test_retrieval.ipynb            # Test vector DB
│   ├── 03_analyze_attacks.ipynb           # Attack analysis
│   └── 04_results_visualization.ipynb     # Final results
│
├── tests/                         # ✅ Unit tests
│   ├── __init__.py
│   ├── test_corpus.py
│   ├── test_rag.py
│   ├── test_defenses.py
│   └── test_metrics.py
│
├── docs_legacy/                   # 📁 Old simple test docs (keep for reference)
│   ├── doc1.txt
│   └── doc2.txt
│
└── rag_bot_legacy.py              # 🗂️ Original simple bot (keep as backup)
```

---

## Key Principles

### 1. Separation of Concerns
```
data/          # Only data files (no code)
src/           # Only source code (no data)
scripts/       # Executable scripts
experiments/   # Results and outputs
```

### 2. Raw vs Processed
```
data/raw/          # Never modify - original downloads
data/processed/    # Generated from raw - can regenerate
```

### 3. Reproducibility
```
data/raw/README.md        # Documents data sources
experiments/baseline/     # Version controlled results
requirements.txt          # Exact package versions
```

### 4. Scalability
```
vector_db/baseline/              # Start here
vector_db/with_defenses/...      # Add as you go
experiments/baseline/            # Expand with more tests
```

---

## Migration Plan (From Current to Research Structure)

### Phase 1: Initial Setup (Now)
```bash
# Create folder structure
mkdir -p data/{raw/{wikipedia,arxiv,synthetic},processed,attacks,benign,cache}
mkdir -p src/{corpus,rag,defenses,attacks,evaluation,utils}
mkdir -p vector_db/baseline
mkdir -p scripts
mkdir -p experiments/{baseline,defense_individual,figures}
mkdir -p tests

# Move existing files
mv docs/ docs_legacy/
mv rag_bot.py rag_bot_legacy.py
mv llm_provider.py src/rag/
```

### Phase 2: Build Corpus (Week 1)
```bash
# Download data → data/raw/
python src/corpus/download_wikipedia.py
python src/corpus/download_arxiv.py
python src/corpus/create_synthetic.py

# Process data → data/processed/
python src/corpus/combine_corpus.py
python src/corpus/chunk_documents.py

# Build database → vector_db/baseline/
python src/corpus/build_vector_db.py
```

### Phase 3: Research Setup (Week 2-3)
```bash
# Create attack patterns → data/attacks/
python src/attacks/static_attacks.py

# Create benign queries → data/benign/
python src/attacks/create_benign_queries.py

# Implement RAG system → src/rag/
python src/rag/rag_pipeline.py

# Test baseline → experiments/baseline/
python scripts/02_test_baseline.py
```

### Phase 4: Defense Implementation (Week 4-8)
```bash
# Implement defenses → src/defenses/
# Each defense as separate module

# Test individually → experiments/defense_individual/
python scripts/03_implement_defenses.py
```

### Phase 5: Adaptive Attacks (Week 9-12)
```bash
# Generate adaptive attacks → data/attacks/
python scripts/04_generate_adaptive_attacks.py

# Test against defenses → experiments/adaptive_testing/
python scripts/05_run_experiments.py
```

---

## Benefits of This Structure

### ✅ For Development
- **Clear separation** - Know where everything goes
- **Modular** - Easy to add new defenses/attacks
- **Testable** - Unit tests for each component
- **Debuggable** - Logs and intermediate results

### ✅ For Research
- **Reproducible** - Others can replicate exactly
- **Versionable** - Git-friendly structure
- **Publishable** - Can share code + data
- **Extensible** - Easy to add more experiments

### ✅ For Collaboration
- **Self-documenting** - Structure tells the story
- **Standard** - Follows ML research conventions
- **Professional** - Ready for paper submission
- **Maintainable** - Easy to update/fix

---

## File Size Estimates

```
data/
  raw/                 ~35 MB     (original corpus)
  processed/           ~40 MB     (chunked + metadata)
  attacks/             ~2 MB      (attack patterns)
  benign/              ~1 MB      (benign queries)
  cache/               ~2 GB      (HuggingFace models)

vector_db/
  baseline/            ~300 MB    (FAISS index + text)
  with_defenses/       ~300 MB    (each variation)

experiments/
  baseline/            ~10 MB     (results JSON)
  defense_*/           ~10 MB     (each experiment)
  figures/             ~5 MB      (PNG plots)

Total: ~3-4 GB
```

---

## .gitignore Recommendations

```gitignore
# Data (too large for git)
data/raw/
data/cache/
vector_db/

# Experiment outputs (regeneratable)
experiments/*/logs/
experiments/*/temp/

# Python
__pycache__/
*.pyc
venv/

# Environment
.env

# Large files
*.faiss
*.pkl

# Keep structure but not content
!data/raw/.gitkeep
!vector_db/.gitkeep
```

---

## What to Keep in Git

### ✅ Include in Git:
- Source code (`src/`)
- Scripts (`scripts/`)
- Configuration files
- Small sample data (for testing)
- Documentation
- Experiment configs (not outputs)

### ❌ Exclude from Git:
- Large datasets (`data/raw/`)
- Vector databases (`vector_db/`)
- Model caches (`data/cache/`)
- Experiment outputs (unless small)

### 📦 Share Separately:
- Corpus → Zenodo/HuggingFace
- Vector DB → Google Drive
- Results → Paper supplementary materials

---

## Next Steps

1. **Review this structure** - Make sure it makes sense for your workflow
2. **Create folders** - Set up the directory tree
3. **Migrate files** - Move existing code into `src/`
4. **Start building** - Follow CORPUS_SETUP_GUIDE.md
5. **Iterate** - Adjust structure as you learn

Would you like me to:
- **A)** Create this folder structure now?
- **B)** Create a migration script to reorganize existing files?
- **C)** Start with just the essential folders (minimal structure)?
- **D)** Something else?
