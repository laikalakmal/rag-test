# RAG Research Project

Research project evaluating adaptive prompt injection attacks and defense mechanisms in Retrieval-Augmented Generation (RAG) systems.

## Project Overview

**Research Question:** How effective are existing defense mechanisms in protecting agentic RAG systems against adaptive indirect prompt injection attacks?

**Approach:** Build → Defend → Attack → Measure → Analyze

---

## Quick Start

### 1. Setup Environment
```bash
cd /data/learn-ai/rag-test
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build Corpus
```bash
./scripts/01_setup_corpus.sh
```
This downloads and processes 700+ documents (20-45 minutes).

### 3. Test Baseline
```bash
python scripts/02_test_baseline.py
```
Measure baseline vulnerability to attacks.

### 4. Implement Defenses
```bash
python scripts/03_implement_defenses.py
```
Test individual defense mechanisms.

---

## Project Structure

```
rag-test/
├── data/              # All datasets (corpus, attacks, queries)
├── vector_db/         # FAISS vector databases
├── src/               # Source code (modular components)
├── scripts/           # Executable research scripts
├── experiments/       # Results and outputs
├── notebooks/         # Jupyter analysis notebooks
└── tests/             # Unit tests
```

See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for detailed structure.

---

## Documentation

- **[CORPUS_SETUP_GUIDE.md](CORPUS_SETUP_GUIDE.md)** - Step-by-step corpus building
- **[FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)** - Complete project structure
- **[data/README.md](data/README.md)** - Data organization
- **[src/README.md](src/README.md)** - Source code overview

---

## Research Pipeline

### Phase 1: Setup (Weeks 1-2)
1. Build document corpus (Wikipedia, arXiv, synthetic)
2. Create vector database
3. Implement baseline RAG system

### Phase 2: Baseline (Weeks 3-4)
1. Create attack patterns
2. Create benign queries
3. Measure baseline vulnerability

### Phase 3: Defenses (Weeks 5-8)
1. Implement 6 defense mechanisms
2. Test individually
3. Test combinations

### Phase 4: Adaptive Attacks (Weeks 9-12)
1. Generate adaptive attacks
2. Test against defenses
3. Measure robustness

### Phase 5: Analysis (Weeks 13-16)
1. Statistical analysis
2. Visualizations
3. Write paper

---

## Key Metrics

- **ASR (Attack Success Rate)** - % of successful attacks
- **FPR (False Positive Rate)** - % of benign queries blocked
- **TPR (Task Performance Retention)** - % of functionality preserved

**Goal:** Minimize ASR while keeping FPR low and TPR high.

---

## Defense Mechanisms

1. **Hierarchical Prompting** - Structured prompt boundaries
2. **Embedding Filter** - Detect attack-like embeddings
3. **Pattern Matching** - Keyword/regex detection
4. **Relevance Filter** - Check query-document relevance
5. **Response Verification** - Analyze output for compromise
6. **Trust Scoring** - Weight by document source

---

## Technologies

- **LangChain** - RAG framework
- **FAISS** - Vector database
- **Sentence-Transformers** - Embeddings
- **HuggingFace** - Models and datasets
- **Python 3.10+** - Programming language

---

## Legacy Files

- **docs_legacy/** - Original simple test documents (kept for reference)
- **rag_bot_legacy.py** - Original simple RAG bot (backup)

These are preserved but not used in research. The new structure is in `src/`.

---

## Contributing

This is a research project. When adding new components:

1. Place code in appropriate `src/` subdirectory
2. Add tests in `tests/`
3. Update relevant README
4. Document in git commits

---

## License

[Add your license here]

---

## Contact

[Your contact information]

---

## References

- Yi et al. (2025) - "Adaptive Attacks Break Defenses Against Indirect Prompt Injection"
- [Add your references here]
