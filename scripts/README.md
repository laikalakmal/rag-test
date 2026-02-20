# Executable Scripts

High-level scripts for research pipeline.

## Usage Order

1. `01_setup_corpus.sh` - Download and build corpus (20-45 min)
2. `02_test_baseline.py` - Test undefended RAG
3. `03_implement_defenses.py` - Test defenses
4. `04_generate_adaptive_attacks.py` - Evolve attacks
5. `05_run_experiments.py` - Full evaluation
6. `06_analyze_results.py` - Generate report
