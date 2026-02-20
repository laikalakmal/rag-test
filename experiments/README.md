# Experiment Results

Stores all experimental outputs.

## Structure

- **baseline/** - Undefended RAG results
- **defense_individual/** - Each defense tested alone
- **defense_combinations/** - Multiple defenses together
- **adaptive_testing/** - Adaptive attack results
- **figures/** - Plots and visualizations

## Metrics

Each experiment produces:
- `results.json` - Raw results
- `metrics.json` - ASR, FPR, TPR
