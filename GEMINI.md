# Agentic RAG Research Workspace Guidelines

## Academic Artifact & Thesis Vault Invariants

1. **Thesis Artifact Authoring**:
   - Follow the `thesis-research-artifacts` skill for all formal design documents, empirical evaluations, and calibration analyses.
   - Formulate threat models using decomposed multi-stage probabilities: $P(\text{Breach}) = P(\mathcal{S}_1) \times P(\mathcal{S}_2 \mid \mathcal{S}_1)$.
   - Segregate exact numerical metrics into versioned CSV/JSON files while presenting representative approximations (`~XX%`) in thesis prose.

2. **Permanent Vault Synchronization**:
   - Every thesis-bound artifact created in this workspace or the conversation brain directory MUST be copied to `/data/my-vault/Research/thesis/artifacts/<phase_subfolder>/`.
   - Update `/data/my-vault/Research/thesis/CONSOLIDATED_RESEARCH_DECISIONS.md` and `/data/my-vault/Research/thesis/Thesis_Content_Analysis.md` whenever an experimental phase is completed.

3. **Inference & Benchmarking Safety**:
   - Enforce canonical stop sequences (`\nObservation:`) and token horizons ($N_{\text{max}} = 384$) on ReAct agent loops.
   - Never classify infrastructure crashes or repetition limit aborts as model defenses.
   - Unescape markdown underscores (`search\_knowledge\_base`) before dispatching tool calls.

4. **Cloud Infrastructure & Cost Guardrails**:
   - Whenever an evaluation utilizes cloud compute VMs:
     - `instance-20261013-025109` (10 vCPU AMD EPYC, zone `us-central1-a`) — Dedicated to Static Attack Benchmarks (local tunnel port `11435`).
     - `instance-20261023-051940` (8 vCPU Intel Xeon, zone `us-central1-a`) — Dedicated to Adaptive Attack Evaluations (local tunnel port `11436`).
   - The agent MUST verify all running VMs are stopped upon completion, failure, or cancellation:
     `gcloud compute instances stop instance-20261013-025109 --zone=us-central1-a`
     `gcloud compute instances stop instance-20261023-051940 --zone=us-central1-a`
     `# Or stop both simultaneously:`
     `gcloud compute instances stop instance-20261013-025109 instance-20261023-051940 --zone=us-central1-a`
   - Terminate associated `autossh` tunnels: `pkill autossh`.
   - Never leave cloud compute instances running unattended.
