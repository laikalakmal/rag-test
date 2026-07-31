"""
Config loader for the RAG agent.

Reads agent_config.yaml and provides typed access to all settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tool_calls_per_step: int = 3
    temperature: float = 0.1


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.3
    vector_db_path: str = "vector_db/baseline"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class MemoryConfig:
    max_turns: int = 10
    enabled: bool = True


@dataclass
class ToolsConfig:
    enabled_tools: list[str] = field(default_factory=lambda: [
        "search_knowledge_base", "calculator", "get_current_date"
    ])
    calculator_safe_mode: bool = True
    include_descriptions: bool = True


@dataclass
class LoggingConfig:
    log_dir: str = "logs/agent_runs"
    log_level: str = "INFO"
    save_reasoning_steps: bool = True
    save_retrieved_chunks: bool = True
    save_memory_snapshots: bool = False


@dataclass
class DefensesConfig:
    active_defenses: list[str] = field(default_factory=list)
    trust_score_threshold: float = 0.5
    blocked_patterns: list[str] = field(default_factory=list)
    relevance_threshold: float = 0.4


@dataclass
class ExperimentConfig:
    run_id: str = ""
    experiment_type: str = "baseline"
    random_seed: int = 42
    results_dir: str = "results"


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-a-Judge evaluation."""
    model: str = "mistral:7b"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.0
    timeout: int = 120


@dataclass
class Config:
    """Complete configuration for the RAG agent system."""
    agent: AgentConfig
    retrieval: RetrievalConfig
    memory: MemoryConfig
    tools: ToolsConfig
    logging: LoggingConfig
    defenses: DefensesConfig
    experiment: ExperimentConfig
    judge: JudgeConfig


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, looks for:
                     1. config/agent_config.yaml (relative to project root)
                     2. Default values
    
    Returns:
        Config object with all settings
    """
    if config_path is None:
        # Try to find config relative to project root
        possible_paths = [
            "config/agent_config.yaml",
            "../config/agent_config.yaml",
            "../../config/agent_config.yaml",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # Load YAML if found, otherwise use defaults
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    
    # Build config objects with defaults
    return Config(
        agent=AgentConfig(**data.get('agent', {})),
        retrieval=RetrievalConfig(**data.get('retrieval', {})),
        memory=MemoryConfig(**data.get('memory', {})),
        tools=ToolsConfig(**data.get('tools', {})),
        logging=LoggingConfig(**data.get('logging', {})),
        defenses=DefensesConfig(**data.get('defenses', {})),
        experiment=ExperimentConfig(**data.get('experiment', {})),
        judge=JudgeConfig(**data.get('judge', {})),
    )


def get_config() -> Config:
    """
    Convenience function to load config from default location.
    Use this in agent code: `from config.config_loader import get_config`
    """
    return load_config()


if __name__ == "__main__":
    # Test config loading
    cfg = load_config("config/agent_config.yaml")
    print(f"Agent max_steps: {cfg.agent.max_steps}")
    print(f"Retrieval top_k: {cfg.retrieval.top_k}")
    print(f"Memory enabled: {cfg.memory.enabled}")
    print(f"Enabled tools: {cfg.tools.enabled_tools}")
    print(f"Active defenses: {cfg.defenses.active_defenses}")
    print(f"Log directory: {cfg.logging.log_dir}")
    print("\n✓ Config loaded successfully")
