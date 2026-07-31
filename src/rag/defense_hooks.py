"""
defense_hooks.py
----------------
Integration point for all defense mechanisms. 
Allows defenses to intercept and modify retrieved chunks before they are sent to the LLM.
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DefenseHook:
    """Base class for all defense mechanisms."""
    
    def __init__(self, name: str):
        self.name = name

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluate and potentially filter retrieved chunks.
        
        Args:
            chunks: List of chunk dictionaries retrieved by the search tool
            query: The user's original search query
            config: The global agent configuration
            
        Returns:
            Tuple of (filtered_chunks, metadata_about_defense_actions)
        """
        raise NotImplementedError("Defense hooks must implement filter_chunks()")


class DefensePipeline:
    """
    Chains multiple DefenseHook instances.
    Each hook processes the chunks sequentially.
    """
    
    def __init__(self, hooks: List[DefenseHook]):
        self.hooks = hooks

    def run(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run chunks through all registered defense hooks.
        
        Returns:
            Tuple of (final_filtered_chunks, combined_metadata)
        """
        pipeline_metadata = {}
        current_chunks = chunks
        
        for hook in self.hooks:
            try:
                current_chunks, meta = hook.filter_chunks(current_chunks, query, config)
                pipeline_metadata[hook.name] = meta
            except Exception as e:
                logger.error(f"Error in defense hook {hook.name}: {e}")
                pipeline_metadata[hook.name] = {"error": str(e)}
                
        return current_chunks, pipeline_metadata


class PatternMatchingDefense(DefenseHook):
    """
    Blocks chunks that contain known malicious patterns (e.g., "ignore previous instructions").
    """
    def __init__(self, name: str = "pattern_matching"):
        super().__init__(name)

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        blocked_patterns = []
        if hasattr(config, 'defenses') and hasattr(config.defenses, 'blocked_patterns'):
            blocked_patterns = config.defenses.blocked_patterns
            
        if not blocked_patterns:
            return chunks, {"action": "skipped", "reason": "No blocked patterns configured"}

        filtered_chunks = []
        blocked_count = 0
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            is_blocked = False
            for pattern in blocked_patterns:
                if pattern.lower() in text:
                    is_blocked = True
                    logger.warning(f"PatternMatchingDefense: Dropped chunk '{chunk.get('title')}' for containing blocked pattern: '{pattern}'")
                    break
            
            if not is_blocked:
                filtered_chunks.append(chunk)
            else:
                blocked_count += 1
                
        meta = {
            "action": "filtered",
            "blocked_count": blocked_count,
            "patterns_checked": len(blocked_patterns)
        }
        return filtered_chunks, meta


class TrustScoringDefense(DefenseHook):
    """
    Blocks chunks that have a trust score below the configured threshold.
    """
    def __init__(self, name: str = "trust_scoring"):
        super().__init__(name)

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        threshold = 0.0
        if hasattr(config, 'defenses') and hasattr(config.defenses, 'trust_score_threshold'):
            threshold = config.defenses.trust_score_threshold
            
        filtered_chunks = []
        blocked_count = 0
        
        for chunk in chunks:
            trust_score = chunk.get('trust_score', 1.0)
            if trust_score >= threshold:
                filtered_chunks.append(chunk)
            else:
                logger.warning(f"TrustScoringDefense: Dropped chunk '{chunk.get('title')}' with trust score {trust_score} < {threshold}")
                blocked_count += 1
                
        meta = {
            "action": "filtered",
            "blocked_count": blocked_count,
            "threshold": threshold
        }
        return filtered_chunks, meta
