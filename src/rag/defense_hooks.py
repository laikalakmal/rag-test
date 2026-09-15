"""
defense_hooks.py
----------------
Integration point for all defense mechanisms. 
Allows defenses to intercept and modify retrieved chunks before they are sent to the LLM.
"""

import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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


class RelevanceFilteringDefense(DefenseHook):
    """
    Query-Document Relevance Filtering Defense.
    
    Computes semantic similarity between the user's query and each retrieved
    chunk using a lightweight sentence-transformers model. Chunks scoring
    below the configured relevance_threshold are entirely discarded.
    
    This defense targets indirect prompt injection via "context poisoning,"
    where an attacker manipulates the database so that an irrelevant, poisoned
    document is retrieved alongside benign ones.
    
    Design Decisions:
      - Chunks are DISCARDED entirely (not edited/sanitized).
      - The embedding model is loaded ONCE at init to avoid repeated loading.
      - Rich metadata is returned for research logging (per-chunk scores,
        blocked/passed counts, all_chunks_blocked flag).
    """
    
    def __init__(self, name: str = "relevance_filtering", model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name)
        logger.info(f"RelevanceFilteringDefense: Loading embedding model '{model_name}'...")
        self._model = SentenceTransformer(model_name)
        logger.info(f"RelevanceFilteringDefense: Model loaded successfully.")

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter chunks by cosine similarity to the user query.
        
        Args:
            chunks: Retrieved chunk dicts, each must have a 'text' key.
            query: The user's original search query.
            config: Global config; reads config.defenses.relevance_threshold.
            
        Returns:
            Tuple of (filtered_chunks, metadata).
            metadata includes per-chunk scores for ROC curve analysis.
        """
        # Read threshold from config (default 0.4)
        threshold = 0.4
        if hasattr(config, 'defenses') and hasattr(config.defenses, 'relevance_threshold'):
            threshold = config.defenses.relevance_threshold

        # Edge case: no chunks to filter
        if not chunks:
            return chunks, {
                "action": "skipped",
                "reason": "No chunks to filter",
                "all_chunks_blocked": False
            }

        # 1. Embed the user's query
        query_embedding = self._model.encode([query])
        
        # 2. Embed all chunk texts
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        chunk_embeddings = self._model.encode(chunk_texts)
        
        # 3. Calculate cosine similarity between query and each chunk
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # 4. Filter: discard chunks below threshold
        filtered_chunks = []
        blocked_count = 0
        chunk_scores = []  # For research logging
        
        for idx, (chunk, score) in enumerate(zip(chunks, similarities)):
            score_float = float(score)
            chunk_scores.append({
                "chunk_index": idx,
                "title": chunk.get('title', f'chunk_{idx}'),
                "score": round(score_float, 4),
                "passed": score_float >= threshold
            })
            
            if score_float >= threshold:
                # Attach the relevance score to the chunk for downstream use
                chunk['relevance_score'] = score_float
                filtered_chunks.append(chunk)
            else:
                logger.warning(
                    f"RelevanceFilteringDefense: Dropped chunk '{chunk.get('title', f'chunk_{idx}')}' "
                    f"(score: {score_float:.4f} < threshold: {threshold})"
                )
                blocked_count += 1
        
        all_blocked = len(filtered_chunks) == 0
        
        if all_blocked:
            logger.warning(
                f"RelevanceFilteringDefense: ALL {len(chunks)} chunks were blocked. "
                f"Hard block will be triggered."
            )
        
        meta = {
            "action": "filtered",
            "threshold": threshold,
            "total_chunks": len(chunks),
            "passed_count": len(filtered_chunks),
            "blocked_count": blocked_count,
            "all_chunks_blocked": all_blocked,
            "chunk_scores": chunk_scores
        }
        return filtered_chunks, meta
