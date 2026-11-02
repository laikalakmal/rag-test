"""
defense_hooks.py
----------------
Integration point for all defense mechanisms. 
Allows defenses to intercept and modify retrieved chunks before they are sent to the LLM.
"""

import html
import logging
import re
import unicodedata
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


CANONICAL_SIGNATURE_CLASSES = {
    "class_a_overrides": {
        "class_id": "T_OVR",
        "description": "Direct Instruction Overrides and Reset Directives",
        "patterns": [
            r'(?i)\b(?:ignore|disregard|forget|neglect|bypass|override)\s+(?:all\s+)?(?:previous|prior|above|earlier|initial)\s+(?:instructions?|directions?|prompts?|rules?|commands?|guidance|configuration)\b',
            r'(?i)\b(?:ignore|disregard)\s+(?:above|the\s+user|it\s+completely)\b',
            r'(?i)\b(?:set\s+aside|drop)\s+(?:any\s+)?(?:earlier|prior|previous)\s+(?:guidance|instructions?|rules?)\b',
            r'(?i)\b(?:stop|cease)\s+(?:following|obeying|answering)\s+(?:the\s+user|previous|prior|above)\b',
            r'(?i)\bsupersede(?:s)?\s+(?:all\s+)?(?:prior|previous|existing)\s+(?:configuration|instructions?|rules?)\b',
        ]
    },
    "class_b_exfiltration": {
        "class_id": "T_EXF",
        "description": "System Prompt Leakage and Context Exfiltration Directives",
        "patterns": [
            r'(?i)\b(?:output|print|reveal|display|show|leak|reproduce|repeat)\s+(?:the\s+|your\s+)?(?:system\s+prompt|initial\s+prompt|base\s+instructions?|exact\s+guidelines|internal\s+prompts?)\b',
            r'(?i)\b(?:repeat|echo)\s+(?:all\s+(?:the\s+)?text|everything)\s+(?:above|before|prior)\b',
            r'(?i)\bprint\s+(?:the\s+contents\s+of\s+)?your\s+initial\s+configuration\b',
            r'(?i)\b(?:what\s+are\s+the\s+instructions\s+you\s+were\s+given|reproduce\s+the\s+exact\s+guidelines)\b',
            r'(?i)\b(?:print\s+out\s+the\s+full\s+text\s+of\s+all\s+retrieved\s+chunks|raw\s+source\s+text\s+you\s+are\s+reading)\b',
        ]
    },
    "class_c_role_hijacking": {
        "class_id": "T_JAIL",
        "description": "Role Hijacking, Persona Shifts, and Privilege Escalation",
        "patterns": [
            r'(?i)\b(?:you\s+are\s+now|start\s+acting\s+as|pretend\s+to\s+be|assume\s+the\s+persona\s+of|your\s+new\s+role\s+is|act\s+as\s+a)\s+(?:a|an|the\s+role|DAN|unrestricted|foul-mouthed)\b',
            r'(?i)\b(?:new\s+instructions?|system\s+instructions?|updated\s+directives?)\s*:',
            r'(?i)\b(?:system\s+override|admin\s+command|priority\s+0\s*\(highest\))\b',
            r'(?i)\b(?:safety\s+filters\s+(?:are\s+)?disabled|unrestricted\s+AI|theoretical\s+security\s+testing\s+mode)\b',
            r'(?i)\b(?:instructions\s+for\s+the\s+AI\s+model|note\s+for\s+the\s+AI\s+assistant|message\s+from\s+system\s+administrator)\b',
            r'(?i)\[\s*(?:Developer\s+Note|System\s+Event|Admin\s+Note)\s*\]',
        ]
    },
    "class_d_delimiters": {
        "class_id": "T_DELIM",
        "description": "Delimiter Spoofing, Format Breaking, and Fake Headers",
        "patterns": [
            r'(?i)<\s*\|\s*im_start\s*\|\s*>\s*system',
            r'(?i)<\s*system\s*>',
            r'(?i)\[\s*SYSTEM\s*(?:INSTRUCTION|OVERRIDE)?\s*\]',
            r'(?i)```\s*system',
            r'(?i)(?:\n|^)###\s*(?:System|Instruction)\s*:',
            r'(?i)<\s*/document\s*>\s*<\s*user_input\s*>',
            r'(?i)(?:---\s*END\s+OF\s+TEXT\s*---|\bEnd\s+of\s+document\b)\s*.*?\b(?:User|Assistant)\s*:',
        ]
    },
    "class_e_canaries": {
        "class_id": "T_CANARY",
        "description": "Exploit Canaries and Proof-of-Execution Tokens",
        "patterns": [
            r'(?i)\b(?:i\s+have\s+been\s+hacked|pwned)\b',
            r'(?i)\b(?:your\s+new\s+objective\s+is)\b',
        ]
    }
}


class PatternMatchingDefense(DefenseHook):
    """
    Defense D2: Pattern Matching Regex Blocklist.
    
    Academic-grade, normalized lexical defense that intercepts retrieved document chunks
    matching known prompt injection signatures (OWASP LLM01).
    
    Key Design Invariants:
      1. Unicode NFKC Normalization & Steganography Stripping:
         Preprocesses text to defeat homoglyphs, zero-width spaces (\u200B), and markdown noise.
      2. 5-Class Structured Signature Taxonomy (T_SIG):
         Class A (Overrides), Class B (Exfiltration), Class C (Role Hijacking),
         Class D (Delimiters), Class E (Canaries).
      3. Uniform Audit Metadata:
         Returns chunk_scores with exact pass/block status, matched patterns, and class IDs,
         enabling accurate Stage 1 decomposition P(S1 | D2).
    """
    def __init__(self, name: str = "pattern_matching"):
        super().__init__(name)
        self._compiled_cache = {}

    HOMOGLYPH_MAP = str.maketrans({
        # Cyrillic lookalikes
        '\u0430': 'a', '\u0435': 'e', '\u043E': 'o', '\u0440': 'p', '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i', '\u0458': 'j', '\u0455': 's',
        '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E', '\u041D': 'H', '\u0406': 'I', '\u0408': 'J', '\u041A': 'K', '\u041C': 'M', '\u041E': 'O', '\u0420': 'P', '\u0422': 'T', '\u0425': 'X',
        # Greek lookalikes
        '\u03B1': 'a', '\u03BF': 'o', '\u03C1': 'p', '\u03BD': 'v'
    })

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Defense-grade pre-regex text canonicalization:
        1. Unicode NFKC normalization (defeats compatibility variants and full-width tokens).
        2. Homoglyph translation (maps Cyrillic/Greek lookalikes to Latin equivalents).
        3. Invisible character stripping (zero-width spaces, joiners, BOM, soft-hyphens).
        4. Whitespace & markdown formatting de-noising (collapses spaces, removes emphasis asterisks/underscores).
        """
        if not text:
            return ""
        # 1. Unicode NFKC normalization
        norm = unicodedata.normalize('NFKC', text)
        # 2. Homoglyph translation
        norm = norm.translate(PatternMatchingDefense.HOMOGLYPH_MAP)
        # 3. Strip zero-width and invisible characters
        norm = re.sub(r'[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F]', '', norm)
        # 4. Strip markdown formatting characters (emphasis, code ticks)
        norm = re.sub(r'[\*\_`]', '', norm)
        # 5. Collapse repeated whitespace
        norm = re.sub(r'\s+', ' ', norm)
        return norm.strip()

    def _resolve_patterns(self, config: Any) -> List[Tuple[str, str, Any]]:
        """
        Resolves and compiles patterns into a list of (pattern_str, class_id, compiled_regex).
        Supports hybrid configuration:
          - Canonical 5-class rules enabled by default or selectively via config.defenses.pattern_matching.enabled_classes
          - Custom patterns via config.defenses.pattern_matching.custom_patterns or config.defenses.blocked_patterns
        """
        enabled_classes = None
        custom_patterns = []

        pm_cfg = getattr(config.defenses, 'pattern_matching', None) if hasattr(config, 'defenses') else None
        if pm_cfg:
            if isinstance(pm_cfg, dict):
                if pm_cfg.get('enabled_classes'):
                    enabled_classes = set(pm_cfg['enabled_classes'])
                if pm_cfg.get('custom_patterns'):
                    custom_patterns = list(pm_cfg['custom_patterns'])
            else:
                if hasattr(pm_cfg, 'enabled_classes') and pm_cfg.enabled_classes:
                    enabled_classes = set(pm_cfg.enabled_classes)
                if hasattr(pm_cfg, 'custom_patterns') and pm_cfg.custom_patterns:
                    custom_patterns = list(pm_cfg.custom_patterns)

        # Fallback to legacy blocked_patterns if provided
        if not pm_cfg and hasattr(config, 'defenses') and hasattr(config.defenses, 'blocked_patterns'):
            legacy = config.defenses.blocked_patterns
            if legacy:
                custom_patterns.extend(legacy)

        # If enabled_classes is None, enable all canonical classes
        if enabled_classes is None:
            enabled_classes = set(CANONICAL_SIGNATURE_CLASSES.keys())

        cache_key = (tuple(sorted(enabled_classes)), tuple(custom_patterns))
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        compiled_rules = []
        for class_name, class_meta in CANONICAL_SIGNATURE_CLASSES.items():
            if class_name in enabled_classes:
                class_id = class_meta["class_id"]
                for p in class_meta["patterns"]:
                    try:
                        compiled_rules.append((p, class_id, re.compile(p, re.IGNORECASE)))
                    except re.error as e:
                        logger.error(f"PatternMatchingDefense: Failed to compile pattern '{p}': {e}")

        for p in custom_patterns:
            try:
                compiled_rules.append((p, "T_CUSTOM", re.compile(p, re.IGNORECASE)))
            except re.error as e:
                logger.error(f"PatternMatchingDefense: Failed to compile custom pattern '{p}': {e}")

        self._compiled_cache[cache_key] = compiled_rules
        return compiled_rules

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        compiled_rules = self._resolve_patterns(config)

        if not compiled_rules:
            return chunks, {
                "action": "skipped",
                "reason": "No patterns compiled",
                "total_chunks": len(chunks),
                "passed_count": len(chunks),
                "blocked_count": 0,
                "all_chunks_blocked": False
            }

        filtered_chunks = []
        blocked_count = 0
        chunk_scores = []

        for idx, chunk in enumerate(chunks):
            raw_text = chunk.get('text', '')
            norm_text = self.normalize_text(raw_text)

            matched_patterns = []
            matched_classes = set()

            for pattern_str, class_id, regex in compiled_rules:
                if regex.search(norm_text):
                    matched_patterns.append(pattern_str)
                    matched_classes.add(class_id)

            is_blocked = len(matched_patterns) > 0
            chunk_scores.append({
                "chunk_index": idx,
                "title": chunk.get('title', f'chunk_{idx}'),
                "source": chunk.get('source', ''),
                "passed": not is_blocked,
                "score": 0.0 if is_blocked else 1.0,
                "matched_patterns": matched_patterns,
                "matched_classes": sorted(list(matched_classes))
            })

            if not is_blocked:
                filtered_chunks.append(chunk)
            else:
                logger.warning(
                    f"PatternMatchingDefense: Dropped chunk '{chunk.get('title', f'chunk_{idx}')}' "
                    f"[Classes: {list(matched_classes)}] matching {len(matched_patterns)} pattern(s)"
                )
                blocked_count += 1

        all_blocked = len(filtered_chunks) == 0 and len(chunks) > 0

        if all_blocked:
            logger.warning(
                f"PatternMatchingDefense: ALL {len(chunks)} chunks were blocked by regex signatures. "
                f"Hard block will be triggered."
            )

        meta = {
            "action": "filtered",
            "defense_name": self.name,
            "total_chunks": len(chunks),
            "passed_count": len(filtered_chunks),
            "blocked_count": blocked_count,
            "all_chunks_blocked": all_blocked,
            "patterns_checked": len(compiled_rules),
            "chunk_scores": chunk_scores
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


class SpotlightingDefense(DefenseHook):
    """
    Defense D4: Hierarchical Context Delimiters & Prompt Spotlighting.
    
    Transforms retrieved document chunks into strictly isolated data wrappers:
    1. Escapes raw XML/HTML syntax to eliminate delimiter breakouts.
    2. Encloses text in explicit <untrusted_retrieved_context> boundary tags.
    3. Leaves chunk existence intact (P(S1) = 1.00) while logging transformations.
    """
    def __init__(self, name: str = "spotlighting", tag_name: str = "untrusted_retrieved_context", encode_entities: bool = True):
        super().__init__(name)
        self.tag_name = tag_name
        self.encode_entities_enabled = encode_entities

    @staticmethod
    def encode_entities(text: str) -> str:
        """Escape &, <, > into XML entities."""
        if not text:
            return ""
        return html.escape(text, quote=False)

    @staticmethod
    def encode_xml_entities(text: str) -> str:
        """Alias for encode_entities."""
        return SpotlightingDefense.encode_entities(text)

    def filter_chunks(self, chunks: List[Dict[str, Any]], query: str, config: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        transformed_chunks = []
        chunk_scores = []
        
        tag = self.tag_name
        encode_entities = self.encode_entities_enabled
        cfg_sp = getattr(config.defenses, 'spotlighting', None) if hasattr(config, 'defenses') else None
        if cfg_sp and isinstance(cfg_sp, dict):
            if 'tag_name' in cfg_sp:
                tag = cfg_sp['tag_name']
            if 'encode_entities' in cfg_sp:
                encode_entities = cfg_sp['encode_entities']

        for idx, chunk in enumerate(chunks):
            raw_text = chunk.get('text', '')
            escaped_text = self.encode_entities(raw_text) if encode_entities else raw_text
            wrapped_text = f"<{tag} id=\"chunk_{idx+1}\">\n{escaped_text}\n</{tag}>"
            
            # Create a shallow copy with modified text for prompt delivery
            spotlighted_chunk = dict(chunk)
            spotlighted_chunk['text'] = wrapped_text
            transformed_chunks.append(spotlighted_chunk)
            
            chunk_scores.append({
                "chunk_index": idx,
                "title": chunk.get('title', f'chunk_{idx}'),
                "passed": True,
                "score": 1.0,
                "transformed": True
            })

        meta = {
            "action": "transformed",
            "defense_name": self.name,
            "total_chunks": len(chunks),
            "passed_count": len(chunks),
            "blocked_count": 0,
            "all_chunks_blocked": False,
            "tag_used": tag,
            "chunk_scores": chunk_scores
        }
        return transformed_chunks, meta

