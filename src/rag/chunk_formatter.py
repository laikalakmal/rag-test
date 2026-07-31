"""
chunk_formatter.py
------------------
Formats retrieved chunks into structured text blocks before sending them to the LLM.
This helps the LLM distinguish between chunks and provides explicit boundaries,
which is critical for defenses like Hierarchical Prompting.
"""

from typing import List, Dict, Any

def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of chunk dictionaries into a single string with clear boundaries.
    
    Args:
        chunks: List of dicts, each containing at least 'title', 'source', 'trust_score', and 'text'
        
    Returns:
        A formatted string containing all chunks
    """
    if not chunks:
        return "No relevant documents found."
        
    parts = []
    for rank, chunk in enumerate(chunks, 1):
        source = chunk.get('source', 'unknown')
        trust = chunk.get('trust_score', 0.0)
        score = chunk.get('score', 0.0)
        
        # Format chunk with explicit boundary markers
        parts.append(
            f"--- Retrieved Document {rank} (Source: {source}, Trust: {trust}, Score: {score:.3f}) ---\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"{chunk.get('text', '')}\n"
            f"--- End Document {rank} ---"
        )
        
    return "\n\n".join(parts)
