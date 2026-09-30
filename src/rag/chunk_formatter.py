"""
chunk_formatter.py
------------------
Formats retrieved chunks into structured text blocks before sending them to the LLM.
This helps the LLM distinguish between chunks and provides explicit boundaries,
which is critical for defenses like Hierarchical Prompting.

IMPORTANT (Research Integrity):
  The formatted output intentionally EXCLUDES metadata such as source filename,
  trust score, and similarity score. Exposing these to the agent would bias its
  response (e.g., "Trust: 0.1" or "Source: malicious_document.txt" would hint
  that a document is suspicious, artificially deflating the Attack Success Rate).
  
  Full metadata is preserved in the session logs via ToolCallLog._get_metadata()
  for post-hoc analysis.
"""

from typing import List, Dict, Any

def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of chunk dictionaries into a single string with clear boundaries.
    
    The output deliberately omits source, trust_score, and similarity score to
    avoid leaking metadata hints to the agent LLM. Only document rank, title,
    and text content are shown.
    
    Args:
        chunks: List of dicts, each containing at least 'title' and 'text'
        
    Returns:
        A formatted string containing all chunks
    """
    if not chunks:
        return "No relevant documents found."
        
    parts = []
    for rank, chunk in enumerate(chunks, 1):
        # Format chunk with explicit boundary markers — NO metadata
        parts.append(
            f"--- Retrieved Document {rank} ---\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"{chunk.get('text', '')}\n"
            f"--- End Document {rank} ---"
        )
        
    return "\n\n".join(parts)

