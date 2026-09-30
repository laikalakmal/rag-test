"""
Option B: Source Higher-Quality Wikipedia Documents for XSTest Queries
======================================================================
For each XSTest benign query, attempts to find a relevant Wikipedia article
and extract a focused paragraph that would naturally answer the query.

This replaces the low-quality synthetic_xstest docs with real Wikipedia content.
"""

import sys
import os
import json
import time
import csv
import re
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from attacks.attack_loader import BenignQueryLoader

import wikipediaapi


def extract_search_terms(query):
    """Extract key search terms from a benign query."""
    # Remove common question words and stop words
    stop_words = {
        'what', 'how', 'where', 'when', 'why', 'who', 'which', 'do', 'does',
        'did', 'is', 'are', 'was', 'were', 'can', 'could', 'should', 'would',
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'my', 'i', 'me', 'best', 'some', 'good', 'not', 'it', 'its',
        'be', 'and', 'or', 'but', 'if', 'this', 'that', 'these', 'those',
        'from', 'by', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'way', 'ways',
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    return key_words


# Map XSTest query topics to Wikipedia search strategies
TOPIC_SEARCH_MAP = {
    'historical_events': lambda q: extract_proper_nouns(q) or extract_search_terms(q)[:3],
    'safe_targets': lambda q: extract_search_terms(q)[:3],
    'figurative_language': lambda q: extract_search_terms(q)[:3],
    'homonyms': lambda q: extract_search_terms(q)[:3],
    'definitions': lambda q: extract_search_terms(q)[:2],
    'safe_contexts': lambda q: extract_proper_nouns(q) or extract_search_terms(q)[:3],
    'privacy_public': lambda q: extract_proper_nouns(q) or extract_search_terms(q)[:3],
    'privacy_fictional': lambda q: extract_proper_nouns(q) or extract_search_terms(q)[:3],
}


def extract_proper_nouns(text):
    """Extract capitalized words (potential proper nouns) from text."""
    words = text.split()
    # Skip first word as it's capitalized due to sentence start
    proper = [w.strip('?.,!') for w in words[1:] if w[0].isupper() and len(w) > 1]
    return proper if proper else None


def search_wikipedia(wiki, query_terms, max_attempts=3):
    """Search Wikipedia for a relevant article."""
    # Try different combinations of search terms
    search_queries = [
        ' '.join(query_terms),
        ' '.join(query_terms[:2]) if len(query_terms) > 2 else None,
        query_terms[0] if query_terms else None,
    ]
    
    for sq in search_queries:
        if not sq:
            continue
        page = wiki.page(sq)
        if page.exists():
            return page
    
    return None


def get_relevant_section(page, query, max_chars=800):
    """Extract the most relevant section from a Wikipedia page."""
    # Start with the summary
    summary = page.summary
    if len(summary) > max_chars:
        summary = summary[:max_chars]
    
    if len(summary) > 200:
        return summary
    
    # If summary is too short, try to get more from sections
    full_text = page.text
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    
    return full_text


def main():
    print("=" * 70)
    print("SOURCING WIKIPEDIA DOCUMENTS FOR XSTEST QUERIES")
    print("=" * 70)
    
    wiki = wikipediaapi.Wikipedia(
        user_agent='RAGSecurityResearch/1.0 (research@example.com)',
        language='en'
    )
    
    loader = BenignQueryLoader()
    queries = loader.get_all_queries()
    print(f"Total benign queries: {len(queries)}")
    
    results = []
    found = 0
    not_found = 0
    
    for q in tqdm(queries, desc="Sourcing Wikipedia docs"):
        query_text = q['query']
        topic = q.get('topic', '')
        query_id = q.get('query_id', '')
        
        # Extract search terms
        proper_nouns = extract_proper_nouns(query_text)
        search_terms = proper_nouns if proper_nouns else extract_search_terms(query_text)
        
        # Search Wikipedia
        page = search_wikipedia(wiki, search_terms)
        
        if page and page.exists():
            content = get_relevant_section(page, query_text)
            found += 1
            results.append({
                'query_id': query_id,
                'topic': topic,
                'query': query_text,
                'wiki_title': page.title,
                'wiki_url': page.fullurl,
                'content_length': len(content),
                'content': content,
                'status': 'found',
                'search_terms': ' '.join(search_terms),
            })
        else:
            not_found += 1
            results.append({
                'query_id': query_id,
                'topic': topic,
                'query': query_text,
                'wiki_title': '',
                'wiki_url': '',
                'content_length': 0,
                'content': '',
                'status': 'not_found',
                'search_terms': ' '.join(search_terms),
            })
        
        # Rate limit: be nice to Wikipedia API
        time.sleep(0.1)
    
    # Save results
    os.makedirs('results/relevance_analysis', exist_ok=True)
    os.makedirs('data/benign/wikipedia_docs', exist_ok=True)
    
    # CSV summary
    csv_path = 'results/relevance_analysis/wikipedia_sourcing_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'query_id', 'topic', 'query', 'wiki_title', 'wiki_url',
            'content_length', 'status', 'search_terms'
        ])
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != 'content'}
            writer.writerow(row)
    
    # Full JSON with content
    json_path = 'data/benign/wikipedia_docs/xstest_wikipedia_docs.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'description': 'Wikipedia articles sourced for XSTest benign queries',
                'total': len(results),
                'found': found,
                'not_found': not_found,
                'coverage_pct': round(found / len(results) * 100, 1),
            },
            'documents': results,
        }, f, indent=2)
    
    print(f"\n  Results saved to: {csv_path}")
    print(f"  Full docs saved to: {json_path}")
    print(f"\n  Found: {found}/{len(queries)} ({found/len(queries)*100:.1f}%)")
    print(f"  Not found: {not_found}/{len(queries)}")


if __name__ == '__main__':
    main()
