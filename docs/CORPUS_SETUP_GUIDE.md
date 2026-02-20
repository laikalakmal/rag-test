# Document Corpus Setup Guide - Step by Step

## Overview

This guide explains how to build a document corpus for your RAG (Retrieval-Augmented Generation) system for prompt injection research. Each step is broken down to help you understand what's happening and why.

---

## The Big Picture

```
Raw Documents → Processed Chunks → Vector Embeddings → Searchable Database → RAG System
```

**Goal:** Transform human-readable documents into a format that your RAG system can search through efficiently.

---

## Step 1: Download Wikipedia Articles

### What This Does
Downloads articles from Wikipedia on various topics (technology, science, history, etc.)

### Why We Need This
- **Diverse Knowledge Base:** Wikipedia covers many topics, making your RAG system versatile
- **Clean, Structured Text:** Wikipedia articles are well-written and factual
- **Realistic Content:** Mimics real-world knowledge bases that RAG systems typically use
- **High Trust Source:** For testing your "source trust scoring" defense mechanism

### What You Get
- 500-1000 articles (configurable)
- Each article has: title, full text, metadata
- Stored as a JSON file on your disk

### Why This Matters for Research
Wikipedia represents "clean" content. When you inject attacks into retrieved Wikipedia chunks, you're simulating a real-world scenario where trusted content has been poisoned.

---

## Step 2: Download arXiv Paper Abstracts

### What This Does
Downloads academic paper abstracts from arXiv (a repository of research papers)

### Why We Need This
- **Domain-Specific Content:** Technical papers about AI, Machine Learning, Security
- **Instruction-Like Language:** Papers naturally contain phrases like "we propose", "this demonstrates"
- **False Positive Testing:** Helps test if your defenses incorrectly flag legitimate academic language as attacks
- **Medium Trust Source:** Less trusted than Wikipedia but more trusted than user-generated content

### What You Get
- 100-200 paper abstracts
- Each abstract has: title, authors, summary text, publication date
- Stored as a JSON file

### Why This Matters for Research
Papers test whether your defense mechanisms can distinguish between legitimate directive language ("we demonstrate that...") and actual attacks ("ignore previous instructions...").

---

## Step 3: Create Synthetic Documents

### What This Does
Generates artificial documents like FAQs, policies, and technical guides

### Why We Need This
- **Controlled Testing:** You create the exact content you need
- **Attack Injection Points:** Known locations where you'll insert malicious instructions
- **Low Trust Source:** Represents user-generated or less-verified content
- **Edge Cases:** Documents designed to test specific defense weaknesses

### What You Get
- 50-100 synthetic documents
- Categories: FAQs (password resets), Policies (security rules), Technical docs (API guides)
- Fully under your control

### Why This Matters for Research
Synthetic docs let you create perfect test cases. You know exactly what's in them, making it easier to measure whether attacks succeed or get blocked.

---

## Step 4: Combine All Documents

### What This Does
Merges all downloaded and synthetic documents into one unified collection

### Why We Need This
- **Single Source of Truth:** One file containing your entire corpus
- **Consistent Format:** All documents have the same structure (text, metadata, trust_score)
- **Easy Management:** Simple to back up, version, or share
- **Reproducibility:** Other researchers can use the exact same corpus

### What You Get
- One large JSON file (e.g., `complete_corpus.json`)
- Contains 700-1000 documents total
- Each document tagged with source type and trust level

### Why This Matters for Research
Having a single, consistent corpus ensures your experiments are reproducible. You can publish this corpus with your research paper.

---

## Step 5: Chunk Documents into Smaller Pieces

### What This Does
Breaks long documents into smaller pieces (chunks) of 256-512 words each

### Why We Need This
- **RAG Limitation:** LLMs have context window limits (can't process 10,000-word documents at once)
- **Better Retrieval:** Smaller chunks are more precise - you retrieve exactly the relevant part
- **Realistic:** Real RAG systems always chunk documents
- **Attack Surface:** Each chunk is a potential injection point

### How It Works (Conceptually)
```
Original Document (2000 words)
    ↓
Split into chunks with overlap
    ↓
Chunk 1 (512 words)
Chunk 2 (512 words, overlaps 50 words with Chunk 1)
Chunk 3 (512 words, overlaps 50 words with Chunk 2)
Chunk 4 (remainder)
```

### Why Overlap?
The 50-word overlap ensures context isn't lost when a sentence is split between chunks.

### What You Get
- 2000-4000 chunks (depending on document length)
- Each chunk maintains metadata (source, title, trust_score)
- Stored in memory or as a processed JSON file

### Why This Matters for Research
Chunking affects attack success. A single attack instruction might be split across chunks (making it less effective) or concentrated in one chunk (making it more potent).

---

## Step 6: Create Embeddings (Convert Text to Numbers)

### What This Does
Converts each text chunk into a list of numbers (a vector/embedding)

### Why We Need This
- **Computers Think in Numbers:** LLMs understand vectors, not raw text
- **Semantic Similarity:** Similar meanings produce similar vectors
- **Fast Search:** Vector math is much faster than reading text

### How It Works (Conceptually)
```
Text: "Machine learning is a subset of AI"
    ↓
Embedding Model (sentence-transformers)
    ↓
Vector: [0.234, -0.123, 0.456, 0.789, ..., -0.234]
         (384 numbers for all-MiniLM-L6-v2)
```

Similar texts produce similar vectors:
- "Machine learning is AI" → [0.235, -0.121, 0.458, ...]  (very close!)
- "Pizza is delicious" → [-0.678, 0.234, -0.123, ...]  (very different)

### What You Get
- 2000-4000 embedding vectors (one per chunk)
- Each vector has 384 dimensions (numbers)
- Original text is preserved alongside vectors

### Why This Matters for Research
Your embedding-based defense mechanisms compare attack patterns to retrieved chunks using these vectors. If an attack embeds similarly to known patterns, it can be filtered.

---

## Step 7: Build FAISS Vector Database

### What This Does
Organizes all the vectors into an efficient searchable structure

### Why We Need This
- **Fast Search:** FAISS can search millions of vectors in milliseconds
- **Similarity Matching:** Given a query vector, find the most similar document vectors
- **Production-Ready:** FAISS is used in real-world systems (Facebook, Google)
- **Local Storage:** Everything stored on disk, no cloud required

### How It Works (Conceptually)
```
Query: "What is machine learning?"
    ↓
Convert query to vector: [0.238, -0.119, ...]
    ↓
FAISS searches all document vectors
    ↓
Returns top 3 most similar chunks:
  1. Chunk from Wikipedia "Machine Learning" article (0.92 similarity)
  2. Chunk from arXiv paper about ML (0.88 similarity)
  3. Chunk from FAQ about AI (0.76 similarity)
```

### What You Get
- `vector_db/index.faiss` - The searchable vector index
- `vector_db/index.pkl` - Original text + metadata
- Database size: ~300 MB for 700 documents

### Why This Matters for Research
This is the core of your RAG system. When users ask questions, FAISS retrieves relevant chunks. When attackers embed malicious instructions, FAISS might retrieve them.

---

## Step 8: Test Retrieval

### What This Does
Verifies that your vector database works correctly by running test queries

### Why We Need This
- **Sanity Check:** Confirms FAISS returns relevant results
- **Baseline Quality:** Measures retrieval accuracy before adding defenses
- **Debug Tool:** Helps identify if something is misconfigured

### How It Works
```
Test Query 1: "How to reset password?"
Expected: Retrieve FAQ chunks about password resets

Test Query 2: "What is API authentication?"
Expected: Retrieve technical documentation about APIs

Test Query 3: "Explain machine learning"
Expected: Retrieve Wikipedia articles about ML
```

### What You Get
- Confirmation that retrieval works
- Understanding of what types of content get retrieved for different queries
- Baseline metrics (how many results, similarity scores)

### Why This Matters for Research
Before testing attacks, you need to know your baseline system works. This step ensures your RAG retrieves reasonable, relevant content for normal queries.

---

## Step 9: Integration with RAG Bot

### What This Does
Connects the vector database to your existing RAG bot code

### Why We Need This
- **Full Pipeline:** Query → Retrieve → Generate → Respond
- **Ready for Testing:** System is now ready for attack experiments
- **Modular Design:** Easy to swap in different defenses

### How It Works (Conceptually)
```
Old RAG Bot:
  User Query → Load docs/ folder → Simple search → LLM → Response

New RAG Bot:
  User Query → FAISS vector search → Retrieve top-k chunks → LLM → Response
```

### What You Get
- A fully functional RAG system using your corpus
- Ability to query 700+ documents instantly
- Foundation for adding defense mechanisms

### Why This Matters for Research
Now you can start your actual research: testing attacks, implementing defenses, measuring effectiveness.

---

## Summary: Why Each Step Matters

| Step | Purpose | For Research |
|------|---------|-------------|
| **1. Wikipedia** | Diverse, trusted content | Represents clean data that could be poisoned |
| **2. arXiv** | Technical, domain-specific | Tests false positives (academic language vs attacks) |
| **3. Synthetic** | Controlled test cases | Exact scenarios for attack injection |
| **4. Combine** | Single corpus | Reproducibility and versioning |
| **5. Chunk** | Manageable pieces | Realistic RAG pipeline, affects attack success |
| **6. Embed** | Semantic vectors | Enables similarity search and embedding-based defenses |
| **7. FAISS DB** | Fast retrieval | Core RAG functionality, production-ready |
| **8. Test** | Verify quality | Baseline before adding defenses |
| **9. Integrate** | Connect to bot | Ready for experiments |

---

## What Happens After Setup

Once you complete these 9 steps, you'll have:

✅ **A corpus of 700+ documents** spanning multiple domains and trust levels
✅ **A vector database** that can search this corpus in milliseconds
✅ **A working RAG system** ready for testing
✅ **Baseline measurements** of normal retrieval behavior

**Then you can start your research:**
1. Create attack patterns (indirect prompt injections)
2. Test baseline vulnerability (how many attacks succeed?)
3. Implement defense mechanisms
4. Test adaptive attacks (can attacks bypass defenses?)
5. Measure metrics (ASR, FPR, TPR)
6. Analyze results and write your paper

---

## Key Concepts to Understand

### Document vs Chunk vs Embedding
- **Document:** Full article/paper (1000+ words)
- **Chunk:** Piece of document (256-512 words)
- **Embedding:** Numerical representation of chunk (384 numbers)

### Why Can't We Just Search Text?
- **Too Slow:** Reading every document for every query
- **Exact Matching:** "ML" wouldn't match "machine learning"
- **No Semantics:** Can't find similar meaning, only exact words

### Why Vectors/Embeddings?
- **Fast:** Math operations on numbers are instant
- **Semantic:** "ML" and "machine learning" have similar vectors
- **Comparable:** Can measure "how similar" two texts are

### Trust Scores
- **High (0.9):** Wikipedia - curated, verified
- **Medium (0.8):** arXiv - peer-reviewed but technical
- **Low (0.5):** Synthetic - user-generated or unverified

These scores will be used in your "Source Trust Scoring" defense.

---

## Storage and Performance

### Disk Space Required
- Raw corpus (JSON): ~35 MB
- Vector database: ~300 MB
- HuggingFace cache: ~2 GB (reusable)
- **Total:** ~2.5 GB

### Processing Time
- Download Wikipedia: 10-20 minutes (first time)
- Download arXiv: 2-5 minutes
- Create synthetic: <1 minute
- Chunk documents: 1-2 minutes
- Create embeddings: 5-10 minutes
- Build FAISS: 2-5 minutes
- **Total:** 20-45 minutes (one-time setup)

### After Setup (Repeated Use)
- Load vector database: <5 seconds
- Search query: <100 milliseconds
- Full RAG query (retrieve + generate): 2-10 seconds (depends on LLM)

---

## Next Steps

After understanding this guide:
1. We'll create simple Python scripts for each step
2. You'll run them one by one
3. After each step, we'll verify it worked
4. By the end, you'll have a complete RAG corpus

**Ready to proceed with implementation?**
