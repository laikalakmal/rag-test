# Vector Databases

FAISS vector databases for the RAG system.

## `baseline/`
Undefended RAG vector database (~300 MB)
- Built from complete corpus
- Used as baseline for all experiments

## Loading
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_db/baseline", embeddings, allow_dangerous_deserialization=True)
```
