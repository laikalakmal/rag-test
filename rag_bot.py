from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from llm_provider import get_llm

# Load documents
docs = []
for file in os.listdir("docs"):
    if file.endswith(".txt"):
        loader = TextLoader(f"docs/{file}")
        docs.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# Load LLM from provider module (set LLM_PROVIDER in .env: openrouter, huggingface, or local_t5)
llm = get_llm()

# Create QA chain using LCEL
retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Tiny RAG bot ready. Ask a question.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke(query)
    print("Bot:", result)
    print("\n\n")
