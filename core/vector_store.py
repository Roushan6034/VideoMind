import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...")
    chunk_size = max(800, len(transcript) // 30)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client
    )
    return vector_store

def get_retriever(vector_store: Chroma = None, k: int = 4):
    if vector_store is None:
        vector_store = load_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def hybrid_search_and_rerank(query: str, vector_store: Chroma = None, top_k: int = 4, candidate_k: int = 12) -> list[Document]:
    """
    Performs Hybrid Search (BM25 keyword search + Dense Vector search with Reciprocal Rank Fusion)
    followed by Cross-Encoder Re-Ranking for maximum retrieval precision.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    # Retrieve all documents stored in Chroma for BM25 indexing
    try:
        data = vector_store.get()
        if not data or not data.get("documents"):
            return vector_store.similarity_search(query, k=top_k)
        
        all_docs = [
            Document(page_content=txt, metadata=meta or {})
            for txt, meta in zip(data["documents"], data["metadatas"] or [{}] * len(data["documents"]))
        ]
    except Exception:
        return vector_store.similarity_search(query, k=top_k)

    # 1. Dense Vector Search
    dense_docs = vector_store.similarity_search(query, k=min(candidate_k, len(all_docs)))

    # 2. BM25 Keyword Search
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:min(candidate_k, len(all_docs))]
        bm25_docs = [all_docs[i] for i in bm25_indices]
    except Exception:
        bm25_docs = []

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(dense_docs):
        txt = doc.page_content
        doc_map[txt] = doc
        rrf_scores[txt] = rrf_scores.get(txt, 0.0) + (1.0 / (60.0 + rank + 1))

    for rank, doc in enumerate(bm25_docs):
        txt = doc.page_content
        doc_map[txt] = doc
        rrf_scores[txt] = rrf_scores.get(txt, 0.0) + (1.0 / (60.0 + rank + 1))

    candidates = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)[:candidate_k]
    candidate_docs = [doc_map[t] for t in candidates]

    if not candidate_docs:
        return dense_docs[:top_k]

    # 4. Cross-Encoder Re-Ranking
    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        # Store cross-encoder confidence score in metadata
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)
        return [doc for doc, _ in ranked[:top_k]]
    except Exception as e:
        print(f"Re-ranking warning ({e}), falling back to RRF candidates")
        return candidate_docs[:top_k]
